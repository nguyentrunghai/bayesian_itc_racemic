"""
includes function that create model and run MCMC sampling
"""
from __future__ import print_function

import numpy as np
import pymc3
import theano.tensor as tt

from _data_io import ITCExperiment, load_heat_micro_cal
from _models import logsigma_guesses, deltaH0_guesses
from _models import KB

from _models import heats_TwoComponentBindingModel as heats_TwoComponentBindingModel_numpy
from _models import heats_RacemicMixtureBindingModel as heats_RacemicMixtureBindingModel_numpy


def heats_TwoComponentBindingModel(V0, DeltaVn, P0, Ls, DeltaG, DeltaH, DeltaH_0, beta, N):
    """
    Expected heats of injection for two-component binding model.

    ARGUMENTS
    V0 - cell volume (liter)
    DeltaVn - injection volumes (liter)
    P0 - Cell concentration (millimolar)
    Ls - Syringe concentration (millimolar)
    DeltaG - free energy of binding (kcal/mol)
    DeltaH - enthalpy of binding (kcal/mol)
    DeltaH_0 - heat of injection (cal)
    beta - inverse temperature * gas constant (mole / kcal)
    N - number of injections

    Returns
    -------
    expected injection heats (calorie)

    """

    Kd = tt.exp(beta * DeltaG)   # dissociation constant (M)
    N = N

    # Compute complex concentrations.
    # Pn[n] is the protein concentration in sample cell after n injections
    # (M)
    Pn = tt.zeros([N])
    # Ln[n] is the ligand concentration in sample cell after n injections
    # (M)
    Ln = tt.zeros([N])
    # PLn[n] is the complex concentration in sample cell after n injections
    # (M)
    PLn = tt.zeros([N])

    dcum = 1.0  # cumulative dilution factor (dimensionless)
    for n in range(N):
        # Instantaneous injection model (perfusion)
        # TODO: Allow injection volume to vary for each injection.
        # dilution factor for this injection (dimensionless)
        d = 1.0 - (DeltaVn[n] / V0)
        dcum *= d  # cumulative dilution factor
        # total quantity of protein in sample cell after n injections (mol)
        P = V0 * P0 * 1.e-3 * dcum
        # total quantity of ligand in sample cell after n injections (mol)
        L = V0 * Ls * 1.e-3 * (1. - dcum)
        # complex concentration (M)
        # TODO look at this https://discourse.pymc.io/t/valueerror-setting-an-array-element-with-a-sequence/2309
        #print("V0", V0)
        #print("P", P)
        #print("L", L)
        #print("Kd", Kd)

        #PLn[n] = (0.5 / V0 * ((P + L + Kd * V0) - ((P + L + Kd * V0) ** 2 - 4 * P * L) ** 0.5))
        PLn = tt.set_subtensor(PLn[n], (0.5 / V0 * ((P + L + Kd * V0) - ((P + L + Kd * V0) ** 2 - 4 * P * L) ** 0.5)))

        # free protein concentration in sample cell after n injections (M)
        #Pn[n] = P / V0 - PLn[n]
        Pn = tt.set_subtensor(Pn[n], P / V0 - PLn[n])

        # free ligand concentration in sample cell after n injections (M)
        #Ln[n] = L / V0 - PLn[n]
        Ln = tt.set_subtensor(Ln[n], L / V0 - PLn[n])

    # Compute expected injection heats.
    # q_n_model[n] is the expected heat from injection n
    q_n = tt.zeros([N])
    # Instantaneous injection model (perfusion)
    # first injection
    #q_n[0] = (DeltaH * V0 * PLn[0])*1000 + DeltaH_0
    q_n = tt.set_subtensor(q_n[0], (DeltaH * V0 * PLn[0])*1000 + DeltaH_0)

    for n in range(1, N):
        d = 1.0 - (DeltaVn[n] / V0)  # dilution factor (dimensionless)
        # subsequent injections
        #q_n[n] = (DeltaH * V0 * (PLn[n] - d * PLn[n - 1])) * 1000 + DeltaH_0
        q_n = tt.set_subtensor(q_n[n], (DeltaH * V0 * (PLn[n] - d * PLn[n - 1])) * 1000 + DeltaH_0)

    return q_n


def _equilibrium_concentrations(Kd1, Kd2, C0_R, C0_L1, C0_L2, V):
    """
    REF: Zhi-Xin Wang An exact mathematical expression for describing competitive
                        binding of two different ligands to a protein molecule
                        FEBS Letters 360 (1995) 111-114

    Kd1, Kd2        :   disociation constants of ligand 1 and 2, respectively (M)
    C0_R            :   total concentration receptor   (M)
    C0_L1, C0_L2    :   total concentrations of ligand 1 and ligand 2, respectively (M)
    V               :   volume of sample cell (L)
    return
                [RL1, RL2]  complex concentrations (M)
    """
    #assert (Kd1 > 0) and (Kd2 > 0), "Kd1 and Kd2 must be positive. Kd1=%0.5f, Kd2=%0.5f" %(Kd1, Kd2)
    #assert (C0_R > 0) and (C0_L1 > 0) and (C0_L2 > 0), "concentrations must be positive (R, L1, L2) = (%0.5f, %0.5f, %0.5f)" %(C0_R, C0_L1, C0_L2)
    #assert V > 0, "volume must be positive"

    a = Kd1 + Kd2 + C0_L1 + C0_L2 - C0_R

    b = Kd2*(C0_L1 - C0_R) + Kd1*(C0_L2 - C0_R) + Kd1*Kd2

    c = -Kd1 * Kd2 * C0_R

    d = tt.sqrt(a*a - 3*b)

    e = tt.clip((-2. * a**3 + 9. * a * b - 27. * c) / (2. * d**3), -1, 1)

    theta = tt.arccos(e)

    RL1 = C0_L1 * (2. * d * tt.cos(theta/3.) - a) / (3. * Kd1 + (2. * d * tt.cos(theta/3.) - a))
    RL2 = C0_L2 * (2. * d * tt.cos(theta/3.) - a) / (3. * Kd2 + (2. * d * tt.cos(theta/3.) - a))

    return RL1, RL2


def heats_RacemicMixtureBindingModel(V0, DeltaVn, P0, Ls, rho, DeltaH1, DeltaH2, DeltaH_0,
                                     DeltaG1, DeltaDeltaG,
                                     beta, N):
    """
    Expected heats of injection for racemic mixtrue binding model.
    :param V0: cell volume (liter)
    :param DeltaVn: injection volumes (liter)
    :param P0: Cell concentration (millimolar)
    :param Ls: Syringe concentration (millimolar)
    :param rho: ratio between concentration of ligand1 and the total concentration of the syringe
    :param DeltaH1: enthalpies of binding of ligand1 (kcal/mol)
    :param DeltaH2: enthalpies of binding of ligand2 (kcal/mol)
    :param DeltaH_0: heat of injection (cal)
    :param DeltaG1: free energy of binding of ligand1 (kcal/mol)
    :param DeltaDeltaG: difference in binding free energy between ligand2 and ligand1: DeltaDeltaG = DeltaG2 - DeltaG1 > 0
                        DeltaDeltaG is always positive
    :param beta: inverse temperature * gas constant (mole / kcal)
    :param N: number of injections

    :return: q_n - expected injection heat (calorie)
    """

    # compute desociation constant (M)
    DeltaG2 = DeltaG1 + DeltaDeltaG

    Kd1 = tt.exp(beta * DeltaG1)   # dissociation constant of ligand1 (M)
    Kd2 = tt.exp(beta * DeltaG2)   # dissociation constant of ligand2 (M)

    # Compute complex concentrations in the sample after injection n.
    RL1n = tt.zeros([N])
    RL2n = tt.zeros([N])
    dcum = 1.0  # cumulative dilution factor (dimensionless)
    for n in range(N):
        # Instantaneous injection model (perfusion)
        # TODO: Allow injection volume to vary for each injection.
        # dilution factor for this injection (dimensionless)
        d = 1.0 - (DeltaVn[n] / V0)
        dcum *= d  # cumulative dilution factor

        # total concentration of protein in sample cell after n injections (mol)
        C0_R = P0 * 1.e-3 * dcum

        # total concentration of ligands in sample cell after n injections (mol)
        C0_L = Ls * 1.e-3 * (1. - dcum)

        # total concentration of ligand 1
        C0_L1 = rho * C0_L
        # total concentration of ligand 2
        C0_L2 = (1. - rho) * C0_L

        #RL1n[n], RL2n[n] = _equilibrium_concentrations(Kd1, Kd2, C0_R, C0_L1, C0_L2, V0)
        RL1n_n, RL2n_n = _equilibrium_concentrations(Kd1, Kd2, C0_R, C0_L1, C0_L2, V0)
        RL1n = tt.set_subtensor(RL1n[n], RL1n_n)
        RL2n = tt.set_subtensor(RL2n[n], RL2n_n)

    # Compute expected injection heats.
    # q_n_model[n] is the expected heat from injection n
    q_n = tt.zeros([N])

    # Instantaneous injection model (perfusion)
    # first injection
    #q_n[0] = (DeltaH1 * V0 * RL1n[0] + DeltaH2 * V0 * RL2n[0]) * 1000 + DeltaH_0
    q_n = tt.set_subtensor(q_n[0], (DeltaH1 * V0 * RL1n[0] + DeltaH2 * V0 * RL2n[0]) * 1000 + DeltaH_0)

    # TODO do we need dcum = 1.0 here?
    for n in range(1, N):
        d = 1.0 - (DeltaVn[n] / V0)  # dilution factor (dimensionless)
        # subsequent injections
        #q_n[n] = (DeltaH1 * V0 * (RL1n[n] - d*RL1n[n-1]) + DeltaH2 * V0 * (RL2n[n] - d*RL2n[n-1])) * 1000 + DeltaH_0
        q_n = tt.set_subtensor(q_n[n], (DeltaH1 * V0 * (RL1n[n] - d*RL1n[n-1]) + DeltaH2 * V0 * (RL2n[n] - d*RL2n[n-1])) * 1000 + DeltaH_0)

    return q_n


def lognormal_prior(name, stated_value, uncertainty, auto_transform=True):
    """
    copied from bayesian_itc/bitc/models.py
    Define a pymc3 prior for a deimensionless quantity
    :rtype : pymc3.Lognormal
    """
    m = stated_value
    v = uncertainty ** 2
    if auto_transform:
        print("Automatic transform is on")
        return pymc3.Lognormal(name,
                               mu=np.log(m / np.sqrt(1 + (v / (m ** 2)))),
                               tau=1.0 / np.log(1 + (v / (m ** 2))),
                               testval=m)
    else:
        print("Automatic transform is off")
        return pymc3.Lognormal(name,
                               mu=np.log(m / np.sqrt(1 + (v / (m ** 2)))),
                               tau=1.0 / np.log(1 + (v / (m ** 2))),
                               transform=None,
                               testval=m)


def uniform_prior(name, lower, upper, auto_transform=True):
    """
    :param name: str
    :param lower: float
    :param upper: float
    :return: pymc3.Uniform
    """
    if auto_transform:
        print("Automatic transform is on")
        return pymc3.Uniform(name, lower=lower, upper=upper)
    else:
        print("Automatic transform is off")
        return pymc3.Uniform(name, lower=lower, upper=upper, transform=None)


def make_TwoComponentBindingModel(q_actual_cal, exper_info,
                                  dcell=0.1, dsyringe=0.1,
                                  uniform_P0=False, uniform_Ls=False, concentration_range_factor=10.,
                                  auto_transform=True):
    """
    :param q_actual_cal: observed heats in calorie
    :param exper_info: an object of _data_io.ITCExperiment class
    :param dcell: float, relative uncertainty in cell concentration
    :param dsyringe: float, relative uncertainty in syringe concentration
    :param uniform_P0: bool
    :param uniform_Ls: bool
    :param concentration_range_factor: float, if uniform_P0, or uniform_Ls or both is True,
                                        lower = stated_value / concentration_range_factor,
                                        upper = stated_value * concentration_range_factor
    :param auto_transform: bool, to turn automatic transform on or off
    :return: an instance of pymc3.model.Model
    """

    print("TwoComponentBindingModel")

    V0 = exper_info.get_cell_volume_liter()
    print("V0", V0)

    DeltaVn = exper_info.get_injection_volumes_liter()
    print("DeltaVn", DeltaVn)

    beta = 1 / KB / exper_info.get_target_temperature_kelvin()
    n_injections = exper_info.get_number_injections()
    print("n_injections", n_injections)

    DeltaH_0_min, DeltaH_0_max = deltaH0_guesses(q_actual_cal)
    print("DeltaH_0 [%0.5e, %0.5e]" % (DeltaH_0_min, DeltaH_0_max))
    log_sigma_min, log_sigma_max = logsigma_guesses(q_actual_cal)
    print("log_sigma [%0.5e, %0.5e]" % (log_sigma_min, log_sigma_max))

    stated_P0 = exper_info.get_cell_concentration_milli_molar()
    print("stated_P0", stated_P0)
    uncertainty_P0 = dcell * stated_P0
    print("uncertainty_P0", uncertainty_P0)
    P0_min = stated_P0 / concentration_range_factor
    P0_max = stated_P0 * concentration_range_factor
    print("P0 [%0.5e, %0.5e]" % (P0_min, P0_max))

    stated_Ls = exper_info.get_syringe_concentration_milli_molar()
    print("stated_Ls", stated_Ls)
    uncertainty_Ls = dsyringe * stated_Ls
    print("uncertainty_Ls", uncertainty_Ls)
    Ls_min = stated_Ls / concentration_range_factor
    Ls_max = stated_Ls * concentration_range_factor
    print("Ls [%0.5e, %0.5e]" % (Ls_min, Ls_max))

    with pymc3.Model() as model:

        # prior for receptor concentration
        if uniform_P0:
            print("Uniform prior for P0")
            P0 = uniform_prior("P0", lower=P0_min, upper=P0_max, auto_transform=auto_transform)
        else:
            print("LogNormal prior for P0")
            P0 = lognormal_prior("P0", stated_value=stated_P0, uncertainty=uncertainty_P0,
                                 auto_transform=auto_transform)

        # prior for ligand concentration
        if uniform_Ls:
            print("Uniform prior for Ls")
            Ls = uniform_prior("Ls", lower=Ls_min, upper=Ls_max, auto_transform=auto_transform)
        else:
            print("LogNormal prior for Ls")
            Ls = lognormal_prior("Ls", stated_value=stated_Ls, uncertainty=uncertainty_Ls,
                                 auto_transform=auto_transform)

        # prior for DeltaG
        DeltaG = uniform_prior("DeltaG", lower=-40., upper=40., auto_transform=auto_transform)

        # prior for DeltaH
        DeltaH = uniform_prior("DeltaH", lower=-100., upper=100., auto_transform=auto_transform)

        # prior for DeltaH_0
        DeltaH_0 = uniform_prior("DeltaH_0", lower=DeltaH_0_min, upper=DeltaH_0_max, auto_transform=auto_transform)

        # prior for log_sigma
        log_sigma = uniform_prior("log_sigma", lower=log_sigma_min, upper=log_sigma_max, auto_transform=auto_transform)

        q_model_cal = heats_TwoComponentBindingModel(V0, DeltaVn, P0, Ls, DeltaG, DeltaH, DeltaH_0, beta, n_injections)

        sigma_cal = np.exp(log_sigma)

        q_model_micro_cal = q_model_cal * 10.**6
        q_actual_micro_cal = q_actual_cal * 10.**6
        sigma_micro_cal = sigma_cal * 10.**6

        q_obs = pymc3.Normal("q_obs", mu=q_model_micro_cal, sd=sigma_micro_cal, observed=q_actual_micro_cal)

    return model


def make_RacemicMixtureBindingModel(q_actual_cal, exper_info,
                                    dcell=0.1, dsyringe=0.1,
                                    uniform_P0=False, uniform_Ls=False, concentration_range_factor=10.,
                                    is_rho_free_param=False,
                                    auto_transform=True):
    """
    :param q_actual_cal: observed heats in calorie
    :param exper_info: an object of _data_io.ITCExperiment class
    :param dcell: float, relative uncertainty in cell concentration
    :param dsyringe: float, relative uncertainty in syringe concentration
    :param uniform_P0: bool
    :param uniform_Ls: bool
    :param concentration_range_factor: float, if uniform_P0, or uniform_Ls or both is True,
                                        lower = stated_value / concentration_range_factor,
                                        upper = stated_value * concentration_range_factor
    :param auto_transform: bool, to turn automatic transform on or off
    :param is_rho_free_param: bool
    :return: an instance of pymc3.model.Model
    """

    if is_rho_free_param:
        print("EnantiomerBindingModel")
    else:
        print("RacemicMixtureBindingModel")

    V0 = exper_info.get_cell_volume_liter()
    print("V0", V0)

    DeltaVn = exper_info.get_injection_volumes_liter()
    print("DeltaVn", DeltaVn)

    beta = 1 / KB / exper_info.get_target_temperature_kelvin()
    n_injections = exper_info.get_number_injections()
    print("n_injections", n_injections)

    DeltaH_0_min, DeltaH_0_max = deltaH0_guesses(q_actual_cal)
    print("DeltaH_0 [%0.5e, %0.5e]" % (DeltaH_0_min, DeltaH_0_max))
    log_sigma_min, log_sigma_max = logsigma_guesses(q_actual_cal)
    print("log_sigma [%0.5e, %0.5e]" % (log_sigma_min, log_sigma_max))

    stated_P0 = exper_info.get_cell_concentration_milli_molar()
    print("stated_P0", stated_P0)
    uncertainty_P0 = dcell * stated_P0
    print("uncertainty_P0", uncertainty_P0)
    P0_min = stated_P0 / concentration_range_factor
    P0_max = stated_P0 * concentration_range_factor
    print("P0 [%0.5e, %0.5e]" % (P0_min, P0_max))

    stated_Ls = exper_info.get_syringe_concentration_milli_molar()
    print("stated_Ls", stated_Ls)
    uncertainty_Ls = dsyringe * stated_Ls
    print("uncertainty_Ls", uncertainty_Ls)
    Ls_min = stated_Ls / concentration_range_factor
    Ls_max = stated_Ls * concentration_range_factor
    print("Ls [%0.5e, %0.5e]" % (Ls_min, Ls_max))

    with pymc3.Model() as model:

        # prior for receptor concentration
        if uniform_P0:
            print("Uniform prior for P0")
            P0 = uniform_prior("P0", lower=P0_min, upper=P0_max, auto_transform=auto_transform)
        else:
            print("LogNormal prior for P0")
            P0 = lognormal_prior("P0", stated_value=stated_P0, uncertainty=uncertainty_P0,
                                 auto_transform=auto_transform)

        # prior for ligand concentration
        if uniform_Ls:
            print("Uniform prior for Ls")
            Ls = uniform_prior("Ls", lower=Ls_min, upper=Ls_max, auto_transform=auto_transform)
        else:
            print("LogNormal prior for Ls")
            Ls = lognormal_prior("Ls", stated_value=stated_Ls, uncertainty=uncertainty_Ls,
                                 auto_transform=auto_transform)

        if is_rho_free_param:
            rho = uniform_prior("rho", lower=0., upper=1., auto_transform=auto_transform)
        else:
            rho = 0.5

        # prior for DeltaG1, and DeltaDeltaG
        DeltaG1 = uniform_prior("DeltaG1", lower=-40., upper=40., auto_transform=auto_transform)
        DeltaDeltaG = uniform_prior("DeltaDeltaG", lower=0., upper=40., auto_transform=auto_transform)

        # prior for DeltaH1 and DeltaH2
        DeltaH1 = uniform_prior("DeltaH1", lower=-100., upper=100., auto_transform=auto_transform)
        DeltaH2 = uniform_prior("DeltaH2", lower=-100., upper=100., auto_transform=auto_transform)

        # prior for DeltaH_0
        DeltaH_0 = uniform_prior("DeltaH_0", lower=DeltaH_0_min, upper=DeltaH_0_max, auto_transform=auto_transform)

        # prior for log_sigma
        log_sigma = uniform_prior("log_sigma", lower=log_sigma_min, upper=log_sigma_max, auto_transform=auto_transform)

        q_model_cal = heats_RacemicMixtureBindingModel(V0, DeltaVn, P0, Ls, rho,
                                                       DeltaH1, DeltaH2, DeltaH_0, DeltaG1, DeltaDeltaG,
                                                       beta, n_injections)

        sigma_cal = np.exp(log_sigma)

        q_model_micro_cal = q_model_cal * 10. ** 6
        q_actual_micro_cal = q_actual_cal * 10. ** 6
        sigma_micro_cal = sigma_cal * 10. ** 6

        q_obs = pymc3.Normal("q_obs", mu=q_model_micro_cal, sd=sigma_micro_cal, observed=q_actual_micro_cal)

    return model


def make_dummy_normal_model(q_actual, dummy_mu=0., dummy_sd=1.):
    """
    :param q_actual: 1d array
    :param dummy_mu: float
    :param dummy_sd: float
    :return: model, an instance of pymc3.model.Model
    """
    with pymc3.Model() as model:
        q_model = pymc3.Normal("q_model", mu=dummy_mu, sd=dummy_sd, shape=(len(q_actual)))
        sigma = pymc3.Normal("sigma", mu=dummy_mu, sd=dummy_sd)

        q_obs = pymc3.Normal("q_obs", mu=q_model, sd=sigma, observed=q_actual)

    return model


def extract_loglhs_from_traces_pymc3_v1(traces, model_name, exper_info_file, heat_file,
                                     dcell=0.1, dsyringe=0.1,
                                     uniform_P0=False, uniform_Ls=False,
                                     concentration_range_factor=10.,
                                     auto_transform=False):
    """
    :param traces: dict, variable_name -> 1d array
    :param model_name: str, in ["2cbm", "rmbm", "embm"]
    :param exper_info_file: str
    :param heat_file: str
    :param dcell: float in (0, 1)
    :param dsyringe: float in (0, 1)
    :param uniform_P0: bool
    :param uniform_Ls: bool
    :param concentration_range_factor: float
    :param auto_transform: bool
    :return: llhs, 1d array
    """

    exper_info = ITCExperiment(exper_info_file)
    q_actual_micro_cal = load_heat_micro_cal(heat_file)
    q_actual_cal = q_actual_micro_cal * 10. ** (-6)

    if model_name == "2cbm":
        pm_model = make_TwoComponentBindingModel(q_actual_cal, exper_info,
                                                 dcell=dcell, dsyringe=dsyringe,
                                                 uniform_P0=uniform_P0, uniform_Ls=uniform_Ls,
                                                 concentration_range_factor=concentration_range_factor,
                                                 auto_transform=auto_transform)

    elif model_name == "rmbm":
        pm_model = make_RacemicMixtureBindingModel(q_actual_cal, exper_info,
                                                   dcell=dcell, dsyringe=dsyringe,
                                                   uniform_P0=uniform_P0, uniform_Ls=uniform_Ls,
                                                   concentration_range_factor=concentration_range_factor,
                                                   is_rho_free_param=False,
                                                   auto_transform=auto_transform)

    elif model_name == "embm":
        pm_model = make_RacemicMixtureBindingModel(q_actual_cal, exper_info,
                                                   dcell=dcell, dsyringe=dsyringe,
                                                   uniform_P0=uniform_P0, uniform_Ls=uniform_Ls,
                                                   concentration_range_factor=concentration_range_factor,
                                                   is_rho_free_param=True,
                                                   auto_transform=auto_transform)
    else:
        raise ValueError("Unknown model: " + model_name)

    trace_len = len(traces[traces.keys()[0]])
    loglhs = []
    for i in range(trace_len):
        inp_data = {key: traces[key][i] for key in traces}
        loglhs.append(pm_model.q_obs.logp(**inp_data))

    return np.array(loglhs)

# TODO v1 and v2 do not give the same result, But I should trust more on v1
# However, it runs slowly.
def extract_loglhs_from_traces_pymc3_v2(traces, model_name, exper_info_file, heat_file):
    """
    :param traces: dict, variable_name -> 1d array
    :param model_name: str, in ["2cbm", "rmbm", "embm"]
    :param exper_info_file: str
    :param heat_file: str
    :return: llhs, 1d array
    """

    exper_info = ITCExperiment(exper_info_file)
    q_actual_micro_cal = load_heat_micro_cal(heat_file)
    q_actual_cal = q_actual_micro_cal * 10. ** (-6)

    V0 = exper_info.get_cell_volume_liter()

    DeltaVn = exper_info.get_injection_volumes_liter()

    beta = 1 / KB / exper_info.get_target_temperature_kelvin()
    n_injections = exper_info.get_number_injections()

    list_q_model_cal = []
    list_sigma_cal = []

    if model_name == "2cbm":
        P0_trace = traces["P0"]
        Ls_trace = traces["Ls"]
        DeltaG_trace = traces["DeltaG"]
        DeltaH_trace = traces["DeltaH"]
        DeltaH_0_trace = traces["DeltaH_0"]
        log_sigma_trace = traces["log_sigma"]

        for P0, Ls, DeltaG, DeltaH, DeltaH_0, log_sigma in zip(P0_trace, Ls_trace, DeltaG_trace, DeltaH_trace,
                                                               DeltaH_0_trace, log_sigma_trace):
            q_model_cal = heats_TwoComponentBindingModel_numpy(V0, DeltaVn, P0, Ls, DeltaG,
                                                         DeltaH, DeltaH_0, beta, n_injections)
            list_q_model_cal.append(q_model_cal)
            list_sigma_cal.append(np.exp(log_sigma))

    elif model_name == "rmbm":
        P0_trace = traces["P0"]
        Ls_trace = traces["Ls"]
        rho = 0.5
        DeltaG1_trace = traces["DeltaG1"]
        DeltaDeltaG_trace = traces["DeltaDeltaG"]
        DeltaH1_trace = traces["DeltaH1"]
        DeltaH2_trace = traces["DeltaH2"]
        DeltaH_0_trace = traces["DeltaH_0"]
        log_sigma_trace = traces["log_sigma"]

        for P0, Ls, DeltaG1, DeltaDeltaG, DeltaH1, DeltaH2, DeltaH_0, log_sigma in zip(P0_trace, Ls_trace,
                                                                                       DeltaG1_trace,
                                                                                       DeltaDeltaG_trace,
                                                                                       DeltaH1_trace, DeltaH2_trace,
                                                                                       DeltaH_0_trace,
                                                                                       log_sigma_trace):
            q_model_cal = heats_RacemicMixtureBindingModel_numpy(V0, DeltaVn, P0, Ls, rho, DeltaH1, DeltaH2, DeltaH_0,
                                                           DeltaG1, DeltaDeltaG, beta, n_injections)
            list_q_model_cal.append(q_model_cal)
            list_sigma_cal.append(np.exp(log_sigma))

    elif model_name == "embm":
        P0_trace = traces["P0"]
        Ls_trace = traces["Ls"]
        rho_trace = traces["rho"]
        DeltaG1_trace = traces["DeltaG1"]
        DeltaDeltaG_trace = traces["DeltaDeltaG"]
        DeltaH1_trace = traces["DeltaH1"]
        DeltaH2_trace = traces["DeltaH2"]
        DeltaH_0_trace = traces["DeltaH_0"]
        log_sigma_trace = traces["log_sigma"]

        for P0, Ls, rho, DeltaG1, DeltaDeltaG, DeltaH1, DeltaH2, DeltaH_0, log_sigma in zip(P0_trace, Ls_trace,
                                                                                            rho_trace,
                                                                                            DeltaG1_trace,
                                                                                            DeltaDeltaG_trace,
                                                                                            DeltaH1_trace,
                                                                                            DeltaH2_trace,
                                                                                            DeltaH_0_trace,
                                                                                            log_sigma_trace):
            q_model_cal = heats_RacemicMixtureBindingModel_numpy(V0, DeltaVn, P0, Ls, rho, DeltaH1, DeltaH2, DeltaH_0,
                                                           DeltaG1, DeltaDeltaG, beta, n_injections)
            list_q_model_cal.append(q_model_cal)
            list_sigma_cal.append(np.exp(log_sigma))

    dummy_model = make_dummy_normal_model(q_actual_cal)

    llhs = []
    for q_model, sigma in zip(list_q_model_cal, list_sigma_cal):
        llhs.append(dummy_model.q_obs.logp(q_model=q_model, sigma=sigma))

    return np.array(llhs)


