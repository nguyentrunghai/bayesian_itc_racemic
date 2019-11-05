"""
includes function that create model and run MCMC sampling
"""

import numpy as np
import pymc3
import theano.tensor as tt

from _models import logsigma_guesses, deltaH0_guesses
from _models import KB


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

    d = np.sqrt(a*a - 3*b)

    e = np.clip((-2.*a**3 + 9.*a*b - 27.*c) / (2.*d**3), a_min=-1, a_max=1)

    theta = np.arccos(e)

    RL1 = C0_L1*(2.*d*np.cos(theta/3.) - a) / (3.*Kd1 + (2.*d*np.cos(theta/3.) - a))
    RL2 = C0_L2*(2.*d*np.cos(theta/3.) - a) / (3.*Kd2 + (2.*d*np.cos(theta/3.) - a))

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

    assert 0 < rho < 1, "rho out of range: %0.5f" %rho

    # compute desociation constant (M)
    DeltaG2 = DeltaG1 + DeltaDeltaG

    Kd1 = np.exp(beta * DeltaG1)   # dissociation constant of ligand1 (M)
    Kd2 = np.exp(beta * DeltaG2)   # dissociation constant of ligand2 (M)

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


def lognormal_prior(name, stated_value, uncertainty):
    """
    copied from bayesian_itc/bitc/models.py
    Define a pymc3 prior for a deimensionless quantity
    :rtype : pymc3.Lognormal
    """
    m = stated_value
    v = uncertainty ** 2
    return pymc3.Lognormal(name,
                           mu=np.log(m / np.sqrt(1 + (v / (m ** 2)))),
                           tau=1.0 / np.log(1 + (v / (m ** 2))))


def uniform_prior(name, lower, upper):
    """
    :param name: str
    :param lower: float
    :param upper: float
    :return: pymc3.Uniform
    """
    return pymc3.Uniform(name, lower=lower, upper=upper)


def make_TwoComponentBindingModel(q_actual_cal, exper_info,
                                  dcell=0.1, dsyringe=0.1,
                                  uniform_P0=False, uniform_Ls=False, concentration_range_factor=10):
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
    :return: an instance of pymc3.model.Model
    """

    V0 = exper_info.get_cell_volume_liter()
    DeltaVn = exper_info.get_injection_volumes_liter()
    beta = 1 / KB / exper_info.get_target_temperature_kelvin()
    n_injections = exper_info.get_number_injections()

    DeltaH_0_min, DeltaH_0_max = deltaH0_guesses(q_actual_cal)
    log_sigma_min, log_sigma_max = logsigma_guesses(q_actual_cal)

    stated_P0 = exper_info.get_cell_concentration_milli_molar()
    uncertainty_P0 = dcell * stated_P0
    P0_min = stated_P0 / concentration_range_factor
    P0_max = stated_P0 * concentration_range_factor

    stated_Ls = exper_info.get_syringe_concentration_milli_molar()
    uncertainty_Ls = dsyringe * stated_Ls
    Ls_min = stated_Ls / concentration_range_factor
    Ls_max = stated_Ls * concentration_range_factor

    with pymc3.Model() as model:

        # prior for receptor concentration
        if uniform_P0:
            print("Uniform prior for P0")
            P0 = uniform_prior("P0", lower=P0_min, upper=P0_max)
        else:
            print("LogNormal prior for P0")
            P0 = lognormal_prior("P0", stated_value=stated_P0, uncertainty=uncertainty_P0)

        # prior for ligand concentration
        if uniform_Ls:
            print("Uniform prior for Ls")
            Ls = uniform_prior("Ls", lower=Ls_min, upper=Ls_max)
        else:
            print("LogNormal prior for Ls")
            Ls = lognormal_prior("Ls", stated_value=stated_Ls, uncertainty=uncertainty_Ls)

        # prior for DeltaG
        DeltaG = uniform_prior("DeltaG", lower=-40., upper=40.)

        # prior for DeltaH
        DeltaH = uniform_prior("DeltaH", lower=-100., upper=100.)

        # prior for DeltaH_0
        DeltaH_0 = uniform_prior("DeltaH_0", lower=DeltaH_0_min, upper=DeltaH_0_max)

        # prior for log_sigma
        log_sigma = uniform_prior("log_sigma", lower=log_sigma_min, upper=log_sigma_max)

        q_model_cal = heats_TwoComponentBindingModel(V0, DeltaVn, P0, Ls, DeltaG, DeltaH, DeltaH_0, beta, n_injections)

        sigma_cal = np.exp(log_sigma)

        q_model_micro_cal = q_model_cal * 10.**6
        q_actual_micro_cal = q_actual_cal * 10.**6
        sigma_micro_cal = sigma_cal * 10.**6

        q_obs_cal = pymc3.Normal("q_obs_cal", mu=q_model_micro_cal, sd=sigma_micro_cal, observed=q_actual_micro_cal)

    return model


def make_RacemicMixtureBindingModel(q_actual_cal, exper_info,
                                    dcell=0.1, dsyringe=0.1,
                                    uniform_P0=False, uniform_Ls=False, concentration_range_factor=10,
                                    is_rho_free_param=False):
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
    :param is_rho_free_param: bool
    :return: an instance of pymc3.model.Model
    """

    V0 = exper_info.get_cell_volume_liter()
    DeltaVn = exper_info.get_injection_volumes_liter()
    beta = 1 / KB / exper_info.get_target_temperature_kelvin()
    n_injections = exper_info.get_number_injections()

    DeltaH_0_min, DeltaH_0_max = deltaH0_guesses(q_actual_cal)
    log_sigma_min, log_sigma_max = logsigma_guesses(q_actual_cal)

    stated_P0 = exper_info.get_cell_concentration_milli_molar()
    uncertainty_P0 = dcell * stated_P0
    P0_min = stated_P0 / concentration_range_factor
    P0_max = stated_P0 * concentration_range_factor

    stated_Ls = exper_info.get_syringe_concentration_milli_molar()
    uncertainty_Ls = dsyringe * stated_Ls
    Ls_min = stated_Ls / concentration_range_factor
    Ls_max = stated_Ls * concentration_range_factor

    # P0, Ls, rho, DeltaH1, DeltaH2, DeltaH_0, DeltaG1, DeltaDeltaG

    with pymc3.Model() as model:

        # prior for receptor concentration
        if uniform_P0:
            print("Uniform prior for P0")
            P0 = uniform_prior("P0", lower=P0_min, upper=P0_max)
        else:
            print("LogNormal prior for P0")
            P0 = lognormal_prior("P0", stated_value=stated_P0, uncertainty=uncertainty_P0)

        # prior for ligand concentration
        if uniform_Ls:
            print("Uniform prior for Ls")
            Ls = uniform_prior("Ls", lower=Ls_min, upper=Ls_max)
        else:
            print("LogNormal prior for Ls")
            Ls = lognormal_prior("Ls", stated_value=stated_Ls, uncertainty=uncertainty_Ls)

        # prior for DeltaG1, and DeltaDeltaG
        DeltaG1 = uniform_prior("DeltaG1", lower=-40., upper=40.)
        DeltaDeltaG = uniform_prior("DeltaDeltaG", lower=0., upper=40.)

        # prior for DeltaH1 and DeltaH2
        DeltaH1 = uniform_prior("DeltaH1", lower=-100., upper=100.)
        DeltaH2 = uniform_prior("DeltaH2", lower=-100., upper=100.)

        # prior for DeltaH_0
        DeltaH_0 = uniform_prior("DeltaH_0", lower=DeltaH_0_min, upper=DeltaH_0_max)

        # prior for log_sigma
        log_sigma = uniform_prior("log_sigma", lower=log_sigma_min, upper=log_sigma_max)

        if is_rho_free_param:
            print("EnantiomerBindingModel")
            rho = uniform_prior("rho", lower=0., upper=1.)
            q_model_cal = heats_RacemicMixtureBindingModel(V0, DeltaVn, P0, Ls, rho,
                                                           DeltaH1, DeltaH2, DeltaH_0, DeltaG1, DeltaDeltaG,
                                                           beta, n_injections)
        else:
            print("RacemicMixtureBindingModel")
            q_model_cal = heats_RacemicMixtureBindingModel(V0, DeltaVn, P0, Ls, 0.5,
                                                           DeltaH1, DeltaH2, DeltaH_0, DeltaG1, DeltaDeltaG,
                                                           beta, n_injections)

        sigma = np.exp(log_sigma)

        q_obs_cal = pymc3.Normal("q_obs_cal", mu=q_model_cal, sd=sigma, observed=q_actual_cal)

    return model
