"""
define different heat models
"""

import numpy as np
from scipy import stats
import pymc

from _data_io import ITCExperiment, load_heat_micro_cal

KB = 0.0019872041      # in kcal/mol/K

# copied from the method expected_injection_heats of the class TwoComponentBindingModel in bayesian_itc/bitc/models.py
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

    Kd = np.exp(beta * DeltaG)   # dissociation constant (M)
    N = N

    # Compute complex concentrations.
    # Pn[n] is the protein concentration in sample cell after n injections
    # (M)
    Pn = np.zeros([N])
    # Ln[n] is the ligand concentration in sample cell after n injections
    # (M)
    Ln = np.zeros([N])
    # PLn[n] is the complex concentration in sample cell after n injections
    # (M)
    PLn = np.zeros([N])
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
        PLn[n] = (0.5 / V0 * ((P + L + Kd * V0) - ((P + L + Kd * V0) ** 2 - 4 * P * L) ** 0.5))
        # free protein concentration in sample cell after n injections (M)
        Pn[n] = P / V0 - PLn[n]
        # free ligand concentration in sample cell after n injections (M)
        Ln[n] = L / V0 - PLn[n]

    # Compute expected injection heats.
    # q_n_model[n] is the expected heat from injection n
    q_n = np.zeros([N])
    # Instantaneous injection model (perfusion)
    # first injection
    q_n[0] = (DeltaH * V0 * PLn[0])*1000 + DeltaH_0
    for n in range(1, N):
        d = 1.0 - (DeltaVn[n] / V0)  # dilution factor (dimensionless)
        # subsequent injections
        q_n[n] = (DeltaH * V0 * (PLn[n] - d * PLn[n - 1])) * 1000 + DeltaH_0

    return np.array(q_n)


# copied from method equilibrium_concentrations of class RacemicMixtureBindingModel in bayesian_itc/bitc/models.py
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


def heats_RacemicMixtureBindingModel(V0, DeltaVn, P0, Ls, rho, DeltaH1, DeltaH2, DeltaH_0, DeltaG1, DeltaDeltaG, beta, N):
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
    RL1n = np.zeros([N])
    RL2n = np.zeros([N])
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

        RL1n[n], RL2n[n] = _equilibrium_concentrations(Kd1, Kd2, C0_R, C0_L1, C0_L2, V0)

    # Compute expected injection heats.
    # q_n_model[n] is the expected heat from injection n
    q_n = np.zeros([N])
    # Instantaneous injection model (perfusion)
    # first injection
    q_n[0] = (DeltaH1 * V0 * RL1n[0] + DeltaH2 * V0 * RL2n[0]) * 1000 + DeltaH_0
    # TODO do we need dcum = 1.0 here?
    for n in range(1, N):
        d = 1.0 - (DeltaVn[n] / V0)  # dilution factor (dimensionless)
        # subsequent injections
        q_n[n] = (DeltaH1 * V0 * (RL1n[n] - d*RL1n[n-1]) + DeltaH2 * V0 * (RL2n[n] - d*RL2n[n-1])) * 1000 + DeltaH_0

    return np.array(q_n)


def log_likelihood_normal(q_actual, q_model, sigma):
    """
    :param q_actual: 1d ndarray, actual or observed values of heats
    :param q_model: heat calculated from a model
    :param sigma: standard deviation
    :return: likelihood, float

    log_likelihood = -(N/2)\ln(2 \pi \sigma^2) - 1/(2 \sigma^2) \sum_{i=1}^N \epsilon^2
    """
    zs = (q_model - q_actual) / sigma
    norm_rv = stats.norm(loc=0, scale=1)
    log_likelihood = np.sum(norm_rv.logpdf(zs))

    return log_likelihood


def lognormal_pdf(x, stated_center, uncertainty):
    """
    :param x: float
    :param stated_center: float
    :param uncertainty: float
    :return: pdf, float
    """
    if x <= 0:
        return 0.

    m = stated_center
    v = uncertainty**2

    mu = np.log(m / np.sqrt(1 + (v / (m ** 2))))
    sigma_2 = np.log(1 + (v / (m**2)))

    pdf = 1 / x / np.sqrt(2 * np.pi * sigma_2) * np.exp(-0.5 / sigma_2 * (np.log(x) - mu)**2)
    return pdf


def uniform_pdf(x, lower, upper):
    """
    :param x:float
    :param lower: float
    :param upper: float
    :return: pdf, float
    """
    assert upper > lower, "upper must be greater than lower"
    if (x < lower) or (x > upper):
        return 0.

    return 1. / (upper - lower)


# copied from bayesian_itc/bitc/models
def logsigma_guesses(q_n_cal):
    """
    :param q_n_cal: heats in calorie
    :return:
    """
    log_sigma_guess = np.log(q_n_cal[-4:].std())
    log_sigma_min = log_sigma_guess - 10
    log_sigma_max = log_sigma_guess + 5
    return log_sigma_min, log_sigma_max


# copied from bayesian_itc/bitc/models
def deltaH0_guesses(q_n_cal):
    # Assume the last injection has the best guess for H0
    #DeltaH_0_guess = q_n_cal[-1]
    heat_interval = (q_n_cal.max() - q_n_cal.min())
    DeltaH_0_min = q_n_cal.min() - heat_interval
    DeltaH_0_max = q_n_cal.max() + heat_interval
    return DeltaH_0_min, DeltaH_0_max


def log_prior_likelihood_2cbm(q_actual_cal, exper_info, mcmc_trace,
                              dcell=0.1, dsyringe=0.1,
                              uniform_P0=False, uniform_Ls=False, concentration_range_factor=10,
                              nsamples=None):
    """
    :param q_actual_cal: observed heats in calorie
    :param exper_info: an object of _data_io.ITCExperiment class
    :param mcmc_trace: dict, "parameter" --> 1d ndarray
    :param dcell: float, relative uncertainty in cell concentration
    :param dsyringe: float, relative uncertainty in syringe concentration
    :param uniform_P0: bool
    :param uniform_Ls: bool
    :param concentration_range_factor: float
    :param nsamples: int
    :return: values of parameters that maximize the posterior
    """
    if nsamples is None:
        nsamples = len(mcmc_trace["P0"])
    assert nsamples <= len(mcmc_trace["P0"]), "nsamples too big"

    P0_trace = mcmc_trace["P0"][: nsamples]
    Ls_trace = mcmc_trace["Ls"][: nsamples]
    DeltaG_trace = mcmc_trace["DeltaG"][: nsamples]
    DeltaH_trace = mcmc_trace["DeltaH"][: nsamples]
    DeltaH_0_trace = mcmc_trace["DeltaH_0"][: nsamples]
    log_sigma_trace = mcmc_trace["log_sigma"][: nsamples]

    V0 = exper_info.get_cell_volume_liter()
    DeltaVn = exper_info.get_injection_volumes_liter()
    beta = 1 / KB / exper_info.get_target_temperature_kelvin()
    n_injections = exper_info.get_number_injections()

    DeltaH_0_min, DeltaH_0_max = deltaH0_guesses(q_actual_cal)
    logsigma_min, logsigma_max = logsigma_guesses(q_actual_cal)

    stated_P0 = exper_info.get_cell_concentration_milli_molar()
    uncertainty_P0 = dcell * stated_P0
    P0_min = stated_P0 / concentration_range_factor
    P0_max = stated_P0 * concentration_range_factor

    stated_Ls = exper_info.get_syringe_concentration_milli_molar()
    uncertainty_Ls = dsyringe * stated_Ls
    Ls_min = stated_Ls / concentration_range_factor
    Ls_max = stated_Ls * concentration_range_factor

    log_priors = []
    log_likelihoods = []

    for P0, Ls, DeltaG, DeltaH, DeltaH_0, log_sigma in zip(P0_trace, Ls_trace, DeltaG_trace, DeltaH_trace,
                                                           DeltaH_0_trace, log_sigma_trace):
        q_model_cal = heats_TwoComponentBindingModel(V0, DeltaVn, P0, Ls, DeltaG, DeltaH,
                                                     DeltaH_0, beta, n_injections)

        sigma_cal = np.exp(log_sigma)
        likelihood = log_likelihood_normal(q_actual_cal, q_model_cal, sigma_cal)

        prior = 0.
        if not uniform_P0:
            prior += np.log(lognormal_pdf(P0, stated_center=stated_P0, uncertainty=uncertainty_P0))
        else:
            prior += np.log(uniform_pdf(P0, lower=P0_min, upper=P0_max))

        if not uniform_Ls:
            prior += np.log(lognormal_pdf(Ls, stated_center=stated_Ls, uncertainty=uncertainty_Ls))
        else:
            prior += np.log(uniform_pdf(Ls, lower=Ls_min, upper=Ls_max))

        prior += np.log(uniform_pdf(DeltaG, lower=-40., upper=40.))
        prior += np.log(uniform_pdf(DeltaH, lower=-100., upper=100.))

        prior += np.log(uniform_pdf(DeltaH_0, lower=DeltaH_0_min, upper=DeltaH_0_max))
        prior += np.log(uniform_pdf(log_sigma, lower=logsigma_min, upper=logsigma_max))

        log_priors.append(prior)
        log_likelihoods.append(likelihood)

    return np.array(log_priors), np.array(log_likelihoods)


def log_prior_likelihood_rmbm(q_actual_cal, exper_info, mcmc_trace,
                              dcell=0.1, dsyringe=0.1,
                              uniform_P0=False, uniform_Ls=False, concentration_range_factor=10,
                              nsamples=None):
    """
    :param q_actual_cal: observed heats in calorie
    :param exper_info: an object of _data_io.ITCExperiment class
    :param mcmc_trace: dict, "parameter" --> 1d ndarray
    :param dcell: float, relative uncertainty in cell concentration
    :param dsyringe: float, relative uncertainty in syringe concentration
    :param uniform_P0: bool
    :param uniform_Ls: bool
    :param concentration_range_factor: float
    :param nsamples: int
    :return: values of parameters that maximize the posterior
    """
    P0_trace = mcmc_trace["P0"][: nsamples]
    Ls_trace = mcmc_trace["Ls"][: nsamples]
    rho = 0.5
    DeltaG1_trace = mcmc_trace["DeltaG1"][: nsamples]
    DeltaDeltaG_trace = mcmc_trace["DeltaDeltaG"][: nsamples]
    DeltaH1_trace = mcmc_trace["DeltaH1"][: nsamples]
    DeltaH2_trace = mcmc_trace["DeltaH2"][: nsamples]
    DeltaH_0_trace = mcmc_trace["DeltaH_0"][: nsamples]
    log_sigma_trace = mcmc_trace["log_sigma"][: nsamples]

    V0 = exper_info.get_cell_volume_liter()
    DeltaVn = exper_info.get_injection_volumes_liter()
    beta = 1 / KB / exper_info.get_target_temperature_kelvin()
    n_injections = exper_info.get_number_injections()

    DeltaH_0_min, DeltaH_0_max = deltaH0_guesses(q_actual_cal)
    logsigma_min, logsigma_max = logsigma_guesses(q_actual_cal)

    stated_P0 = exper_info.get_cell_concentration_milli_molar()
    uncertainty_P0 = dcell * stated_P0
    P0_min = stated_P0 / concentration_range_factor
    P0_max = stated_P0 * concentration_range_factor

    stated_Ls = exper_info.get_syringe_concentration_milli_molar()
    uncertainty_Ls = dsyringe * stated_Ls
    Ls_min = stated_Ls / concentration_range_factor
    Ls_max = stated_Ls * concentration_range_factor

    log_priors = []
    log_likelihoods = []

    for P0, Ls, DeltaG1, DeltaDeltaG, DeltaH1, DeltaH2, DeltaH_0, log_sigma in zip(P0_trace, Ls_trace,
                                                                                   DeltaG1_trace,
                                                                                   DeltaDeltaG_trace,
                                                                                   DeltaH1_trace, DeltaH2_trace,
                                                                                   DeltaH_0_trace,
                                                                                   log_sigma_trace):
        q_model_cal = heats_RacemicMixtureBindingModel(V0, DeltaVn, P0, Ls, rho, DeltaH1, DeltaH2, DeltaH_0,
                                                       DeltaG1, DeltaDeltaG, beta, n_injections)
        sigma_cal = np.exp(log_sigma)
        likelihood = log_likelihood_normal(q_actual_cal, q_model_cal, sigma_cal)

        prior = 0.
        if not uniform_P0:
            prior += np.log(lognormal_pdf(P0, stated_center=stated_P0, uncertainty=uncertainty_P0))
        else:
            prior += np.log(uniform_pdf(P0, lower=P0_min, upper=P0_max))

        if not uniform_Ls:
            prior += np.log(lognormal_pdf(Ls, stated_center=stated_Ls, uncertainty=uncertainty_Ls))
        else:
            prior += np.log(uniform_pdf(Ls, lower=Ls_min, upper=Ls_max))

        prior += np.log(uniform_pdf(DeltaG1, lower=-40., upper=40.))
        prior += np.log(uniform_pdf(DeltaDeltaG, lower=0., upper=40.))

        prior += np.log(uniform_pdf(DeltaH1, lower=-100., upper=100.))
        prior += np.log(uniform_pdf(DeltaH2, lower=-100., upper=100.))

        prior += np.log(uniform_pdf(DeltaH_0, lower=DeltaH_0_min, upper=DeltaH_0_max))
        prior += np.log(uniform_pdf(log_sigma, lower=logsigma_min, upper=logsigma_max))

        log_priors.append(prior)
        log_likelihoods.append(likelihood)

    return np.array(log_priors), np.array(log_likelihoods)


def log_prior_likelihood_embm(q_actual_cal, exper_info, mcmc_trace,
                              dcell=0.1, dsyringe=0.1,
                              uniform_P0=False, uniform_Ls=False, concentration_range_factor=10,
                              nsamples=None):
    """
    :param q_actual_cal: observed heats in calorie
    :param exper_info: an object of _data_io.ITCExperiment class
    :param mcmc_trace: dict, "parameter" --> 1d ndarray
    :param dcell: float, relative uncertainty in cell concentration
    :param dsyringe: float, relative uncertainty in syringe concentration
    :param uniform_P0: bool
    :param uniform_Ls: bool
    :param concentration_range_factor: float
    :param nsamples: int
    :return: values of parameters that maximize the posterior
    """
    P0_trace = mcmc_trace["P0"][: nsamples]
    Ls_trace = mcmc_trace["Ls"][: nsamples]
    rho_trace = mcmc_trace["rho"][: nsamples]
    DeltaG1_trace = mcmc_trace["DeltaG1"][: nsamples]
    DeltaDeltaG_trace = mcmc_trace["DeltaDeltaG"][: nsamples]
    DeltaH1_trace = mcmc_trace["DeltaH1"][: nsamples]
    DeltaH2_trace = mcmc_trace["DeltaH2"][: nsamples]
    DeltaH_0_trace = mcmc_trace["DeltaH_0"][: nsamples]
    log_sigma_trace = mcmc_trace["log_sigma"][: nsamples]

    V0 = exper_info.get_cell_volume_liter()
    DeltaVn = exper_info.get_injection_volumes_liter()
    beta = 1 / KB / exper_info.get_target_temperature_kelvin()
    n_injections = exper_info.get_number_injections()

    DeltaH_0_min, DeltaH_0_max = deltaH0_guesses(q_actual_cal)
    logsigma_min, logsigma_max = logsigma_guesses(q_actual_cal)

    stated_P0 = exper_info.get_cell_concentration_milli_molar()
    uncertainty_P0 = dcell * stated_P0
    P0_min = stated_P0 / concentration_range_factor
    P0_max = stated_P0 * concentration_range_factor

    stated_Ls = exper_info.get_syringe_concentration_milli_molar()
    uncertainty_Ls = dsyringe * stated_Ls
    Ls_min = stated_Ls / concentration_range_factor
    Ls_max = stated_Ls * concentration_range_factor

    log_priors = []
    log_likelihoods = []

    for P0, Ls, rho, DeltaG1, DeltaDeltaG, DeltaH1, DeltaH2, DeltaH_0, log_sigma in zip(P0_trace, Ls_trace, rho_trace,
                                                                                        DeltaG1_trace,
                                                                                        DeltaDeltaG_trace,
                                                                                        DeltaH1_trace, DeltaH2_trace,
                                                                                        DeltaH_0_trace,
                                                                                        log_sigma_trace):
        q_model_cal = heats_RacemicMixtureBindingModel(V0, DeltaVn, P0, Ls, rho, DeltaH1, DeltaH2, DeltaH_0,
                                                       DeltaG1, DeltaDeltaG, beta, n_injections)
        sigma_cal = np.exp(log_sigma)
        likelihood = log_likelihood_normal(q_actual_cal, q_model_cal, sigma_cal)

        prior = 0.
        if not uniform_P0:
            prior += np.log(lognormal_pdf(P0, stated_center=stated_P0, uncertainty=uncertainty_P0))
        else:
            prior += np.log(uniform_pdf(P0, lower=P0_min, upper=P0_max))

        if not uniform_Ls:
            prior += np.log(lognormal_pdf(Ls, stated_center=stated_Ls, uncertainty=uncertainty_Ls))
        else:
            prior += np.log(uniform_pdf(Ls, lower=Ls_min, upper=Ls_max))

        prior += np.log(uniform_pdf(rho, lower=0., upper=1.))

        prior += np.log(uniform_pdf(DeltaG1, lower=-40., upper=40.))
        prior += np.log(uniform_pdf(DeltaDeltaG, lower=0., upper=40.))

        prior += np.log(uniform_pdf(DeltaH1, lower=-100., upper=100.))
        prior += np.log(uniform_pdf(DeltaH2, lower=-100., upper=100.))

        prior += np.log(uniform_pdf(DeltaH_0, lower=DeltaH_0_min, upper=DeltaH_0_max))
        prior += np.log(uniform_pdf(log_sigma, lower=logsigma_min, upper=logsigma_max))

        log_priors.append(prior)
        log_likelihoods.append(likelihood)

    return np.array(log_priors), np.array(log_likelihoods)


class PyMCLogNormal(object):
    def __init__(self, name, stated_value, uncertainty_percent):
        """
        :param name: str
        :param stated_value: float, mM
        :param uncertainty: float, 0 < uncertainty < 1
        """
        m = stated_value
        uncertainty = stated_value * uncertainty_percent
        v = uncertainty ** 2
        model = pymc.Lognormal(name,
                               mu=np.log(m / np.sqrt(1 + (v / (m**2)))),
                               tau=1.0 / np.log(1 + (v / (m**2))),
                               value=m)

        setattr(self, name, model)


class PyMCUniform(object):
    def __init__(self, name, lower, upper):
        """
        :param name: str
        :param lower: float
        :param upper: float
        """
        assert lower < upper, "lower must be less than upper"
        initial_value = (lower + upper) / 2.

        model = pymc.Uniform(name, lower=lower, upper=upper, value=initial_value)
        setattr(self, name, model)


def run_mcmc(model, iterations, burn, thin):
    mcmc = pymc.MCMC(model)
    mcmc.sample(iter=iterations, burn=burn, thin=thin)
    #pymc.Matplot.plot(mcmc)
    traces = {}
    for s in mcmc.stochastics:
        traces[s.__name__] = s.trace(chain=None)
    return traces


def sample_priors(nsamples, burn, thin,
                  q_actual_cal,
                  stated_P0, stated_Ls, dP0=0.1, dLs=0.1,
                  uniform_P0=False, uniform_Ls=False,
                  concentration_range_factor=10.0):
    """
    :param nsamples: int
    :param burn: int
    :param thin: int
    :param q_actual_cal: observed heats in calorie
    :param stated_P0: float, unit is milli Molar
    :param stated_Ls: float, unit is milli Molar
    :param dP0: float in [0, 1]
    :param dLs: float in [0, 1]
    :param uniform_P0: bool
    :param uniform_Ls: bool
    :param concentration_range_factor: float
    :return: traces, dict, {para_name (str) -> 1D ndarray}
    """
    iterations = nsamples * thin + burn

    all_traces = {}

    print("Sampling DeltaG")
    model_DeltaG = PyMCUniform("DeltaG", lower=-40., upper=40.)
    traces = run_mcmc(model_DeltaG, iterations, burn, thin)
    all_traces.update(traces)

    print("Sampling DeltaG1")
    model_DeltaG1 = PyMCUniform("DeltaG1", lower=-40., upper=40.)
    traces = run_mcmc(model_DeltaG1, iterations, burn, thin)
    all_traces.update(traces)

    print("Sampling DeltaDeltaG")
    model_DeltaDeltaG = PyMCUniform("DeltaDeltaG", lower=0., upper=40.)
    traces = run_mcmc(model_DeltaDeltaG, iterations, burn, thin)
    all_traces.update(traces)

    print("Sampling DeltaH")
    model_DeltaH = PyMCUniform("DeltaH", lower=-100., upper=100.)
    traces = run_mcmc(model_DeltaH, iterations, burn, thin)
    all_traces.update(traces)

    print("Sampling DeltaH1")
    model_DeltaH1 = PyMCUniform("DeltaH1", lower=-100., upper=100.)
    traces = run_mcmc(model_DeltaH1, iterations, burn, thin)
    all_traces.update(traces)

    print("Sampling DeltaH2")
    model_DeltaH2 = PyMCUniform("DeltaH2", lower=-100., upper=100.)
    traces = run_mcmc(model_DeltaH2, iterations, burn, thin)
    all_traces.update(traces)

    if not uniform_P0:
        model_P0 = PyMCLogNormal("P0", stated_value=stated_P0, uncertainty_percent=dP0)
    else:
        model_P0 = PyMCUniform("P0", lower=stated_P0/concentration_range_factor,
                               upper=stated_P0*concentration_range_factor)

    print("Sampling P0")
    traces = run_mcmc(model_P0, iterations, burn, thin)
    all_traces.update(traces)

    if not uniform_Ls:
        model_Ls = PyMCLogNormal("Ls", stated_value=stated_Ls, uncertainty_percent=dLs)
    else:
        model_Ls = PyMCUniform("Ls", lower=stated_Ls/concentration_range_factor,
                               upper=stated_Ls*concentration_range_factor)

    print("Sampling Ls")
    traces = run_mcmc(model_Ls, iterations, burn, thin)
    all_traces.update(traces)

    print("Sampling rho")
    model_rho = PyMCUniform("rho", lower=0., upper=1)
    traces = run_mcmc(model_rho, iterations, burn, thin)
    all_traces.update(traces)

    print("Sampling DeltaH_0")
    DeltaH_0_min, DeltaH_0_max = deltaH0_guesses(q_actual_cal)
    model_DeltaH_0 = PyMCUniform("DeltaH_0", lower=DeltaH_0_min, upper=DeltaH_0_max)
    traces = run_mcmc(model_DeltaH_0, iterations, burn, thin)
    all_traces.update(traces)

    print("Sampling log_sigma")
    logsigma_min, logsigma_max = logsigma_guesses(q_actual_cal)
    model_log_sigma = PyMCUniform("log_sigma", lower=logsigma_min, upper=logsigma_max)
    traces = run_mcmc(model_log_sigma, iterations, burn, thin)
    all_traces.update(traces)

    return all_traces


def extract_loglhs_from_traces_manual(traces, model_name, exper_info_file, heat_file):
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

    if model_name == "2cbm":
        P0_trace = traces["P0"]
        Ls_trace = traces["Ls"]
        DeltaG_trace = traces["DeltaG"]
        DeltaH_trace = traces["DeltaH"]
        DeltaH_0_trace = traces["DeltaH_0"]
        log_sigma_trace = traces["log_sigma"]

        llhs = []
        for P0, Ls, DeltaG, DeltaH, DeltaH_0, log_sigma in zip(P0_trace, Ls_trace, DeltaG_trace, DeltaH_trace,
                                                               DeltaH_0_trace, log_sigma_trace):
            q_model_cal = heats_TwoComponentBindingModel(V0, DeltaVn, P0, Ls, DeltaG, DeltaH,
                                                         DeltaH_0, beta, n_injections)

            sigma_cal = np.exp(log_sigma)
            llhs.append(log_likelihood_normal(q_actual_cal, q_model_cal, sigma_cal))

        return np.array(llhs)

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

        llhs = []
        for P0, Ls, DeltaG1, DeltaDeltaG, DeltaH1, DeltaH2, DeltaH_0, log_sigma in zip(P0_trace, Ls_trace,
                                                                                       DeltaG1_trace,
                                                                                       DeltaDeltaG_trace,
                                                                                       DeltaH1_trace, DeltaH2_trace,
                                                                                       DeltaH_0_trace,
                                                                                       log_sigma_trace):
            q_model_cal = heats_RacemicMixtureBindingModel(V0, DeltaVn, P0, Ls, rho, DeltaH1, DeltaH2, DeltaH_0,
                                                           DeltaG1, DeltaDeltaG, beta, n_injections)
            sigma_cal = np.exp(log_sigma)
            llhs.append(log_likelihood_normal(q_actual_cal, q_model_cal, sigma_cal))

        return np.array(llhs)

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

        llhs = []
        for P0, Ls, rho, DeltaG1, DeltaDeltaG, DeltaH1, DeltaH2, DeltaH_0, log_sigma in zip(P0_trace, Ls_trace,
                                                                                            rho_trace,
                                                                                            DeltaG1_trace,
                                                                                            DeltaDeltaG_trace,
                                                                                            DeltaH1_trace,
                                                                                            DeltaH2_trace,
                                                                                            DeltaH_0_trace,
                                                                                            log_sigma_trace):
            q_model_cal = heats_RacemicMixtureBindingModel(V0, DeltaVn, P0, Ls, rho, DeltaH1, DeltaH2, DeltaH_0,
                                                           DeltaG1, DeltaDeltaG, beta, n_injections)
            sigma_cal = np.exp(log_sigma)
            llhs.append(log_likelihood_normal(q_actual_cal, q_model_cal, sigma_cal))

        return np.array(llhs)


def marginal_likelihood_v1(log_likelihoods):
    """
    :param log_likelihoods: 1d array
    :return: marg_llh, float
    """
    log_weights = -(log_likelihoods - log_likelihoods.max())
    weights = np.exp(log_weights)
    weights = weights / np.sum(weights)
    llhs = np.exp(log_likelihoods)
    llhs_weighted = llhs * weights

    marg_llh = np.sum(llhs_weighted)
    return marg_llh


def marginal_likelihood_v2(log_likelihoods):
    """
    :param log_likelihoods: 1d array
    :return: marg_llh, float
    """
    log_weights = -(log_likelihoods - log_likelihoods.max())
    sum_weight = np.sum(np.exp(log_weights))
    log_sum_weight = np.log(sum_weight)

    marg_llh = np.sum(np.exp(log_weights - log_sum_weight + log_likelihoods))
    return marg_llh


def marginal_likelihood_v3(log_likelihoods):
    """
    :param log_likelihoods: 1d array
    :return: marg_llh, float
    """
    n = len(log_likelihoods)
    sum_weight = np.sum(np.exp(-log_likelihoods))
    marg_llh = n / sum_weight
    return marg_llh


marginal_likelihood = marginal_likelihood_v2


def log_marginal_likelihood_v2(log_likelihoods):
    """
    :param log_likelihoods: 1d array
    :return: log_marg_llh, float, log of marginal likelihood
    """
    log_weights = -log_likelihoods
    log_weights = log_weights - log_weights.max()
    total_weight = np.sum(np.exp(log_weights))

    a_const = log_likelihoods.max()
    log_likelihoods = log_likelihoods - a_const

    weighted_llhs = np.exp(log_weights + log_likelihoods)
    total_weighted_llh = np.sum(weighted_llhs)

    log_marg_llh = a_const + np.log(total_weighted_llh) - np.log(total_weight)
    return log_marg_llh


def log_marginal_likelihood_v3(log_likelihoods):
    """
    :param log_likelihoods: 1d array
    :return: log_marg_llh, float, log of marginal likelihood
    """
    n = len(log_likelihoods)
    log_weights = -log_likelihoods
    a_const = log_weights.max()
    log_weights = log_weights - a_const
    total_weight = np.sum(np.exp(log_weights))

    log_marg_llh = np.log(n) - a_const - np.log(total_weight)
    return log_marg_llh


log_marginal_likelihood = log_marginal_likelihood_v3
