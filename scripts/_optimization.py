"""
define function to optimize the posterior
"""

import numpy as np

from _models import heats_TwoComponentBindingModel


KB = 0.0019872041      # in kcal/mol/K


def log_likelihood(q_actual, q_model, sigma):
    """
    :param q_actual: 1d ndarray, actual or observed values of heats
    :param q_model: heat calculated from a model
    :param sigma: standard deviation
    :return: log_likelihood, float

    log_likelihood = -(N/2)\ln(2 \pi \sigma^2) - 1/(2 \sigma^2) \sum_{i=1}^N \epsilon^2
    """
    assert len(q_actual) == len(q_model), "q_actual and q_model must have the same len"
    sum_e_squared = np.sum((q_model - q_actual)**2)

    n_injections = len(q_actual)
    sigma_2 = sigma**2
    log_likelihood = - n_injections / 2. * np.log(2 * np.pi * sigma_2) - sum_e_squared / 2. / sigma_2

    return log_likelihood


def log_lognormal(x, stated_center, uncertainty):
    """
    :param x: float
    :param stated_center: float
    :param uncertainty: float
    :return: log_pdf, float
    """
    if x <= 0:
        return 0.

    m = stated_center
    v = uncertainty**2

    mu = np.log(m / np.sqrt(1 + (v / (m ** 2))))
    sigma_2 = np.log(1 + (v / (m**2)))

    pdf = 1 / x / np.sqrt(2 * np.pi * sigma_2) * np.exp(-0.5 / sigma_2 * (np.log(x) - mu)**2)

    return np.log(pdf)


def minus_log_posterior_2cbm(q_actual_cal, exper_info,
                             DeltaG, DeltaH, P0, Ls, DeltaH_0, log_sigma,
                             dcell=0.1, dsyringe=0.1,
                             uniform_P0=False, uniform_Ls=False):
    """
    :param q_actual_cal: observed heats in calorie
    :param exper_info: an object of _data_io.ITCExperiment class
    :param DeltaG: float, free energy of binding (kcal/mol)
    :param DeltaH: float, enthalpy of binding (kcal/mol)
    :param P0: float, Cell concentration (millimolar)
    :param Ls: float, Syringe concentration (millimolar)
    :param DeltaH_0: float, heat of injection (cal)
    :param log_sigma: float, log of sigma, sigma is in cal
    :param dcell: float, relative uncertainty in cell concentration (0 < dcell < 1)
    :param dsyringe: float, relative uncertainty in syringe concentration (0 < dcell < 1)
    :param uniform_P0: bool
    :param uniform_Ls: bool
    :return: values of parameters that maximize the posterior
    """

    V0 = exper_info.get_cell_volume_liter()
    DeltaVn = exper_info.get_injection_volumes_liter()
    beta = 1 / KB / exper_info.get_target_temperature_kelvin()
    n_injections = exper_info.get_number_injections()

    q_model_cal = heats_TwoComponentBindingModel(V0, DeltaVn, P0, Ls, DeltaG, DeltaH,
                                                 DeltaH_0, beta, n_injections)

    sigma_cal = np.exp(log_sigma)

    log_posterior = log_likelihood(q_actual_cal, q_model_cal, sigma_cal)

    if not uniform_P0:
        stated_P0 = exper_info.get_cell_concentration_milli_molar()
        uncertainty_P0 = dcell * stated_P0
        log_posterior += log_lognormal(P0, stated_P0, uncertainty_P0)

    if not uniform_Ls:
        stated_Ls = exper_info.get_syringe_concentration_milli_molar()
        uncertainty_Ls = dsyringe * stated_Ls
        log_posterior += log_lognormal(Ls, stated_Ls, uncertainty_Ls)

    return -log_posterior


def generate_objective_2cbm(q_actual_cal, exper_info,
                             dcell=0.1, dsyringe=0.1,
                             uniform_P0=False, uniform_Ls=False):
    """
    :param q_actual_cal: observed heats in calorie
    :param exper_info: an object of _data_io.ITCExperiment class
    :param dcell: float, relative uncertainty in cell concentration (0 < dcell < 1)
    :param dsyringe: float, relative uncertainty in syringe concentration (0 < dcell < 1)
    :param uniform_P0: float, Cell concentration (millimolar)
    :param uniform_Ls: float, Syringe concentration (millimolar)
    :return: the objective function to be optimized
    """

    def objective(x):
        DeltaG, DeltaH, P0, Ls, DeltaH_0, log_sigma = x
        m_log_posterior = minus_log_posterior_2cbm(q_actual_cal, exper_info,
                             DeltaG, DeltaH, P0, Ls, DeltaH_0, log_sigma,
                             dcell=dcell, dsyringe=dsyringe,
                             uniform_P0=uniform_P0, uniform_Ls=uniform_Ls)
        return m_log_posterior

    return objective

