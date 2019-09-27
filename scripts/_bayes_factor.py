"""
define function to calculate bayes factor
"""

from __future__ import print_function

import numpy as np

from _models import heats_TwoComponentBindingModel, heats_RacemicMixtureBindingModel


def normal_likelihood(q_actual, q_model, sigma):
    """
    :param q_actual: 1d ndarray, actual or observed values of heats
    :param q_model: heat calculated from a model
    :param sigma: standard deviation
    :return: likelihood, float

    log_likelihood = -(N/2)\ln(2 \pi \sigma^2) - 1/(2 \sigma^2) \sum_{i=1}^N \epsilon^2
    """
    assert len(q_actual) == len(q_model), "q_actual and q_model must have the same len"
    sum_e_squared = np.sum((q_model - q_actual)**2)

    n_injections = len(q_actual)
    sigma_2 = sigma**2
    log_likelihood = - n_injections / 2 * np.log(2 * np.pi * sigma_2) - sum_e_squared / 2 / sigma_2

    return np.exp(log_likelihood)


def average_likelihood_TwoComponentBindingModel(q_actual, V0, DeltaVn, beta, n_injections, mcmc_trace):
    """
    :param q_actual: observed heats, (micro calorie)
    :param V0: cell volume (liter)
    :param DeltaVn: injection volumes (liter)
    :param beta: inverse temperature * gas constant (mole / kcal)
    :param n_injections: int
    :param mcmc_trace: dict, "parameter" --> 1d ndarray
    :return: aver_likelihood, float
    """
    P0_trace = mcmc_trace["P0"]
    Ls_trace = mcmc_trace["Ls"]
    DeltaG_trace = mcmc_trace["DeltaG"]
    DeltaH_trace = mcmc_trace["DeltaH"]
    DeltaH_0_trace = mcmc_trace["DeltaH_0"]
    log_sigma_trace = mcmc_trace["log_sigma"]

    aver_likelihood = 0.
    for P0, Ls, DeltaG, DeltaH, DeltaH_0, log_sigma in zip(P0_trace, Ls_trace, DeltaG_trace, DeltaH_trace,
                                                           DeltaH_0_trace, log_sigma_trace):
        q_model_cal = heats_TwoComponentBindingModel(V0, DeltaVn, P0, Ls, DeltaG, DeltaH,
                                                     DeltaH_0, beta, n_injections)
        q_model_micro_cal = q_model_cal * 10.**6

        sigma_cal = np.exp(log_sigma)
        sigma_micro_cal = sigma_cal * 10**6

        aver_likelihood += normal_likelihood(q_actual, q_model_micro_cal, sigma_micro_cal)

    return aver_likelihood / len(P0_trace)


def average_likelihood_RacemicMixtureBindingModel(q_actual, V0, DeltaVn, beta, n_injections, mcmc_trace):
    """
    :param q_actual: observed heats, (micro calorie)
    :param V0: cell volume (liter)
    :param DeltaVn: injection volumes (liter)
    :param beta: inverse temperature * gas constant (mole / kcal)
    :param n_injections: int
    :param mcmc_trace: dict, "parameter" --> 1d ndarray
    :return: aver_likelihood, float
    """
    P0_trace = mcmc_trace["P0"]
    Ls_trace = mcmc_trace["Ls"]
    rho_trace = mcmc_trace["rho"]
    DeltaG1_trace = mcmc_trace["DeltaG1"]
    DeltaDeltaG_trace = mcmc_trace["DeltaDeltaG"]
    DeltaH1_trace = mcmc_trace["DeltaH1"]
    DeltaH2_trace = mcmc_trace["DeltaH2"]
    DeltaH_0_trace = mcmc_trace["DeltaH_0"]
    log_sigma_trace = mcmc_trace["log_sigma"]

    aver_likelihood = 0.
    for P0, Ls, rho, DeltaG1, DeltaDeltaG, DeltaH1, DeltaH2, DeltaH_0, log_sigma in zip(P0_trace, Ls_trace, rho_trace,
                                                                                        DeltaG1_trace, DeltaDeltaG_trace,
                                                                                        DeltaH1_trace, DeltaH2_trace,
                                                                                        DeltaH_0_trace, log_sigma_trace):
        q_model_cal = heats_RacemicMixtureBindingModel(V0, DeltaVn, P0, Ls, rho, DeltaH1, DeltaH2, DeltaH_0,
                                                       DeltaG1, DeltaDeltaG, beta, n_injections)
        q_model_micro_cal = q_model_cal * 10. ** 6

        sigma_cal = np.exp(log_sigma)
        sigma_micro_cal = sigma_cal * 10 ** 6

        aver_likelihood += normal_likelihood(q_actual, q_model_micro_cal, sigma_micro_cal)

    return aver_likelihood / len(P0_trace)