"""
define function to calculate bayes factor
"""

from __future__ import print_function

import numpy as np

from _models import normal_likelihood
from _models import heats_TwoComponentBindingModel, heats_RacemicMixtureBindingModel


def average_likelihood_TwoComponentBindingModel(q_actual, V0, DeltaVn, beta, n_injections, mcmc_trace, nsamples=None):
    """
    :param q_actual: observed heats, (micro calorie)
    :param V0: cell volume (liter)
    :param DeltaVn: injection volumes (liter)
    :param beta: inverse temperature * gas constant (mole / kcal)
    :param n_injections: int
    :param mcmc_trace: dict, "parameter" --> 1d ndarray
    :param nsamples: int
    :return: aver_likelihood, float
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


def average_likelihood_RacemicMixtureBindingModel(q_actual, V0, DeltaVn, beta, n_injections, mcmc_trace, nsamples=None):
    """
    :param q_actual: observed heats, (micro calorie)
    :param V0: cell volume (liter)
    :param DeltaVn: injection volumes (liter)
    :param beta: inverse temperature * gas constant (mole / kcal)
    :param n_injections: int
    :param mcmc_trace: dict, "parameter" --> 1d ndarray
    :param nsamples: int
    :return: aver_likelihood, float
    """
    if nsamples is None:
        nsamples = len(mcmc_trace["P0"])
    assert nsamples <= len(mcmc_trace["P0"]), "nsamples too big"

    P0_trace = mcmc_trace["P0"][: nsamples]
    Ls_trace = mcmc_trace["Ls"][: nsamples]
    rho = 0.5
    DeltaG1_trace = mcmc_trace["DeltaG1"][: nsamples]
    DeltaDeltaG_trace = mcmc_trace["DeltaDeltaG"][: nsamples]
    DeltaH1_trace = mcmc_trace["DeltaH1"][: nsamples]
    DeltaH2_trace = mcmc_trace["DeltaH2"][: nsamples]
    DeltaH_0_trace = mcmc_trace["DeltaH_0"][: nsamples]
    log_sigma_trace = mcmc_trace["log_sigma"][: nsamples]

    aver_likelihood = 0.
    for P0, Ls, DeltaG1, DeltaDeltaG, DeltaH1, DeltaH2, DeltaH_0, log_sigma in zip(P0_trace, Ls_trace,
                                                                                   DeltaG1_trace, DeltaDeltaG_trace,
                                                                                   DeltaH1_trace, DeltaH2_trace,
                                                                                   DeltaH_0_trace, log_sigma_trace):
        q_model_cal = heats_RacemicMixtureBindingModel(V0, DeltaVn, P0, Ls, rho, DeltaH1, DeltaH2, DeltaH_0,
                                                       DeltaG1, DeltaDeltaG, beta, n_injections)
        if np.isnan(q_model_cal):
            print("q_model_cal = nan with V0, DeltaVn, P0, Ls, DeltaG1, DeltaDeltaG, DeltaH1, DeltaH2, DeltaH_0, log_sigma, n_injections")
            print(V0, DeltaVn, P0, Ls, DeltaG1, DeltaDeltaG, DeltaH1, DeltaH2, DeltaH_0, log_sigma, n_injections)

        q_model_micro_cal = q_model_cal * 10. ** 6

        sigma_cal = np.exp(log_sigma)
        sigma_micro_cal = sigma_cal * 10 ** 6

        llh = normal_likelihood(q_actual, q_model_micro_cal, sigma_micro_cal)
        aver_likelihood += llh

        if np.isnan(llh):
            print("llh = nan with V0, DeltaVn, P0, Ls, DeltaG1, DeltaDeltaG, DeltaH1, DeltaH2, DeltaH_0, log_sigma, n_injections")
            print(V0, DeltaVn, P0, Ls, DeltaG1, DeltaDeltaG, DeltaH1, DeltaH2, DeltaH_0, log_sigma, n_injections)

    return aver_likelihood / len(P0_trace)


def average_likelihood_EnantiomerBindingModel(q_actual, V0, DeltaVn, beta, n_injections, mcmc_trace, nsamples=None):
    """
    :param q_actual: observed heats, (micro calorie)
    :param V0: cell volume (liter)
    :param DeltaVn: injection volumes (liter)
    :param beta: inverse temperature * gas constant (mole / kcal)
    :param n_injections: int
    :param mcmc_trace: dict, "parameter" --> 1d ndarray
    :param nsamples: int
    :return: aver_likelihood, float
    """
    if nsamples is None:
        nsamples = len(mcmc_trace["P0"])
    assert nsamples <= len(mcmc_trace["P0"]), "nsamples too big"

    P0_trace = mcmc_trace["P0"][: nsamples]
    Ls_trace = mcmc_trace["Ls"][: nsamples]
    rho_trace = mcmc_trace["rho"][: nsamples]
    DeltaG1_trace = mcmc_trace["DeltaG1"][: nsamples]
    DeltaDeltaG_trace = mcmc_trace["DeltaDeltaG"][: nsamples]
    DeltaH1_trace = mcmc_trace["DeltaH1"][: nsamples]
    DeltaH2_trace = mcmc_trace["DeltaH2"][: nsamples]
    DeltaH_0_trace = mcmc_trace["DeltaH_0"][: nsamples]
    log_sigma_trace = mcmc_trace["log_sigma"][: nsamples]

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

    print("nsamples", nsamples)
    print("total likelihood", aver_likelihood)
    print("len(P0_trace)", len(P0_trace))

    return aver_likelihood / len(P0_trace)
