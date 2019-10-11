"""
define function to calculate bayes factor
"""

from __future__ import print_function

import numpy as np

from _models import normal_likelihood
from _models import heats_TwoComponentBindingModel, heats_RacemicMixtureBindingModel
from _models import log_prior_likelihood_2cbm, log_prior_likelihood_rmbm, log_prior_likelihood_embm


def average_likelihood_from_prior_2cbm(q_actual, V0, DeltaVn, beta, n_injections, mcmc_trace, nsamples=None):
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


def average_likelihood_from_prior_rmbm(q_actual, V0, DeltaVn, beta, n_injections, mcmc_trace, nsamples=None):
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

        q_model_micro_cal = q_model_cal * 10. ** 6

        sigma_cal = np.exp(log_sigma)
        sigma_micro_cal = sigma_cal * 10 ** 6

        aver_likelihood += normal_likelihood(q_actual, q_model_micro_cal, sigma_micro_cal)

    return aver_likelihood / len(P0_trace)


def average_likelihood_from_prior_embm(q_actual, V0, DeltaVn, beta, n_injections, mcmc_trace, nsamples=None):
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

    return aver_likelihood / len(P0_trace)


def average_likelihood_from_posterior(model, q_actual_cal, exper_info, mcmc_trace,
                                      dcell=0.1, dsyringe=0.1,
                                      uniform_P0=False, uniform_Ls=False, concentration_range_factor=10,
                                      nsamples=None):
    """
    :param model: str
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
    if model == "2cbm":
        log_prior_likelihood = log_prior_likelihood_2cbm
    elif model == "rmbm":
        log_prior_likelihood = log_prior_likelihood_rmbm
    elif model == "embm":
        log_prior_likelihood = log_prior_likelihood_embm
    else:
        raise ValueError("Unknown model: " + model)

    log_priors, log_likelihoods = log_prior_likelihood(q_actual_cal, exper_info, mcmc_trace,
                                                       dcell=dcell, dsyringe=dsyringe,
                                                       uniform_P0=uniform_P0, uniform_Ls=uniform_Ls,
                                                       concentration_range_factor=concentration_range_factor,
                                                       nsamples=nsamples)

    log_weights = -log_likelihoods
    weights = np.exp(log_weights)
    weights = weights / np.sum(weights)

    llhs = np.exp(log_likelihoods)
    llhs_weighted = llhs * weights

    llh_weighted_max = llhs_weighted.max()

    llh_mean = np.sum(llhs_weighted / llh_weighted_max)
    llh_max_log = np.log(llh_weighted_max)

    # the final result will be llh_mean * np.exp(llh_max_log)
    return llh_mean, llh_max_log
