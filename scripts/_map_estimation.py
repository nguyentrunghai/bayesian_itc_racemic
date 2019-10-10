"""
define function to perform MAP (Maximum a posterior) estimation
"""

from __future__ import print_function

import numpy as np

from _models import heats_TwoComponentBindingModel, heats_RacemicMixtureBindingModel
from _models import normal_likelihood, lognormal_pdf, uniform_pdf, deltaH0_guesses, logsigma_guesses
from _models import log_unnormalized_posterior_2cbm, log_unnormalized_posterior_rmbm


KB = 0.0019872041      # in kcal/mol/K


def map_TwoComponentBindingModel(q_actual_cal, exper_info, mcmc_trace,
                                 dcell=0.1, dsyringe=0.1,
                                 uniform_P0=False, uniform_Ls=False, concentration_range_factor=10):
    """
    maximum a posterior
    :param q_actual_cal: observed heats in calorie
    :param exper_info: an object of _data_io.ITCExperiment class
    :param mcmc_trace: dict, "parameter" --> 1d ndarray
    :param dcell: float, relative uncertainty in cell concentration
    :param dsyringe: float, relative uncertainty in syringe concentration
    :param uniform_P0: bool
    :param uniform_Ls: bool
    :param concentration_range_factor: float
    :return: values of parameters that maximize the posterior
    """
    P0_trace = mcmc_trace["P0"]
    Ls_trace = mcmc_trace["Ls"]
    DeltaG_trace = mcmc_trace["DeltaG"]
    DeltaH_trace = mcmc_trace["DeltaH"]
    DeltaH_0_trace = mcmc_trace["DeltaH_0"]

    log_probs = log_unnormalized_posterior_2cbm(q_actual_cal, exper_info, mcmc_trace,
                                                dcell=dcell, dsyringe=dsyringe,
                                                uniform_P0=uniform_P0, uniform_Ls=uniform_Ls,
                                                concentration_range_factor=concentration_range_factor)
    map_idx = np.argmax(log_probs)
    print("Map index: %d" % map_idx)

    map_P0 = P0_trace[map_idx]
    map_Ls = Ls_trace[map_idx]
    map_DeltaG = DeltaG_trace[map_idx]
    map_DeltaH = DeltaH_trace[map_idx]
    map_DeltaH_0 = DeltaH_0_trace[map_idx]

    return map_P0, map_Ls, map_DeltaG, map_DeltaH, map_DeltaH_0


def map_RacemicMixtureBindingModel(q_actual_cal, exper_info, mcmc_trace,
                                   dcell=0.1, dsyringe=0.1,
                                   uniform_P0=False, uniform_Ls=False, concentration_range_factor=10):
    """
    maximum a posterior
    :param q_actual_cal: observed heats in calorie
    :param exper_info: an object of _data_io.ITCExperiment class
    :param mcmc_trace: dict, "parameter" --> 1d ndarray
    :param dcell: float, relative uncertainty in cell concentration
    :param dsyringe: float, relative uncertainty in syringe concentration
    :param uniform_P0: bool
    :param uniform_Ls: bool
    :param concentration_range_factor: float
    :return: values of parameters that maximize the posterior
    """
    P0_trace = mcmc_trace["P0"]
    Ls_trace = mcmc_trace["Ls"]
    rho = 0.5
    DeltaG1_trace = mcmc_trace["DeltaG1"]
    DeltaDeltaG_trace = mcmc_trace["DeltaDeltaG"]
    DeltaH1_trace = mcmc_trace["DeltaH1"]
    DeltaH2_trace = mcmc_trace["DeltaH2"]
    DeltaH_0_trace = mcmc_trace["DeltaH_0"]

    log_probs = log_unnormalized_posterior_rmbm(q_actual_cal, exper_info, mcmc_trace,
                                                dcell=dcell, dsyringe=dsyringe,
                                                uniform_P0=uniform_P0, uniform_Ls=uniform_Ls,
                                                concentration_range_factor=concentration_range_factor)
    map_idx = np.argmax(log_probs)
    print("Map index: %d" % map_idx)

    map_P0 = P0_trace[map_idx]
    map_Ls = Ls_trace[map_idx]
    map_DeltaG1 = DeltaG1_trace[map_idx]
    map_DeltaDeltaG = DeltaDeltaG_trace[map_idx]
    map_DeltaH1 = DeltaH1_trace[map_idx]
    map_DeltaH2 = DeltaH2_trace[map_idx]
    map_DeltaH_0 = DeltaH_0_trace[map_idx]

    return map_P0, map_Ls, map_DeltaG1, map_DeltaDeltaG, map_DeltaH1, map_DeltaH2, map_DeltaH_0


def map_EnantiomerBindingModel(q_actual_cal, exper_info, mcmc_trace,
                               dcell=0.1, dsyringe=0.1,
                               uniform_P0=False, uniform_Ls=False, concentration_range_factor=10):
    """
    maximum a posterior
    :param q_actual_cal: observed heats in calorie
    :param exper_info: an object of _data_io.ITCExperiment class
    :param mcmc_trace: dict, "parameter" --> 1d ndarray
    :param dcell: float, relative uncertainty in cell concentration
    :param dsyringe: float, relative uncertainty in syringe concentration
    :param uniform_P0: bool
    :param uniform_Ls: bool
    :param concentration_range_factor: float
    :return: values of parameters that maximize the posterior
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

    V0 = exper_info.get_cell_volume_liter()
    DeltaVn = exper_info.get_injection_volumes_liter()
    beta = 1 / KB / exper_info.get_target_temperature_kelvin()
    n_injections = exper_info.get_number_injections()

    DeltaH_0_min, DeltaH_0_max = deltaH0_guesses(q_actual_cal)
    logsigma_min, logsigma_max = logsigma_guesses(q_actual_cal)

    log_probs = []

    for P0, Ls, rho, DeltaG1, DeltaDeltaG, DeltaH1, DeltaH2, DeltaH_0, log_sigma in zip(P0_trace, Ls_trace, rho_trace,
                                                                                        DeltaG1_trace,
                                                                                        DeltaDeltaG_trace,
                                                                                        DeltaH1_trace, DeltaH2_trace,
                                                                                        DeltaH_0_trace,
                                                                                        log_sigma_trace):
        q_model_cal = heats_RacemicMixtureBindingModel(V0, DeltaVn, P0, Ls, rho, DeltaH1, DeltaH2, DeltaH_0,
                                                       DeltaG1, DeltaDeltaG, beta, n_injections)
        sigma_cal = np.exp(log_sigma)
        log_prob = np.log(normal_likelihood(q_actual_cal, q_model_cal, sigma_cal))

        stated_P0 = exper_info.get_cell_concentration_milli_molar()
        if not uniform_P0:
            log_prob += np.log(lognormal_pdf(P0, stated_center=stated_P0, uncertainty=dcell * stated_P0))
        else:
            P0_min = stated_P0 / concentration_range_factor
            P0_max = stated_P0 * concentration_range_factor
            log_prob += np.log(uniform_pdf(P0, lower=P0_min, upper=P0_max))

        stated_Ls = exper_info.get_syringe_concentration_milli_molar()
        if not uniform_Ls:
            log_prob += np.log(lognormal_pdf(Ls, stated_center=stated_Ls, uncertainty=dsyringe * stated_Ls))
        else:
            Ls_min = stated_Ls / concentration_range_factor
            Ls_max = stated_Ls * concentration_range_factor
            log_prob += np.log(uniform_pdf(Ls, lower=Ls_min, upper=Ls_max))


        log_prob += np.log(uniform_pdf(rho, lower=0., upper=1.))


        log_prob += np.log(uniform_pdf(DeltaG1, lower=-40., upper=40.))
        log_prob += np.log(uniform_pdf(DeltaDeltaG, lower=0., upper=40.))

        log_prob += np.log(uniform_pdf(DeltaH1, lower=-100., upper=100.))
        log_prob += np.log(uniform_pdf(DeltaH2, lower=-100., upper=100.))

        log_prob += np.log(uniform_pdf(DeltaH_0, lower=DeltaH_0_min, upper=DeltaH_0_max))
        log_prob += np.log(uniform_pdf(log_sigma, lower=logsigma_min, upper=logsigma_max))

        log_probs.append(log_prob)

    map_idx = np.argmax(log_probs)
    print("Map index: %d" % map_idx)

    map_P0 = P0_trace[map_idx]
    map_Ls = Ls_trace[map_idx]
    map_rho = rho_trace[map_idx]
    map_DeltaG1 = DeltaG1_trace[map_idx]
    map_DeltaDeltaG = DeltaDeltaG_trace[map_idx]
    map_DeltaH1 = DeltaH1_trace[map_idx]
    map_DeltaH2 = DeltaH2_trace[map_idx]
    map_DeltaH_0 = DeltaH_0_trace[map_idx]

    return map_P0, map_Ls, map_rho, map_DeltaG1, map_DeltaDeltaG, map_DeltaH1, map_DeltaH2, map_DeltaH_0
