"""
define function to perform MAP (Maximum a posterior) estimation
"""

from __future__ import print_function

import numpy as np

from _models import log_unnormalized_posterior_2cbm, log_unnormalized_posterior_rmbm, log_unnormalized_posterior_embm


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

    log_probs = log_unnormalized_posterior_embm(q_actual_cal, exper_info, mcmc_trace,
                                                dcell=dcell, dsyringe=dsyringe,
                                                uniform_P0=uniform_P0, uniform_Ls=uniform_Ls,
                                                concentration_range_factor=concentration_range_factor)
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
