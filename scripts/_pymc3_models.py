"""
includes function that create model and run MCMC sampling
"""

import nump as np
import pymc3

from _models import heats_TwoComponentBindingModel, heats_RacemicMixtureBindingModel
from _models import logsigma_guesses, deltaH0_guesses


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
    