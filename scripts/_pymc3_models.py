"""
includes function that create model and run MCMC sampling
"""

import nump as np
import pymc3

from _models import heats_TwoComponentBindingModel, heats_RacemicMixtureBindingModel
from _models import logsigma_guesses, deltaH0_guesses
from _models import KB


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
                           tau=1.0 / np.log(1 + (v / (m ** 2))),
                           value=m)


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
    logsigma_min, logsigma_max = logsigma_guesses(q_actual_cal)

    stated_P0 = exper_info.get_cell_concentration_milli_molar()
    uncertainty_P0 = dcell * stated_P0
    P0_min = stated_P0 / concentration_range_factor
    P0_max = stated_P0 * concentration_range_factor

    stated_Ls = exper_info.get_syringe_concentration_milli_molar()
    uncertainty_Ls = dsyringe * stated_Ls
    Ls_min = stated_Ls / concentration_range_factor
    Ls_max = stated_Ls * concentration_range_factor