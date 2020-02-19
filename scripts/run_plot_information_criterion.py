"""
plot AIC (Akaike information criterion), BIC (Bayesian nformation criterion) and
DIC (deviance information criterion)
"""

import numpy as np

from _pymc3_models import extract_loglhs_from_traces_pymc3


def aic(log_llhs, k):
    """
    aic = -2 \ln p(y | \theta_{MLE}) + 2k
    Ref: Andrew Gelman et al., "Bayesian Data Analysis", 3rd Ed., CRC Press, page 169
    :param log_llhs: array-like of float, log likelihood values
    :param k: number of parameters
    :return: float, Akaike information criterion
    """
    return -2 * np.max(log_llhs) + 2 * k


def bic(log_llhs, n, k):
    """
    bic = -2 \ln p(y | \theta_{MLE}) + \ln(n)k
    Ref: Andrew Gelman et al., "Bayesian Data Analysis", 3rd Ed., CRC Press, page 169
    :param log_llhs: array-like of float, log likelihood values
    :param n: int, number of paramters
    :param k: number of parameters
    :return: float, Bayesian information criterion
    """
    return -2 * np.max(log_llhs) + np.log(n) * k


def dic_1(traces, log_llhs,
          model_name, exper_info_file, heat_file,
          dcell=0.1, dsyringe=0.1,
          uniform_P0=False, uniform_Ls=False,
          concentration_range_factor=10.,
          auto_transform=False):
    """
    dic_1 = 2 \ln p(y|\hat{\theta}_{Bayes}) - 2 \frac{1}{S} \sum_{s=1}^S \ln p(y|\theta^s)
    where \hat{\theta}_{Bayes} is mean of the posterior

    :param traces: dict, variable_name -> 1d array
    :param log_llhs: array-like of float, log likelihood values
    :param model_name: str, in ["2cbm", "rmbm", "embm"]
    :param exper_info_file: str
    :param heat_file: str
    :param dcell: float in (0, 1)
    :param dsyringe: float in (0, 1)
    :param uniform_P0: bool
    :param uniform_Ls: bool
    :param concentration_range_factor: float
    :param auto_transform: bool
    :return: llhs, 1d array
    """
    posterior_mean = {var_name: [np.mean(var_trace)] for var_name, var_trace in traces.items()}
    _, log_llh_bayes = extract_loglhs_from_traces_pymc3(posterior_mean, model_name, exper_info_file, heat_file,
                                                        dcell=dcell, dsyringe=dsyringe,
                                                        uniform_P0=uniform_P0, uniform_Ls=uniform_Ls,
                                                        concentration_range_factor=concentration_range_factor,
                                                        auto_transform=auto_transform)
    log_llh_bayes = log_llh_bayes[0]
    return 2 * log_llh_bayes - 2 * np.mean(log_llhs)


def dic_2(traces, log_llhs,
          model_name, exper_info_file, heat_file,
          dcell=0.1, dsyringe=0.1,
          uniform_P0=False, uniform_Ls=False,
          concentration_range_factor=10.,
          auto_transform=False):
    """
    dic_1 = 2 \ln p(y|\hat{\theta}_{Bayes}) - 2 \frac{1}{S} \sum_{s=1}^S \ln p(y|\theta^s)
    where \hat{\theta}_{Bayes} is mean of the posterior

    :param traces: dict, variable_name -> 1d array
    :param log_llhs: array-like of float, log likelihood values
    :param model_name: str, in ["2cbm", "rmbm", "embm"]
    :param exper_info_file: str
    :param heat_file: str
    :param dcell: float in (0, 1)
    :param dsyringe: float in (0, 1)
    :param uniform_P0: bool
    :param uniform_Ls: bool
    :param concentration_range_factor: float
    :param auto_transform: bool
    :return: llhs, 1d array
    """
    posterior_mean = {var_name: [np.mean(var_trace)] for var_name, var_trace in traces.items()}
    _, log_llh_bayes = extract_loglhs_from_traces_pymc3(posterior_mean, model_name, exper_info_file, heat_file,
                                                        dcell=dcell, dsyringe=dsyringe,
                                                        uniform_P0=uniform_P0, uniform_Ls=uniform_Ls,
                                                        concentration_range_factor=concentration_range_factor,
                                                        auto_transform=auto_transform)
    log_llh_bayes = log_llh_bayes[0]
    return -2 * log_llh_bayes + 4 * np.var(log_llhs)
