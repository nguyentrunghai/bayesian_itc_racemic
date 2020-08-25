"""
contains functions calculate Bayesian Information Criterion (BIC)
and widely applicable Bayesian information Criterion
https://www.jmlr.org/papers/volume14/watanabe13a/watanabe13a.pdf
"""

from __future__ import print_function

import pickle
import copy
import numpy as np

from _data_io_py3 import load_heat_micro_cal
from _bayes_factor import get_values_from_traces
from _bayes_factor import dict_to_list


def get_n_injections(heat_file):
    return len(load_heat_micro_cal(heat_file))


def get_n_params(model):
    return len(model.free_RVs)


def get_values_from_trace_files(model, trace_pkl_files, thin=1, burn=0):
    traces = [pickle.load(open(trace_file)) for trace_file in trace_pkl_files]
    trace_values = get_values_from_traces(model, traces, thin=thin, burn=burn)
    return trace_values


def load_model(model_pkl):
    return pickle.load(open(model_pkl))


def log_likelihood_trace(model, trace_values):

    model_vars = [var.name for var in model.vars]
    trace_vars = trace_values.keys()

    trace_v = {}
    for var in model_vars:
        if var not in trace_vars:
            raise KeyError(var + " is not in trace")
        trace_v[var] = copy.deepcopy(trace_values[var])

    trace_v = dict_to_list(trace_v)
    obs_q = model.observed_RVs[0]
    get_logp = np.vectorize(obs_q.logp)
    logp = get_logp(trace_v)
    return logp


def bic(log_llhs, n_samples, n_params):
    """
    bic = -2 \ln p(y | \theta_{MLE}) + \ln(n)k
    Ref: Andrew Gelman et al., "Bayesian Data Analysis", 3rd Ed., CRC Press, page 169
    :param log_llhs: array-like of float, log likelihood values
    :param n_samples: int, number of samples
    :param n_params: number of parameters
    :return: float, Bayesian information criterion
    """
    return -2 * np.max(log_llhs) + np.log(n_samples) * n_params


def bic_bootstrap(log_llhs, n_samples, n_params, repeats=1000):
    bic_val = bic(log_llhs, n_samples, n_params)

    bic_boostrap_vals = []
    size = len(log_llhs)
    for _ in range(repeats):
        rnd_log_llhs = np.random.choice(log_llhs, size=size, replace=True)
        b = bic(rnd_log_llhs, n_samples, n_params)
        bic_boostrap_vals.append(b)

    bic_std = np.std(bic_boostrap_vals)
    return bic_val, bic_std


def wbic(log_llhs, n_samples):
    beta = 1. / np.log(n_samples)
    weights = log_llhs * (beta - 1)
    weights -= weights.max()
    weights = np.exp(weights)

    result = np.sum(-log_llhs * weights) / np.sum(weights)
    return result


def wbic_bootstrap(log_llhs, n_samples, repeats=1000):
    wbic_val = wbic(log_llhs, n_samples)

    wbic_vals = []
    size = len(log_llhs)
    for _ in range(repeats):
        rnd_log_llhs = np.random.choice(log_llhs, size=size, replace=True)
        b = wbic(rnd_log_llhs, n_samples)
        wbic_vals.append(b)

    wbic_std = np.std(wbic_vals)
    return wbic_val, wbic_std
