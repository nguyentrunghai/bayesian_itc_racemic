"""
contains functions calculate Bayesian Information Criterion (BIC)
and widely applicable Bayesian information Criterion
https://www.jmlr.org/papers/volume14/watanabe13a/watanabe13a.pdf
"""

from __future__ import print_function

import pickle
import copy
import numpy as np

from _bayes_factor import get_values_from_traces
from _bayes_factor import dict_to_list


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
