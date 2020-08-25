"""
contains functions calculate Bayesian Information Criterion (BIC)
and widely applicable Bayesian information Criterion
https://www.jmlr.org/papers/volume14/watanabe13a/watanabe13a.pdf
"""

from __future__ import print_function

import pickle

import numpy as np

from _bayes_factor import get_values_from_traces
from _bayes_factor import dict_to_list


def get_values_from_trace_files(model, trace_pkl_files, thin=1, burn=0):
    traces = [pickle.load(open(trace_file)) for trace_file in trace_pkl_files]
    trace_values = get_values_from_traces(model, traces, thin=thin, burn=burn)
    return trace_values


def load_model(model_pkl):
    return pickle.load(open(model_pkl))