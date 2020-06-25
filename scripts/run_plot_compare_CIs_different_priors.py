"""
To compare confidence intervals between different prior choice
"""

from __future__ import print_function

import argparse
import os
import pickle
import glob

import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

parser = argparse.ArgumentParser()
parser.add_argument("--two_component_dir", type=str,
                    default="/home/tnguye46/bayesian_itc_racemic/07.twocomponent_mcmc")
parser.add_argument("--racemic_mixture_dir", type=str,
                    default="/home/tnguye46/bayesian_itc_racemic/08.racemicmixture_mcmc")
parser.add_argument("--enantiomer_dir", type=str,
                    default="/home/tnguye46/bayesian_itc_racemic/09.enantiomer_mcmc")

parser.add_argument("--prior_dirs", type=str, default="pymc3_nuts_2 pymc3_nuts_4 pymc3_nuts_6 pymc3_nuts_5")

parser.add_argument("--experiments", type=str, default="Fokkens_1_c Fokkens_1_d Fokkens_1_e")

parser.add_argument("--repeat_prefix", type=str, default="repeat_")

parser.add_argument("--trace_pickle", type=str, default="trace_obj.pickle")

parser.add_argument("--font_scale", type=float, default=0.75)

args = parser.parse_args()

font_scale = args.font_scale

experiments = args.experiments.split()
print("experiments", experiments)

prior_dirs = args.prior_dirs.split()
print("prior_dirs", prior_dirs)


def value_from_trace(trace):
    free_vars = [name for name in trace.varnames if not name.endswith("__")]
    tr_val = {name: trace.get_values(name) for name in free_vars}
    return tr_val


def value_from_traces(traces):
    trace_value_list = [value_from_trace(t) for t in traces]
    keys = trace_value_list[0].keys()
    trace_values = {}
    for key in keys:
        trace_values[key] = np.concatenate([tr_val[key] for tr_val in trace_value_list])
    return trace_values


def conf_interv(x, conf_level=95.):
    alpha = 100 - conf_level
    lower = np.percentile(x, alpha/2)
    upper = np.percentile(x, 100 - (alpha/2))
    return lower, upper


def filter_outliers(x, thres=99.):
    lower, upper = conf_interv(x, conf_level=thres)
    keep = (x > lower) & (x < upper)
    return x[keep]


def plot_conf_interv(ci, y, label, linestyle, color, ax):
    lower, upper = ci
    ax.plot([lower, upper], [y, y], color=color, linestyle=linestyle, marker="|", label=label)
    return ax



