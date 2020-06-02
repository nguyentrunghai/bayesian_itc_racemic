"""
To compare confidence intervals between model with log-normal priors for concentrations and
model with flat priors for concentrations.
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

from _bayes_factor import get_values_from_trace, log_posterior_trace

parser = argparse.ArgumentParser()
parser.add_argument("--two_component_lognormal_dir", type=str,
                    default="/home/tnguye46/bayesian_itc_racemic/07.twocomponent_mcmc/pymc3_nuts_2")
parser.add_argument("--racemic_mixture_lognormal_dir", type=str,
                    default="/home/tnguye46/bayesian_itc_racemic/08.racemicmixture_mcmc/pymc3_nuts_2")
parser.add_argument("--enantiomer_lognormal_dir", type=str,
                    default="/home/tnguye46/bayesian_itc_racemic/09.enantiomer_mcmc/pymc3_nuts_2")

parser.add_argument("--two_component_flat_dir", type=str,
                    default="/home/tnguye46/bayesian_itc_racemic/07.twocomponent_mcmc/pymc3_nuts_4")
parser.add_argument("--racemic_mixture_flat_dir", type=str,
                    default="/home/tnguye46/bayesian_itc_racemic/08.racemicmixture_mcmc/pymc3_nuts_4")
parser.add_argument("--enantiomer_flat_dir", type=str,
                    default="/home/tnguye46/bayesian_itc_racemic/09.enantiomer_mcmc/pymc3_nuts_4")

parser.add_argument("--repeat_prefix", type=str, default="repeat_")

parser.add_argument("--trace_pickle", type=str, default="trace_obj.pickle")

parser.add_argument("--experiments", type=str,
default="Fokkens_1_c Fokkens_1_d Fokkens_1_e")

parser.add_argument("--font_scale", type=float, default=0.75)

args = parser.parse_args()

font_scale = args.font_scale

experiments = args.experiments.split()
print("experiments", experiments)


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


def plot_pair(ci_ln, ci_ft, y, color, var, ax):
    y_ln = y
    label_ln = var + " LN"
    plot_conf_interv(ci_ln, y_ln, label_ln, "solid", color, ax)

    y_ft = y + 0.2
    label_ft = var + " FT"
    plot_conf_interv(ci_ft, y_ft, label_ft, "dashed", color, ax)

    return ax


for exper in experiments:
    print("\n\n", exper)

    # 2c
    dirs_2c_ln = glob.glob(os.path.join(
        args.two_component_lognormal_dir, args.repeat_prefix + "*", exper, args.model_pickle))
    dirs_2c_ln = [os.path.dirname(p) for p in dirs_2c_ln]
    print("dirs_2c_ln:", dirs_2c_ln)

    dirs_2c_ft = glob.glob(
        os.path.join(args.two_component_flat_dir, args.repeat_prefix + "*", exper, args.model_pickle))
    dirs_2c_ft = [os.path.dirname(p) for p in dirs_2c_ft]
    print("dirs_2c_ft:", dirs_2c_ft)

    # rm
    dirs_rm_ln = glob.glob(os.path.join(
        args.racemic_mixture_lognormal_dir, args.repeat_prefix + "*", exper, args.model_pickle))
    dirs_rm_ln = [os.path.dirname(p) for p in dirs_rm_ln]
    print("dirs_rm_ln:", dirs_rm_ln)

    dirs_rm_ft = glob.glob(
        os.path.join(args.racemic_mixture_flat_dir, args.repeat_prefix + "*", exper, args.model_pickle))
    dirs_rm_ft = [os.path.dirname(p) for p in dirs_rm_ft]
    print("dirs_rm_ft:", dirs_rm_ft)

    # em
    dirs_em_ln = glob.glob(os.path.join(
        args.enantiomer_lognormal_dir, args.repeat_prefix + "*", exper, args.model_pickle))
    dirs_em_ln = [os.path.dirname(p) for p in dirs_em_ln]
    print("dirs_em_ln:", dirs_em_ln)

    dirs_em_ft = glob.glob(
        os.path.join(args.enantiomer_flat_dir, args.repeat_prefix + "*", exper, args.model_pickle))
    dirs_em_ft = [os.path.dirname(p) for p in dirs_em_ft]
    print("dirs_em_ft:", dirs_em_ft)

    traces_2c_ln = [pickle.load(open(os.path.join(d, args.trace_pickle))) for d in dirs_2c_ln]
    traces_2c_ft = [pickle.load(open(os.path.join(d, args.trace_pickle))) for d in dirs_2c_ft]

    traces_rm_ln = [pickle.load(open(os.path.join(d, args.trace_pickle))) for d in dirs_rm_ln]
    traces_rm_ft = [pickle.load(open(os.path.join(d, args.trace_pickle))) for d in dirs_rm_ft]

    traces_em_ln = [pickle.load(open(os.path.join(d, args.trace_pickle))) for d in dirs_em_ln]
    traces_em_ft = [pickle.load(open(os.path.join(d, args.trace_pickle))) for d in dirs_em_ft]

    tr_val_2c_ln = value_from_traces(traces_2c_ln)
    tr_val_2c_ft = value_from_traces(traces_2c_ft)

    tr_val_rm_ln = value_from_traces(traces_rm_ln)
    tr_val_rm_ft = value_from_traces(traces_rm_ft)

    tr_val_em_ln = value_from_traces(traces_em_ln)
    tr_val_em_ft = value_from_traces(traces_em_ft)

    ylim = [0, 4]

    # plot DeltaG ----------------------------------------------
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(9, 2.4))
    plt.subplots_adjust(wspace=0.02)
    sns.set(font_scale=font_scale)
    xlabel = "$\Delta G$ (kcal/mol)"

    # 2c
    ax = axes[0]
    y = 2
    var = "DeltaG"
    dg_ln = filter_outliers(tr_val_2c_ln[var])
    dg_ft = filter_outliers(tr_val_2c_ft[var])

    ci_ln = conf_interv(dg_ln)
    ci_ft = conf_interv(dg_ft)

    ax = plot_pair(ci_ln, ci_ft, y, "r", var, ax)
    ax.set_ylim(ylim)
    ax.set_xlabel(xlabel)
    ax.legend(loc="best")

    