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


def plot_conf_intervs(cis, ys, label, linestyle, color, ax):
    for i, (ci, y) in enumerate(zip(cis, ys)):
        if i == 0:
            label = label
        else:
            label = None

        plot_conf_interv(ci, y, label, linestyle, color, ax)
    return ax


for exper in experiments:
    print("\n\n", exper)

    # load 2c
    traces_2c = []
    for prior_dir in prior_dirs:

        dirs = glob.glob(os.path.join( args.two_component_dir, prior_dir, args.repeat_prefix + "*",
                                       exper, args.trace_pickle))
        dirs = [os.path.dirname(p) for p in dirs]
        print("Loading traces from", dirs)
        trace = [pickle.load(open(os.path.join(d, args.trace_pickle))) for d in dirs]

        traces_2c.append(value_from_traces(trace))

    # load rm
    traces_rm = []
    for prior_dir in prior_dirs:

        dirs = glob.glob(os.path.join(args.racemic_mixture_dir, prior_dir, args.repeat_prefix + "*",
                                      exper, args.trace_pickle))
        dirs = [os.path.dirname(p) for p in dirs]
        print("Loading traces from", dirs)
        trace = [pickle.load(open(os.path.join(d, args.trace_pickle))) for d in dirs]

        traces_rm.append(value_from_traces(trace))

    # load em
    traces_em = []
    for prior_dir in prior_dirs:

        dirs = glob.glob(os.path.join(args.enantiomer_dir, prior_dir, args.repeat_prefix + "*",
                                      exper, args.trace_pickle))
        dirs = [os.path.dirname(p) for p in dirs]
        print("Loading traces from", dirs)
        trace = [pickle.load(open(os.path.join(d, args.trace_pickle))) for d in dirs]

        traces_em.append(value_from_traces(trace))

    fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(11.4, 7.2))
    plt.subplots_adjust(wspace=0.02)
    sns.set(font_scale=font_scale)

    # plot 2c, DG
    ax = axes[0, 0]
    cis = [conf_interv(trace_2c["DeltaG"]) for trace_2c in traces_2c]
    ys = list(range(1, len(cis)+1))
    plot_conf_intervs(cis, ys, label=None, linestyle="-", color="k", ax=ax)
    ylim = [ys[0]-1, ys[-1]+1]
    ax.set_ylim(ylim)
    ax.set_xlabel("$\Delta G (kcal/mol)$")
    ax.set_ylabel("Two-Component")
    ax.set_yticks([])
    ax.set_yticklabels([])

    # plot 2c, DH
    ax = axes[0, 1]
    cis = [conf_interv(trace_2c["DeltaH"]) for trace_2c in traces_2c]
    ys = list(range(1, len(cis) + 1))
    plot_conf_intervs(cis, ys, label=None, linestyle="-", color="k", ax=ax)
    ylim = [ys[0] - 1, ys[-1] + 1]
    ax.set_ylim(ylim)
    ax.set_xlabel("$\Delta H (kcal/mol)$")
    ax.set_yticks([])
    ax.set_yticklabels([])

    out = exper + ".pdf"
    fig.tight_layout()
    fig.savefig(out, dpi=300)

