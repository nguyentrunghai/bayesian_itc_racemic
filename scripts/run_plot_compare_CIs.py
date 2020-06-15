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

parser.add_argument("--experiments", type=str, default="Fokkens_1_c Fokkens_1_d Fokkens_1_e")

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
        args.two_component_lognormal_dir, args.repeat_prefix + "*", exper, args.trace_pickle))
    dirs_2c_ln = [os.path.dirname(p) for p in dirs_2c_ln]
    print("dirs_2c_ln:", dirs_2c_ln)

    dirs_2c_ft = glob.glob(
        os.path.join(args.two_component_flat_dir, args.repeat_prefix + "*", exper, args.trace_pickle))
    dirs_2c_ft = [os.path.dirname(p) for p in dirs_2c_ft]
    print("dirs_2c_ft:", dirs_2c_ft)

    # rm
    dirs_rm_ln = glob.glob(os.path.join(
        args.racemic_mixture_lognormal_dir, args.repeat_prefix + "*", exper, args.trace_pickle))
    dirs_rm_ln = [os.path.dirname(p) for p in dirs_rm_ln]
    print("dirs_rm_ln:", dirs_rm_ln)

    dirs_rm_ft = glob.glob(
        os.path.join(args.racemic_mixture_flat_dir, args.repeat_prefix + "*", exper, args.trace_pickle))
    dirs_rm_ft = [os.path.dirname(p) for p in dirs_rm_ft]
    print("dirs_rm_ft:", dirs_rm_ft)

    # em
    dirs_em_ln = glob.glob(os.path.join(
        args.enantiomer_lognormal_dir, args.repeat_prefix + "*", exper, args.trace_pickle))
    dirs_em_ln = [os.path.dirname(p) for p in dirs_em_ln]
    print("dirs_em_ln:", dirs_em_ln)

    dirs_em_ft = glob.glob(
        os.path.join(args.enantiomer_flat_dir, args.repeat_prefix + "*", exper, args.trace_pickle))
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

    ylim = [0, 2]

    # plot DeltaG ----------------------------------------------
    fig, axes = plt.subplots(nrows=2, ncols=3, sharey=True, figsize=(9, 4.8))
    plt.subplots_adjust(wspace=0.02)
    sns.set(font_scale=font_scale)
    #xlabel = "$\Delta G$ (kcal/mol)"

    # 2c
    ax = axes[0, 0]
    y = 1.
    var = "$\Delta G$ (kcal/mol)"
    dg_ln = filter_outliers(tr_val_2c_ln["DeltaG"])
    dg_ft = filter_outliers(tr_val_2c_ft["DeltaG"])

    ci_ln = conf_interv(dg_ln)
    ci_ft = conf_interv(dg_ft)

    ax = plot_pair(ci_ln, ci_ft, y, "r", var, ax)
    ax.set_ylim(ylim)
    ax.set_xlabel(var)
    #ax.legend(loc="best")
    ax.set_title("Two-Component")

    # rm
    y1 = 1.
    y2 = 1.
    var1 = "$\Delta G_1$ (kcal/mol)"
    var2 = "$\Delta \Delta G$ (kcal/mol)"

    dg1_ln = tr_val_rm_ln["DeltaG1"]
    dg1_ft = tr_val_rm_ft["DeltaG1"]

    ddg_ln = tr_val_rm_ln["DeltaDeltaG"]
    ddg_ft = tr_val_rm_ft["DeltaDeltaG"]

    dg1_ln = filter_outliers(dg1_ln)
    dg1_ft = filter_outliers(dg1_ft)

    ddg_ln = filter_outliers(ddg_ln)
    ddg_ft = filter_outliers(ddg_ft)

    ci_1_ln = conf_interv(dg1_ln)
    ci_1_ft = conf_interv(dg1_ft)

    ci_2_ln = conf_interv(ddg_ln)
    ci_2_ft = conf_interv(ddg_ft)

    ax = axes[0, 1]
    ax = plot_pair(ci_1_ln, ci_1_ft, y1, "r", var1, ax)
    ax.set_ylim(ylim)
    ax.set_xlabel(var1)
    #ax.legend(loc="best")
    ax.set_title("Racemic Mixture")

    ax = axes[0, 2]
    ax = plot_pair(ci_2_ln, ci_2_ft, y2, "b", var2, ax)
    ax.set_ylim(ylim)
    ax.set_xlabel(var2)
    #ax.legend(loc="best")
    ax.set_title("Racemic Mixture")

    # em
    y1 = 1.
    y2 = 1.
    var1 = "$\Delta G_1$ (kcal/mol)"
    var2 = "$\Delta \Delta G$ (kcal/mol)"

    dg1_ln = tr_val_em_ln["DeltaG1"]
    dg1_ft = tr_val_em_ft["DeltaG1"]

    ddg_ln = tr_val_em_ln["DeltaDeltaG"]
    ddg_ft = tr_val_em_ft["DeltaDeltaG"]

    dg1_ln = filter_outliers(dg1_ln)
    dg1_ft = filter_outliers(dg1_ft)

    ddg_ln = filter_outliers(ddg_ln)
    ddg_ft = filter_outliers(ddg_ft)

    ci_1_ln = conf_interv(dg1_ln)
    ci_1_ft = conf_interv(dg1_ft)

    ci_2_ln = conf_interv(ddg_ln)
    ci_2_ft = conf_interv(ddg_ft)

    ax = axes[1, 0]
    ax = plot_pair(ci_1_ln, ci_1_ft, y1, "r", var1, ax)
    ax.set_ylim(ylim)
    ax.set_xlabel(var1)
    #ax.legend(loc="best")
    ax.set_title("Enantiomer")

    ax = axes[1, 1]
    ax = plot_pair(ci_2_ln, ci_2_ft, y2, "b", var2, ax)
    ax.set_ylim(ylim)
    ax.set_xlabel(var2)
    #ax.legend(loc="best")
    ax.set_title("Enantiomer")

    out = exper + "_DeltaG.pdf"
    fig.tight_layout()
    fig.savefig(out, dpi=300)

    # plot DeltaH ----------------------------------------------
    fig, axes = plt.subplots(nrows=2, ncols=3, sharey=True, figsize=(9, 4.8))
    plt.subplots_adjust(wspace=0.02)
    sns.set(font_scale=font_scale)
    # xlabel = "$\Delta G$ (kcal/mol)"

    # 2c
    ax = axes[0, 0]
    y = 1.
    var = "$\Delta H$ (kcal/mol)"
    dh_ln = filter_outliers(tr_val_2c_ln["DeltaH"])
    dh_ft = filter_outliers(tr_val_2c_ft["DeltaH"])

    ci_ln = conf_interv(dh_ln)
    ci_ft = conf_interv(dh_ft)

    ax = plot_pair(ci_ln, ci_ft, y, "r", var, ax)
    ax.set_ylim(ylim)
    ax.set_xlabel(var)
    # ax.legend(loc="best")
    ax.set_title("Two-Component")

    # rm
    y1 = 1.
    y2 = 1.
    var1 = "$\Delta H_1$ (kcal/mol)"
    var2 = "$\Delta H_2$ (kcal/mol)"

    dh1_ln = filter_outliers(tr_val_rm_ln["DeltaH1"])
    dh1_ft = filter_outliers(tr_val_rm_ft["DeltaH1"])

    dh2_ln = filter_outliers(tr_val_rm_ln["DeltaH2"])
    dh2_ft = filter_outliers(tr_val_rm_ft["DeltaH2"])

    ci_1_ln = conf_interv(dh1_ln)
    ci_1_ft = conf_interv(dh1_ft)

    ci_2_ln = conf_interv(dh2_ln)
    ci_2_ft = conf_interv(dh2_ft)

    ax = axes[0, 1]
    ax = plot_pair(ci_1_ln, ci_1_ft, y1, "r", var1, ax)
    ax.set_ylim(ylim)
    ax.set_xlabel(var1)
    # ax.legend(loc="best")
    ax.set_title("Racemic Mixture")

    ax = axes[0, 2]
    ax = plot_pair(ci_2_ln, ci_2_ft, y2, "b", var2, ax)
    ax.set_ylim(ylim)
    ax.set_xlabel(var2)
    # ax.legend(loc="best")
    ax.set_title("Racemic Mixture")

    # em
    y1 = 1.
    y2 = 1.
    var1 = "$\Delta H_1$ (kcal/mol)"
    var2 = "$\Delta H_2$ (kcal/mol)"

    dh1_ln = filter_outliers(tr_val_em_ln["DeltaH1"])
    dh1_ft = filter_outliers(tr_val_em_ft["DeltaH1"])

    dh2_ln = filter_outliers(tr_val_em_ln["DeltaH2"])
    dh2_ft = filter_outliers(tr_val_em_ft["DeltaH2"])

    ci_1_ln = conf_interv(dh1_ln)
    ci_1_ft = conf_interv(dh1_ft)

    ci_2_ln = conf_interv(dh2_ln)
    ci_2_ft = conf_interv(dh2_ft)

    ax = axes[1, 0]
    ax = plot_pair(ci_1_ln, ci_1_ft, y1, "r", var1, ax)
    ax.set_ylim(ylim)
    ax.set_xlabel(var1)
    # ax.legend(loc="best")
    ax.set_title("Enantiomer")

    ax = axes[1, 1]
    ax = plot_pair(ci_2_ln, ci_2_ft, y2, "b", var2, ax)
    ax.set_ylim(ylim)
    ax.set_xlabel(var2)
    # ax.legend(loc="best")
    ax.set_title("Enantiomer")

    out = exper + "_DeltaH.pdf"
    fig.tight_layout()
    fig.savefig(out, dpi=300)

    # plot concentration --------------------------------
    fig, axes = plt.subplots(nrows=3, ncols=3, sharey=True, figsize=(9, 7.4))
    plt.subplots_adjust(wspace=0.02)
    sns.set(font_scale=font_scale)

    # Ls 2c
    y1 = 1.
    y2 = 1.
    var1 = "$[L]_s$ (mM)"
    var2 = "$[L]_s$ (mM)"
    ax = axes[0, 0]

    ls_ln = filter_outliers(tr_val_2c_ln["Ls"])
    ls_ft = filter_outliers(tr_val_2c_ft["Ls"])

    ci_ln = conf_interv(ls_ln)
    ci_ft = conf_interv(ls_ln)

    ax = plot_pair(ci_1_ln, ci_1_ft, y1, "r", var1, ax)
    ax.set_ylim(ylim)
    ax.set_xlabel(var1)
    # ax.legend(loc="best")
    ax.set_title("Two-Component")

    # Ls rm
    y1 = 1.
    y2 = 1.
    var1 = "$[L]_s$ (mM)"
    var2 = "$[L]_s$ (mM)"
    ax = axes[0, 1]

    ls_ln = filter_outliers(tr_val_rm_ln["Ls"])
    ls_ft = filter_outliers(tr_val_rm_ft["Ls"])

    ci_ln = conf_interv(ls_ln)
    ci_ft = conf_interv(ls_ln)

    ax = plot_pair(ci_1_ln, ci_1_ft, y1, "r", var1, ax)
    ax.set_ylim(ylim)
    ax.set_xlabel(var1)
    # ax.legend(loc="best")
    ax.set_title("Racemic Mixture")

    # Ls em
    y1 = 1.
    y2 = 1.
    var1 = "$[L]_s$ (mM)"
    var2 = "$[L]_s$ (mM)"
    ax = axes[0, 2]

    ls_ln = filter_outliers(tr_val_em_ln["Ls"])
    ls_ft = filter_outliers(tr_val_em_ft["Ls"])

    ci_ln = conf_interv(ls_ln)
    ci_ft = conf_interv(ls_ln)

    ax = plot_pair(ci_1_ln, ci_1_ft, y1, "r", var1, ax)
    ax.set_ylim(ylim)
    ax.set_xlabel(var1)
    # ax.legend(loc="best")
    ax.set_title("Enantiomer")