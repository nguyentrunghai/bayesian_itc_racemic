"""
to compare histogram of DG, DH, P0 and Ls between twocomponent and racemicmixture models
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
parser.add_argument("--two_component_mcmc_dir", type=str,
                    default="/home/tnguye46/bayesian_itc_racemic/07.twocomponent_mcmc/pymc3_nuts_2")
parser.add_argument("--racemic_mixture_mcmc_dir", type=str,
                    default="/home/tnguye46/bayesian_itc_racemic/08.racemicmixture_mcmc/pymc3_nuts_2")
parser.add_argument("--enantiomer_mcmc_dir", type=str,
                    default="/home/tnguye46/bayesian_itc_racemic/09.enantiomer_mcmc/pymc3_nuts_2")

parser.add_argument("--repeat_prefix", type=str, default="repeat_")
parser.add_argument("--exclude_repeats", type=str, default="")

parser.add_argument("--trace_pickle", type=str, default="trace_obj.pickle")
parser.add_argument("--model_pickle", type=str, default="pm_model.pickle")

parser.add_argument("--experiments", type=str,
default="Fokkens_1_a Fokkens_1_b Fokkens_1_c Fokkens_1_d Fokkens_1_e Baum_57 Baum_59 Baum_60_1 Baum_60_2 Baum_60_3 Baum_60_4")

parser.add_argument("--font_scale", type=float, default=0.75)

args = parser.parse_args()


def is_path_excluded(path, exclude_kws):
    for kw in exclude_kws:
        if kw in path:
            return True
    return False


def find_MAP_trace(model, trace):
    tr_val = get_values_from_trace(model, trace)
    logp = log_posterior_trace(model, tr_val)

    idx_max = np.argmax(logp)
    map_logp = logp[idx_max]

    free_vars = [name for name in trace.varnames if not name.endswith("__")]
    map_val = {name: trace.get_values(name)[idx_max] for name in free_vars}
    return map_val, map_logp


def find_MAP_traces(model, traces):
    map_results = [find_MAP_trace(model, trace) for trace in traces]
    map_results.sort(key=lambda i: i[1])
    map_val = map_results[-1][0]
    return map_val


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


def plot_kde_hist(data_list, labels, colors, ax):

    for data, label, color in zip(data_list, labels, colors):
        sns.kdeplot(data, ax=ax, label=label, c=color)

    ax.legend(loc="best")
    return ax


def plot_conf_intervs(conf_intervs, colors, ax):
    y_low, y_high = ax.yaxis.get_data_interval()
    n = len(conf_intervs)
    ys = np.linspace(y_low, y_high, n + 2)
    ys = ys[1:-1]

    for (xl, xh), y, c in zip(conf_intervs, ys, colors):
        ax.plot([xl, xh], [y, y], color=c, linestyle="solid", marker="|")
    return ax


def plot_maps(maps, colors, ax):
    y_low, y_high = ax.yaxis.get_data_interval()
    yrange = y_high - y_low
    y = y_low + yrange / 100.

    for map, c in zip(maps, colors):
        ax.scatter([map], [y], color=c, marker="v")
    return ax


exclude_repeats = args.exclude_repeats.split()
exclude_repeats = [args.repeat_prefix + r for r in exclude_repeats]
print("exclude_repeats:", exclude_repeats)

experiments = args.experiments.split()
print("experiments", experiments)

colors = ("r", "b", "g")
ylabel = "Probability density"

font_scale = args.font_scale

for exper in experiments:
    print("\n\n", exper)

    dirs_2c = glob.glob(os.path.join(args.two_component_mcmc_dir, args.repeat_prefix + "*", exper, args.model_pickle))
    dirs_2c = [os.path.dirname(p) for p in dirs_2c]
    dirs_2c = [p for p in dirs_2c if not is_path_excluded(p, exclude_repeats)]
    print("dirs_2c:", dirs_2c)

    dirs_rm = glob.glob(os.path.join(args.racemic_mixture_mcmc_dir, args.repeat_prefix + "*", exper, args.model_pickle))
    dirs_rm = [os.path.dirname(p) for p in dirs_rm]
    dirs_rm = [p for p in dirs_rm if not is_path_excluded(p, exclude_repeats)]
    print("dirs_rm:", dirs_rm)

    dirs_em = glob.glob(os.path.join(args.enantiomer_mcmc_dir, args.repeat_prefix + "*", exper, args.model_pickle))
    dirs_em = [os.path.dirname(p) for p in dirs_em]
    dirs_em = [p for p in dirs_em if not is_path_excluded(p, exclude_repeats)]
    print("dirs_em:", dirs_em)

    model_2c = pickle.load(open(os.path.join(dirs_2c[0], args.model_pickle)))
    model_rm = pickle.load(open(os.path.join(dirs_rm[0], args.model_pickle)))
    model_em = pickle.load(open(os.path.join(dirs_em[0], args.model_pickle)))

    traces_2c = [pickle.load(open(os.path.join(d, args.trace_pickle))) for d in dirs_2c]
    traces_rm = [pickle.load(open(os.path.join(d, args.trace_pickle))) for d in dirs_rm]
    traces_em = [pickle.load(open(os.path.join(d, args.trace_pickle))) for d in dirs_em]

    tr_val_2c = value_from_traces(traces_2c)
    tr_val_rm = value_from_traces(traces_rm)
    tr_val_em = value_from_traces(traces_em)

    #map_2c = find_MAP_traces(model_2c, traces_2c)
    #print("map_2c", map_2c)

    #map_rm = find_MAP_traces(model_rm, traces_rm)
    #print("map_rm", map_rm)

    #map_em = find_MAP_traces(model_em, traces_em)
    #print("map_em", map_em)

    print("\nConfidence intervals")
    print("----------------------------------")
    # plot DeltaG ----------------------------------------------
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(9, 2.4))
    plt.subplots_adjust(wspace=0.02)
    sns.set(font_scale=font_scale)
    ylabel = "Probability density"
    xlabel = "$\Delta G$ (kcal/mol)"

    # 2c
    ax = axes[0]
    xs = [tr_val_2c["DeltaG"]]
    xs = [filter_outliers(x) for x in xs]
    #maps = [map_2c["DeltaG"]]
    cis = [conf_interv(x) for x in xs]
    labels = ["$\Delta G$"]
    plot_kde_hist(xs, labels, colors, ax)
    plot_conf_intervs(cis, colors, ax)
    #plot_maps(maps, colors, ax)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title("Two-Component")

    ci_str = "[%0.2f, %0.2f]" % cis[0]
    print(exper + ", 2C, DeltaG: " + ci_str)

    # rm
    ax = axes[1]
    xs = [tr_val_rm["DeltaG1"], tr_val_rm["DeltaG1"] + tr_val_rm["DeltaDeltaG"]]
    xs = [filter_outliers(x) for x in xs]
    #maps = [map_rm["DeltaG1"], map_rm["DeltaG1"] + map_rm["DeltaDeltaG"]]
    cis = [conf_interv(x) for x in xs]
    labels = ["$\Delta G_1$", "$\Delta G_2$"]
    plot_kde_hist(xs, labels, colors, ax)
    plot_conf_intervs(cis, colors, ax)
    #plot_maps(maps, colors, ax)
    ax.set_xlabel(xlabel)
    ax.set_title("Racemic Mixture")

    ci1_str = "[%0.2f, %0.2f]" % cis[0]
    ci2_str = "[%0.2f, %0.2f]" % cis[1]
    print(exper + ", RM, DeltaG1: " + ci1_str)
    print(exper + ", RM, DeltaG2: " + ci2_str)

    # em
    ax = axes[2]
    xs = [tr_val_em["DeltaG1"], tr_val_em["DeltaG1"] + tr_val_em["DeltaDeltaG"]]
    xs = [filter_outliers(x) for x in xs]
    #maps = [map_em["DeltaG1"], map_em["DeltaG1"] + map_em["DeltaDeltaG"]]
    cis = [conf_interv(x) for x in xs]
    labels = ["$\Delta G_1$", "$\Delta G_2$"]
    plot_kde_hist(xs, labels, colors, ax)
    plot_conf_intervs(cis, colors, ax)
    #plot_maps(maps, colors, ax)
    ax.set_xlabel(xlabel)
    ax.set_title("Enantiomer")

    ci1_str = "[%0.2f, %0.2f]" % cis[0]
    ci2_str = "[%0.2f, %0.2f]" % cis[1]
    print(exper + ", EM, DeltaG1: " + ci1_str)
    print(exper + ", EM, DeltaG2: " + ci2_str)
    print("")

    out = exper + "_DeltaG.pdf"
    fig.tight_layout()
    fig.savefig(out, dpi=300)


    # plot DeltaH ----------------------------------------------
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(9, 2.4))
    plt.subplots_adjust(wspace=0.02)
    sns.set(font_scale=font_scale)
    ylabel = "Probability density"
    xlabel = "$\Delta H$ (kcal/mol)"

    # 2c
    ax = axes[0]
    xs = [tr_val_2c["DeltaH"]]
    xs = [filter_outliers(x) for x in xs]
    #maps = [map_2c["DeltaH"]]
    cis = [conf_interv(x) for x in xs]
    labels = ["$\Delta H$"]
    plot_kde_hist(xs, labels, colors, ax)
    plot_conf_intervs(cis, colors, ax)
    #plot_maps(maps, colors, ax)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title("Two-Component")

    ci_str = "[%0.2f, %0.2f]" % cis[0]
    print(exper + ", 2C, DeltaH: " + ci_str)

    # rm
    ax = axes[1]
    xs = [tr_val_rm["DeltaH1"], tr_val_rm["DeltaH2"]]
    xs = [filter_outliers(x) for x in xs]
    #maps = [map_rm["DeltaH1"], map_rm["DeltaH2"]]
    cis = [conf_interv(x) for x in xs]
    labels = ["$\Delta H_1$", "$\Delta H_2$"]
    plot_kde_hist(xs, labels, colors, ax)
    plot_conf_intervs(cis, colors, ax)
    #plot_maps(maps, colors, ax)
    ax.set_xlabel(xlabel)
    ax.set_title("Racemic Mixture")

    ci1_str = "[%0.2f, %0.2f]" % cis[0]
    ci2_str = "[%0.2f, %0.2f]" % cis[1]
    print(exper + ", RM, DeltaH1: " + ci1_str)
    print(exper + ", RM, DeltaH2: " + ci2_str)

    # em
    ax = axes[2]
    xs = [tr_val_em["DeltaH1"], tr_val_em["DeltaH2"]]
    xs = [filter_outliers(x) for x in xs]
    #maps = [map_em["DeltaH1"], map_em["DeltaH2"]]
    cis = [conf_interv(x) for x in xs]
    labels = ["$\Delta H_1$", "$\Delta H_2$"]
    plot_kde_hist(xs, labels, colors, ax)
    plot_conf_intervs(cis, colors, ax)
    #plot_maps(maps, colors, ax)
    ax.set_xlabel(xlabel)
    ax.set_title("Enantiomer")

    ci1_str = "[%0.2f, %0.2f]" % cis[0]
    ci2_str = "[%0.2f, %0.2f]" % cis[1]
    print(exper + ", EM, DeltaH1: " + ci1_str)
    print(exper + ", EM, DeltaH2: " + ci2_str)
    print("-------------------------------------")

    out = exper + "_DeltaH.pdf"
    fig.tight_layout()
    fig.savefig(out, dpi=300)


    # plot Ls ----------------------------------------------
    fig, axes = plt.subplots(nrows=1, ncols=3, sharex=True, figsize=(9, 2.4))
    plt.subplots_adjust(wspace=0.02)
    sns.set(font_scale=font_scale)
    ylabel = "Probability density"
    xlabel = "$[L]_s$ (mM)"

    # 2c
    ax = axes[0]
    xs = [tr_val_2c["Ls"]]
    xs = [filter_outliers(x) for x in xs]
    labels = [None]
    plot_kde_hist(xs, labels, colors, ax)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title("Two-Component")

    cis = [conf_interv(x) for x in xs]
    ci_str = "[%0.2f, %0.2f]" % cis[0]
    print(exper + ", 2C, Ls: " + ci_str)

    # rm
    ax = axes[1]
    xs = [tr_val_rm["Ls"]]
    xs = [filter_outliers(x) for x in xs]
    labels = [None]
    plot_kde_hist(xs, labels, colors, ax)
    ax.set_xlabel(xlabel)
    ax.set_title("Racemic Mixture")

    cis = [conf_interv(x) for x in xs]
    ci_str = "[%0.2f, %0.2f]" % cis[0]
    print(exper + ", RM, Ls: " + ci_str)

    # em
    ax = axes[2]
    xs = [tr_val_em["Ls"], tr_val_em["Ls"] * tr_val_em["rho"], tr_val_em["Ls"] * (1 - tr_val_em["rho"])]
    xs = [filter_outliers(x) for x in xs]
    labels = ["$[L]_s$", "$[L_1]_s$", "$[L_2]_s$"]
    plot_kde_hist(xs, labels, colors, ax)
    ax.set_xlabel(xlabel)
    ax.set_title("Enantiomer")

    cis = [conf_interv(x) for x in xs]
    ci_str = "[%0.2f, %0.2f]" % cis[0]
    print(exper + ", EM, Ls: " + ci_str)

    out = exper + "_Ls.pdf"
    fig.tight_layout()
    fig.savefig(out, dpi=300)


    # plot rho and P0 ----------------------------------------------
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(6.4, 2.4))
    sns.set(font_scale=font_scale)
    ylabel = "Probability density"

    ax = axes[0]
    xlabel = "$rho$"
    xs = [tr_val_em["rho"]]
    labels = ["rho"]
    plot_kde_hist(xs, labels, colors, ax)
    ax.set_xlim([0, 1])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    # P0
    ax = axes[1]
    xlabel = "$[R]_0$ (mM)"

    xs = [tr_val_2c["P0"], tr_val_rm["P0"], tr_val_em["P0"]]
    xs = [filter_outliers(x) for x in xs]
    labels = ["2C", "RM", "EM"]
    plot_kde_hist(xs, labels, colors, ax)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    out = exper + "_rho_P0.pdf"
    fig.tight_layout()
    fig.savefig(out, dpi=300)


