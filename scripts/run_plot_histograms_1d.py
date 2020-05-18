"""
to compare histogram of DG, DH, P0 and Ls between twocomponent and racemicmixture models
"""

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
                    default="/home/tnguye46/bayesian_itc_racemic/07.twocomponent_mcmc/pymc3_met_2")
parser.add_argument("--racemic_mixture_mcmc_dir", type=str,
                    default="/home/tnguye46/bayesian_itc_racemic/08.racemicmixture_mcmc/pymc3_met_2")
parser.add_argument("--enantiomer_mcmc_dir", type=str,
                    default="/home/tnguye46/bayesian_itc_racemic/09.enantiomer_mcmc/pymc3_met_2")

parser.add_argument("--repeat_prefix", type=str, default="repeat_")
parser.add_argument("--exclude_repeats", type=str, default="")

parser.add_argument("--trace_pickle", type=str, default="trace_obj.pickle")
parser.add_argument("--model_pickle", type=str, default="pm_model.pickle")

parser.add_argument("--experiments", type=str,
default="Fokkens_1_a Fokkens_1_b Fokkens_1_c Fokkens_1_d Fokkens_1_e Baum_57 Baum_59 Baum_60_1 Baum_60_2 Baum_60_3 Baum_60_4")

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

    map_2c = find_MAP_traces(model_2c, traces_2c)
    print("map_2c", map_2c)

    map_rm = find_MAP_traces(model_rm, traces_rm)
    print("map_rm", map_rm)

    map_em = find_MAP_traces(model_em, traces_em)
    print("map_em", map_em)

    # plot DeltaG
    fig, axes = plt.subplots(nrows=1, ncols=3, sharex=True, sharey=True, figsize=(9, 2.4))
    plt.subplots_adjust(wspace=0.02)
    sns.set(font_scale=font_scale)


for experiment in experiments:
    print("Processing " + experiment)
    twocomponent_traces_file = os.path.join(args.twocomponent_mcmc_dir, experiment, args.mcmc_trace_file)
    print("Loading " + twocomponent_traces_file)
    twocomponent_traces = pickle.load(open(twocomponent_traces_file))

    alter_model_traces_file = os.path.join(args.alternative_model_mcmc_dir, experiment, args.mcmc_trace_file)
    print("Loading " + alter_model_traces_file)
    alter_model_trace = pickle.load(open(alter_model_traces_file))

    # DeltaG
    print("Ploting DeltaG")
    DeltaG1 = alter_model_trace["DeltaG1"]
    DeltaG2 = alter_model_trace["DeltaG1"] + alter_model_trace["DeltaDeltaG"]
    data_list = (twocomponent_traces["DeltaG"], DeltaG1, DeltaG2)
    labels = ("$\Delta G$ (TwoComponent)", "$\Delta G_1$", "$\Delta G_2$")
    xlabel = "$\Delta G$ (kcal/mol)"
    out = experiment + "_" + args.alternative_model + "_deltaG" + ".pdf"
    _plot_kde_hist(data_list, labels, colors, xlabel, ylabel, out)

    # DeltaH
    print("Ploting DeltaH")
    DeltaH1 = alter_model_trace["DeltaH1"]
    DeltaH2 = alter_model_trace["DeltaH2"]
    data_list = (twocomponent_traces["DeltaH"], DeltaH1, DeltaH2)
    labels = ("$\Delta H$ (TwoComponent)", "$\Delta H_1$", "$\Delta H_2$")
    xlabel = "$\Delta H$ (kcal/mol)"
    out = experiment + "_" + args.alternative_model + "_deltaH" + ".pdf"
    _plot_kde_hist(data_list, labels, colors, xlabel, ylabel, out)

    # Ls
    print("Ploting Ls")
    if args.alternative_model == "racemicmixture":
        rho = 0.5
    elif args.alternative_model == "enantiomer":
        rho = alter_model_trace["rho"]

    Ls1 = alter_model_trace["Ls"] * rho
    Ls2 = alter_model_trace["Ls"] * (1 - rho)
    data_list = (twocomponent_traces["Ls"], Ls1, Ls2)
    labels = ("$[L]_s$ (TwoComponent)", "$[L_1]_s$", "$[L_2]_s$")
    xlabel = "$[L]_s$ (mM)"
    out = experiment + "_" + args.alternative_model + "_Ls" + ".pdf"
    _plot_kde_hist(data_list, labels, colors, xlabel, ylabel, out)

    # P0
    print("Ploting P0")
    data_list = (twocomponent_traces["P0"], alter_model_trace["P0"])

    if args.alternative_model == "racemicmixture":
        labels = ("$[R]_0$ (TwoComponent)", "$[R]_0$ (RacemicMixture)")
    elif args.alternative_model == "enantiomer":
        labels = ("$[R]_0$ (TwoComponent)", "$[R]_0$ (Enantiomer)")

    xlabel = "$[R]_0$ (mM)"
    out = experiment + "_" + args.alternative_model + "_P0" + ".pdf"
    _plot_kde_hist(data_list, labels, colors[:-1], xlabel, ylabel, out)
