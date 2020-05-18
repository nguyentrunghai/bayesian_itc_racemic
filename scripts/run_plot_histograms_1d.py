"""
to compare histogram of DG, DH, P0 and Ls between twocomponent and racemicmixture models
"""

import argparse
import os
import pickle

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


def _plot_kde_hist(data_list, labels, colors, xlabel, ylabel, out):
    sns.set(font_scale=0.7)
    figure_size = (3.2, 2.4)
    dpi = 300
    plt.figure(figsize=figure_size)

    for data, label, color in zip(data_list, labels, colors):
        sns.kdeplot(data, label=label, c=color)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(out, dpi=dpi)
    return None


assert args.alternative_model in ["racemicmixture", "enantiomer"], "unknown alternative model: " + args.alternative_model

experiments = args.experiments.split()
print("experiments", experiments)

colors = ("r", "b", "g")
ylabel = "Probability density"
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
