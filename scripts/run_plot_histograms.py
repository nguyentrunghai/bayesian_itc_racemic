"""
to compare histogram of DG, DH, P0 and Ls between twocomponent and racemicmixture models
"""

import argparse
import glob
import os
import pickle

import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--twocomponent_mcmc_dir", type=str, default="twocomponent")
parser.add_argument("--racemicmixture_mcmc_dir", type=str, default="racemicmixture")

parser.add_argument("--exclude_experiments", type=str, default="")
args = parser.parse_args()

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

TRACES_FILE_NAME = "traces.pickle"

twocomponent_traces_files = glob.glob(os.path.join(args.twocomponent_mcmc_dir, "*", TRACES_FILE_NAME))
racemicmixture_traces_files = glob.glob(os.path.join(args.racemicmixture_mcmc_dir, "*", TRACES_FILE_NAME))

twocomponent_traces_files = {os.path.basename(os.path.dirname(f)): f for f in twocomponent_traces_files}
racemicmixture_traces_files = {os.path.basename(os.path.dirname(f)): f for f in racemicmixture_traces_files}
print("twocomponent_traces_files", twocomponent_traces_files)
print("racemicmixture_traces_files", racemicmixture_traces_files)

exclude_experiments = args.exclude_experiments.split()
experiments = set(twocomponent_traces_files.keys()).intersection(racemicmixture_traces_files.keys())
experiments = [experiment for experiment in experiments if experiment not in exclude_experiments]
print("experiments", experiments)

colors = ("r", "b", "g")
ylabel = "Probability density"
for experiment in experiments:
    print("Processing " + experiment)
    twocomponent_traces = pickle.load(open(twocomponent_traces_files[experiment], "r"))
    racemicmixture_trace = pickle.load(open(racemicmixture_traces_files[experiment], "r"))

    # DeltaG
    print("Ploting DeltaG")
    DeltaG1 = racemicmixture_trace["DeltaG1"]
    DeltaG2 = racemicmixture_trace["DeltaG1"] + racemicmixture_trace["DeltaDeltaG"]
    data_list = (twocomponent_traces["DeltaG"], DeltaG1, DeltaG2)
    labels = ("$\Delta G$ (TwoComponent)", "$\Delta G_1$", "$\Delta G_2$")
    xlabel = "$\Delta G$ (kcal/mol)"
    out = experiment + "_deltaG" + ".pdf"
    _plot_kde_hist(data_list, labels, colors, xlabel, ylabel, out)

    # DeltaH
    print("Ploting DeltaH")
    DeltaH1 = racemicmixture_trace["DeltaH1"]
    DeltaH2 = racemicmixture_trace["DeltaH2"]
    data_list = (twocomponent_traces["DeltaH"], DeltaH1, DeltaH2)
    labels = ("$\Delta H$ (TwoComponent)", "$\Delta H_1$", "$\Delta H_2$")
    xlabel = "$\Delta H$ (kcal/mol)"
    out = experiment + "_deltaH" + ".pdf"
    _plot_kde_hist(data_list, labels, colors, xlabel, ylabel, out)

    # Ls
    print("Ploting Ls")
    Ls1 = racemicmixture_trace["Ls"] * racemicmixture_trace["rho"]
    Ls2 = racemicmixture_trace["Ls"] * (1 - racemicmixture_trace["rho"])
    data_list = (twocomponent_traces["Ls"], Ls1, Ls2)
    labels = ("$[L]_s$ (TwoComponent)", "$[L_1]_s$", "$[L_2]_s$")
    xlabel = "$[L]_s$ (mM)"
    out = experiment + "_Ls" + ".pdf"
    _plot_kde_hist(data_list, labels, colors, xlabel, ylabel, out)

    # P0
    print("Ploting P0")
    data_list = (twocomponent_traces["P0"], racemicmixture_trace["P0"])
    labels = ("$[R]_0$ (TwoComponent)", "$[R]_0$ (RacemicMixture)")
    xlabel = "$[R]_0$ (mM)"
    out = experiment + "_P0" + ".pdf"
    _plot_kde_hist(data_list, labels, colors[:-1], xlabel, ylabel, out)
