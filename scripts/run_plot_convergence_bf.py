"""
To plot convergence lines for Bayes factors
"""

import argparse
import os

import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

parser = argparse.ArgumentParser()

parser.add_argument("--data_dir", type=str,
                default="/home/tnguye46/bayesian_itc_racemic/11.analyses/bayes_factor_convergence/pymc3_nuts_2/mod")

parser.add_argument("--experiments", type=str,
default="Fokkens_1_a Fokkens_1_b Fokkens_1_c Fokkens_1_d Fokkens_1_e Baum_57 Baum_59 Baum_60_1 Baum_60_2 Baum_60_3 Baum_60_4")

parser.add_argument("--xlabel", type=str, default="Sample proportion")
parser.add_argument("--font_scale", type=float, default=0.75)

args = parser.parse_args()

sns.set(font_scale=args.font_scale)

experiments = args.experiments.split()
print("experiments:", experiments)

bf_names = ["RM_over_2C", "EM_over_2C"]
ylabels = [r"$log_{10} [\frac{P(D|RM)}{P(D|2C)}]$", r"$log_{10} [\frac{P(D|EM)}{P(D|2C)}]$"]

xlabel = args.xlabel

for exper in experiments:
    print("\n\nPloting " + exper)

    fig, axes = plt.subplots(ncols=2, nrows=1, sharex=True, figsize=(6.4, 2.4))
    plt.subplots_adjust(wspace=0., hspace=0.)

    for bf_name, ylabel, ax in zip(bf_names, ylabels, axes):
        infile = os.path.join(args.data_dir, exper + "_" + bf_name + ".dat")
        print(infile)

        data = np.loadtxt(infile)
        x = data[:, 0]
        y = data[:, -2]
        yerr = data[:, -1]

        ax.errorbar(x, y, yerr=yerr, linestyle="solid", c="k", marker="o", markersize=5)
        ax.set_ylabel(ylabel)
        ax.set_xlabel(xlabel)

    out = exper + ".pdf"
    fig.tight_layout()
    fig.savefig(out, dpi=300)
