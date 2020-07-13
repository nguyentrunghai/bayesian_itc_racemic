"""
plot heat curves with confidence interval bands
"""

import argparse
import os
import pickle

import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

parser = argparse.ArgumentParser()

parser.add_argument("--data_dir", type=str, default="./")
parser.add_argument("--experiments", type=str,
default="Fokkens_1_a Fokkens_1_b Fokkens_1_c Fokkens_1_d Fokkens_1_e Baum_57 Baum_59 Baum_60_1 Baum_60_2 Baum_60_3 Baum_60_4")

parser.add_argument("--ci_level", type=float, default=95)

parser.add_argument("--font_scale", type=float, default=0.75)
parser.add_argument("--xlabel", type=str, default="# injections")
parser.add_argument("--ylabel", type=str, default="heat ($\mu$cal)")

args = parser.parse_args()


def conf_interv(x, conf_level=90.):
    alpha = 100 - conf_level
    lowers = np.percentile(x, alpha/2, axis=0)
    uppers = np.percentile(x, 100-(alpha/2), axis=0)
    return lowers, uppers


def plot_heats(q_actual,
               q_map_2c, lower_2c, upper_2c,
               q_map_rm, lower_rm, upper_rm,
               q_map_em, lower_em, upper_em,
               xlabel="# injections", ylabel="heat ($\mu$cal)",
               ci_label="90% CI",
               out="out.pdf"):
    """ plot heats with confidence interval """
    x = np.array(range(len(q_actual))) + 1

    fig, axes = plt.subplots(nrows=1, ncols=3, sharex=True, sharey=True, figsize=(9, 2.4))
    plt.subplots_adjust(wspace=0.02)

    axes[0].scatter(x, q_actual, c="k", marker="+", s=20, label="observed")
    axes[0].plot(x, q_map_2c, c="k", label="MAP")
    axes[0].fill_between(x, lower_2c, upper_2c, facecolor='grey', alpha=0.5, label=ci_label)
    axes[0].set_xlabel(xlabel)
    axes[0].set_ylabel(ylabel)
    axes[0].legend(loc="best")
    axes[0].set_title("Two-Component")

    axes[1].scatter(x, q_actual, c="k", marker="+", s=20, label="observed")
    axes[1].plot(x, q_map_rm, c="k", label="MAP")
    axes[1].fill_between(x, lower_rm, upper_rm, facecolor='grey', alpha=0.5, label=ci_label)
    axes[1].set_xlabel(xlabel)
    axes[1].legend(loc="best")
    axes[1].set_title("Racemic Mixture")

    axes[2].scatter(x, q_actual, c="k", marker="+", s=20, label="observed")
    axes[2].plot(x, q_map_em, c="k", label="MAP")
    axes[2].fill_between(x, lower_em, upper_em, facecolor='grey', alpha=0.5, label=ci_label)
    axes[2].set_xlabel(xlabel)
    axes[2].legend(loc="best")
    axes[2].set_title("Enantiomer")

    fig.tight_layout()
    fig.savefig(out, dpi=300)

    return None


experiments = args.experiments.split()
print("experiments:", experiments)
ci_level = args.ci_level
ci_label = "%d%% CI" % (int(ci_level))

for exper in experiments:
    infile = os.path.join(args.data_dir, exper + ".pickle")
    print("Loading: " + infile)
    with open(infile, "rb") as handle:
        data = pickle.load(handle)

    q_actual = data["q_actual"]
    q_map_2c = data["q_map_2c"]
    q_map_rm = data["q_map_rm"]
    q_map_em = data["q_map_em"]

    lowers_2c, uppers_2c = conf_interv(data["qs_2c"], conf_level=ci_level)
    lowers_rm, uppers_rm = conf_interv(data["qs_rm"], conf_level=ci_level)
    lowers_em, uppers_em = conf_interv(data["qs_em"], conf_level=ci_level)

    out = exper + ".pdf"
    sns.set(font_scale=args.font_scale)
    plot_heats(q_actual,
               q_map_2c, lowers_2c, uppers_2c,
               q_map_rm, lowers_rm, uppers_rm,
               q_map_em, lowers_em, uppers_em,
               xlabel=args.xlabel, ylabel=args.ylabel,
               ci_label=ci_label,
               out=out)

print("DONE")
