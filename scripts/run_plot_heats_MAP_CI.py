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

parser.add_argument("--font_scale", type=float, default=0.75)
parser.add_argument("--xlabel", type=str, default="# injections")
parser.add_argument("--ylabel", type=str, default="heat ($\mu$cal)")

args = parser.parse_args()


def plot_heats(q_actual,
               q_map_2c, lower_2c, upper_2c,
               q_map_rm, lower_rm, upper_rm,
               q_map_em, lower_em, upper_em,
               font_scale=1,
               xlabel="# injections", ylabel="heat ($\mu$cal)",
               ci_label="90% CI",
               out="out.pdf"):
    """ plot heats with confidence interval """
    x = np.array(range(len(q_actual))) + 1

    fig, axes = plt.subplots(nrows=1, ncols=3, sharex=True, sharey=True, figsize=(9, 2.4))
    plt.subplots_adjust(wspace=0.02)
    sns.set(font_scale=font_scale)

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
    axes[1].set_title("Racemicmixture")

    axes[2].scatter(x, q_actual, c="k", marker="+", s=20, label="observed")
    axes[2].plot(x, q_map_em, c="k", label="MAP")
    axes[2].fill_between(x, lower_em, upper_em, facecolor='grey', alpha=0.5, label=ci_label)
    axes[2].set_xlabel(xlabel)
    axes[2].legend(loc="best")
    axes[2].set_title("Enantiomer")

    fig.tight_layout()
    fig.savefig(out, dpi=300)

    return None
