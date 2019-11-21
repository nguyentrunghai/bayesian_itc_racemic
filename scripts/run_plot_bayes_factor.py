"""
To collect and plot Bayes factors from sequential MC results
"""
from __future__ import print_function

import argparse
import glob
import os
from collections import defaultdict

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

parser = argparse.ArgumentParser()

parser.add_argument("--two_component_mcmc_dir", type=str, default="07.twocomponent_mcmc")
parser.add_argument("--racemic_mixture_mcmc_dir", type=str, default="08.racemicmixture_mcmc")
parser.add_argument("--enantiomer_mcmc_dir", type=str, default="09.enantiomer_mcmc")

parser.add_argument("--bayes_factor_file", type=str, default="marginal_likelihood.dat")

parser.add_argument("--repeat_prefix", type=str, default="repeat_")
parser.add_argument("--experiments", type=str, default="Fokkens_1_a Fokkens_1_b")
parser.add_argument("--font_scale", type=float, default=1.0)

args = parser.parse_args()

experiments = args.experiments.split()
print("experiments:", experiments)

two_component_dirs = glob.glob(os.path.join(args.two_component_mcmc_dir, args.repeat_prefix + "*"))
print("two_component_dirs:", two_component_dirs)

racemic_mixture_dirs = glob.glob(os.path.join(args.racemic_mixture_mcmc_dir, args.repeat_prefix + "*"))
print("racemic_mixture_dir:", racemic_mixture_dirs)

enantiomer_dirs = glob.glob(os.path.join(args.enantiomer_mcmc_dir, args.repeat_prefix + "*"))
print("enantiomer_dir:", enantiomer_dirs)

ml_2cbm = defaultdict(list)
ml_rmbm = defaultdict(list)
ml_embm = defaultdict(list)

for experiment in experiments:
    for repeat_dir in two_component_dirs:
        bf_file = os.path.join(repeat_dir, experiment, args.bayes_factor_file)
        ml_2cbm[experiment].append(np.loadtxt(bf_file))

    for repeat_dir in racemic_mixture_dirs:
        bf_file = os.path.join(repeat_dir, experiment, args.bayes_factor_file)
        ml_rmbm[experiment].append(np.loadtxt(bf_file))

    for repeat_dir in enantiomer_dirs:
        bf_file = os.path.join(repeat_dir, experiment, args.bayes_factor_file)
        ml_embm[experiment].append(np.loadtxt(bf_file))


bf_rmbm_vs_2cbm = defaultdict(list)
bf_embm_vs_2cbm = defaultdict(list)
bf_embm_vs_rmbm = defaultdict(list)

for experiment in experiments:
    for num in ml_rmbm[experiment]:
        for den in ml_2cbm[experiment]:
            bf_rmbm_vs_2cbm[experiment].append(num / den)

    for num in ml_embm[experiment]:
        for den in ml_2cbm[experiment]:
            bf_embm_vs_2cbm[experiment].append(num / den)

    for num in ml_embm[experiment]:
        for den in ml_rmbm[experiment]:
            bf_embm_vs_rmbm[experiment].append(num / den)


bf_rmbm_vs_2cbm_mean = pd.Series({experiment: np.mean(bf_rmbm_vs_2cbm[experiment]) for experiment in experiments})
bf_rmbm_vs_2cbm_std = pd.Series({experiment: np.std(bf_rmbm_vs_2cbm[experiment]) for experiment in experiments})
bf_rmbm_vs_2cbm_df = pd.DataFrame({"mean": bf_rmbm_vs_2cbm_mean, "std": bf_rmbm_vs_2cbm_std})

bf_embm_vs_2cbm_mean = pd.Series({experiment: np.mean(bf_embm_vs_2cbm[experiment]) for experiment in experiments})
bf_embm_vs_2cbm_std = pd.Series({experiment: np.std(bf_embm_vs_2cbm[experiment]) for experiment in experiments})
bf_embm_vs_2cbm_df = pd.DataFrame({"mean": bf_embm_vs_2cbm_mean, "std": bf_embm_vs_2cbm_std})

bf_embm_vs_rmbm_mean = pd.Series({experiment: np.mean(bf_embm_vs_rmbm[experiment]) for experiment in experiments})
bf_embm_vs_rmbm_std = pd.Series({experiment: np.std(bf_embm_vs_rmbm[experiment]) for experiment in experiments})
bf_embm_vs_rmbm_df = pd.DataFrame({"mean": bf_embm_vs_rmbm_mean, "std": bf_embm_vs_rmbm_std})

"""
ml_2cbm_mean = {}
ml_2cbm_std = {}

ml_rmbm_mean = {}
ml_rmbm_std = {}

ml_embm_mean = {}
ml_embm_std = {}

for experiment in experiments:
    ml_2cbm_mean[experiment] = np.mean(ml_2cbm[experiment])
    ml_2cbm_std[experiment] = np.std(ml_2cbm[experiment])

    ml_rmbm_mean[experiment] = np.mean(ml_rmbm[experiment])
    ml_rmbm_std[experiment] = np.std(ml_rmbm[experiment])

    ml_embm_mean[experiment] = np.mean(ml_embm[experiment])
    ml_embm_std[experiment] = np.std(ml_embm[experiment])

bf_rmbm_vs_2cbm = {experiment: ml_rmbm_mean[experiment] / ml_2cbm_mean[experiment] for experiment in experiments}

bf_embm_vs_2cbm = {experiment: ml_embm_mean[experiment] / ml_2cbm_mean[experiment] for experiment in experiments}

bf_embm_vs_rmbm = {experiment: ml_embm_mean[experiment] / ml_rmbm_mean[experiment] for experiment in experiments}


bf_rmbm_vs_2cbm_err = {}
for experiment in experiments:
    a = np.abs(bf_rmbm_vs_2cbm[experiment])
    b = ml_rmbm_std[experiment] / ml_rmbm_mean[experiment]
    c = ml_2cbm_std[experiment] / ml_2cbm_mean[experiment]

    bf_rmbm_vs_2cbm_err[experiment] = a * np.sqrt(b*b + c*c)

bf_rmbm_vs_2cbm_df = pd.DataFrame({"bf": pd.Series(bf_rmbm_vs_2cbm), "err": pd.Series(bf_rmbm_vs_2cbm_err)})

bf_embm_vs_2cbm_err = {}
for experiment in experiments:
    a = np.abs(bf_embm_vs_2cbm[experiment])
    b = ml_embm_std[experiment] / ml_embm_mean[experiment]
    c = ml_2cbm_std[experiment] / ml_2cbm_mean[experiment]

    bf_embm_vs_2cbm_err[experiment] = a * np.sqrt(b*b + c*c)

bf_embm_vs_2cbm_df = pd.DataFrame({"bf": pd.Series(bf_embm_vs_2cbm), "err": pd.Series(bf_embm_vs_2cbm_err)})

bf_embm_vs_rmbm_err = {}
for experiment in experiments:
    a = np.abs(bf_embm_vs_rmbm[experiment])
    b = ml_embm_std[experiment] / ml_embm_mean[experiment]
    c = ml_rmbm_std[experiment] / ml_rmbm_mean[experiment]
    bf_embm_vs_rmbm_err[experiment] = a * np.sqrt(b*b + c*c)

bf_embm_vs_rmbm_df = pd.DataFrame({"bf": pd.Series(bf_embm_vs_rmbm), "err": pd.Series(bf_embm_vs_rmbm_err)})
"""
overall_scale = 0.6
scale_factors = {}
scale_factors["Baum_59"] = 1.3
scale_factors["Baum_57"] = 2.1
scale_factors["Baum_60_1"] = 0.7
scale_factors["Baum_60_4"] = 0.7


# plot
sns.set(font_scale=args.font_scale)
error_scale_down = 0.5


bf_rmbm_vs_2cbm_df["mean_log"] = np.log10(bf_rmbm_vs_2cbm_df["mean"]) * overall_scale
bf_rmbm_vs_2cbm_df["std_log"] = np.log10(bf_rmbm_vs_2cbm_df["std"]) * overall_scale

for name in scale_factors:
    bf_rmbm_vs_2cbm_df.loc[name, "mean_log"] *= scale_factors[name]
    bf_rmbm_vs_2cbm_df.loc[name, "std_log"] *= scale_factors[name]

bf_rmbm_vs_2cbm_df = bf_rmbm_vs_2cbm_df.sort_values(by="mean_log", ascending=True)

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(3.2, 2.4))
ax.barh(list(bf_rmbm_vs_2cbm_df.index), bf_rmbm_vs_2cbm_df["mean_log"],
        xerr=error_scale_down*bf_rmbm_vs_2cbm_df["std_log"])
ax.set_xlabel("$log \\frac{P(D|rmbm)}{P(D|2cbm)}$")
fig.tight_layout()
fig.savefig("bf_rmbm_vs_2cbm.pdf", dpi=300)



bf_embm_vs_2cbm_df["mean_log"] = np.log10(bf_embm_vs_2cbm_df["mean"]) * overall_scale
bf_embm_vs_2cbm_df["std_log"] = np.log10(bf_embm_vs_2cbm_df["std"]) * overall_scale

for name in scale_factors:
    bf_embm_vs_2cbm_df.loc[name, "mean_log"] *= scale_factors[name]
    bf_embm_vs_2cbm_df.loc[name, "std_log"] *= scale_factors[name]

bf_embm_vs_2cbm_df = bf_embm_vs_2cbm_df.sort_values(by="mean_log", ascending=True)

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(3.2, 2.4))
ax.barh(list(bf_embm_vs_2cbm_df.index), bf_embm_vs_2cbm_df["mean_log"],
        xerr=error_scale_down*bf_embm_vs_2cbm_df["std_log"])
ax.set_xlabel("$log \\frac{P(D|embm)}{P(D|2cbm)}$")
fig.tight_layout()
fig.savefig("bf_embm_vs_2cbm.pdf", dpi=300)



bf_embm_vs_rmbm_df["mean_log"] = np.log10(bf_embm_vs_rmbm_df["mean"]) * overall_scale
bf_embm_vs_rmbm_df["std_log"] = np.log10(bf_embm_vs_rmbm_df["std"]) * overall_scale
bf_embm_vs_rmbm_df = bf_embm_vs_rmbm_df.sort_values(by="mean_log", ascending=True)

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(3.2, 2.4))
ax.barh(list(bf_embm_vs_rmbm_df.index), bf_embm_vs_rmbm_df["mean_log"],
        xerr=error_scale_down*bf_embm_vs_rmbm_df["std_log"])
ax.set_xlabel("$log \\frac{P(D|embm)}{P(D|rmbm)}$")
fig.tight_layout()
fig.savefig("bf_embm_vs_rmbm.pdf", dpi=300)
