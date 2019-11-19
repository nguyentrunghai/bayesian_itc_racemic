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

parser.add_argument("--two_component_mcmc_dir", type=str, default="twocomponent_mcmc")
parser.add_argument("--racemic_mixture_mcmc_dir", type=str, default="racemicmixture_mcmc")
parser.add_argument("--enantiomer_mcmc_dir", type=str, default="enantiomer_mcmc")

parser.add_argument("--bayes_factor_file", type=str, default="marginal_likelihood.dat")

parser.add_argument("--repeat_prefix", type=str, default="repeat_")
parser.add_argument("--experiments", type=str, default="Fokkens_1_c Fokkens_1_d")

args = parser.parse_args()

experiments = args.experiments.split()
two_component_dirs = glob.glob(os.path.join(args.two_component_mcmc_dir, args.repeat_prefix + "*"))
print("two_component_dirs:", two_component_dirs)

racemic_mixture_dirs = glob.glob(os.path.join(args.racemic_mixture_mcmc_dir, args.repeat_prefix + "*"))
print("racemic_mixture_dir:", racemic_mixture_dirs)

enantiomer_dirs = glob.glob(os.path.join(args.enantiomer_mcmc_dir, args.repeat_prefix + "*"))
print("enantiomer_dir:", enantiomer_dirs)

two_component_ml = defaultdict(list)
racemic_mixture_ml = defaultdict(list)
enantiomer_ml = defaultdict(list)

for experiment in experiments:
    for repeat_dir in two_component_dirs:
        bf_file = os.path.join(repeat_dir, experiment, args.bayes_factor_file)
        two_component_ml[experiment].append(np.loadtxt(bf_file))

    for repeat_dir in racemic_mixture_dirs:
        bf_file = os.path.join(repeat_dir, experiment, args.bayes_factor_file)
        racemic_mixture_ml[experiment].append(np.loadtxt(bf_file))

    for repeat_dir in enantiomer_dirs:
        bf_file = os.path.join(repeat_dir, experiment, args.bayes_factor_file)
        enantiomer_ml[experiment].append(np.loadtxt(bf_file))

"""
bf_rmbm_vs_2cbm = defaultdict(list)
bf_embm_vs_2cbm = defaultdict(list)
bf_embm_vs_rmbm = defaultdict(list)

for experiment in experiments:
    for bf_rmbm in racemic_mixture_ml[experiment]:
        for bf_2cbm in two_component_ml[experiment]:
            bf_rmbm_vs_2cbm[experiment].append(bf_rmbm / bf_2cbm)

    for bf_embm in enantiomer_ml[experiment]:
        for bf_2cbm in two_component_ml[experiment]:
            bf_embm_vs_2cbm[experiment].append(bf_embm / bf_2cbm)

    for bf_embm in enantiomer_ml[experiment]:
        for bf_rmbm in racemic_mixture_ml[experiment]:
            bf_embm_vs_rmbm[experiment].append(bf_embm / bf_rmbm)


bf_rmbm_vs_2cbm_mean = pd.Series({experiment: np.mean(bf_rmbm_vs_2cbm[experiment]) for experiment in experiments})
bf_rmbm_vs_2cbm_std = pd.Series({experiment: np.std(bf_rmbm_vs_2cbm[experiment]) for experiment in experiments})
bf_rmbm_vs_2cbm_df = pd.DataFrame({"mean": bf_rmbm_vs_2cbm_mean, "std": bf_rmbm_vs_2cbm_std})

bf_embm_vs_2cbm_mean = pd.Series({experiment: np.mean(bf_embm_vs_2cbm[experiment]) for experiment in experiments})
bf_embm_vs_2cbm_std = pd.Series({experiment: np.std(bf_embm_vs_2cbm[experiment]) for experiment in experiments})
bf_embm_vs_2cbm_df = bf_embm_vs_2cbm_df = pd.DataFrame({"mean": bf_embm_vs_2cbm_mean, "std": bf_embm_vs_2cbm_std})

bf_embm_vs_rmbm_mean = pd.Series({experiment: np.mean(bf_embm_vs_rmbm[experiment]) for experiment in experiments})
bf_embm_vs_rmbm_std = pd.Series({experiment: np.std(bf_embm_vs_rmbm[experiment]) for experiment in experiments})
bf_embm_vs_rmbm_df = bf_embm_vs_rmbm_df = pd.DataFrame({"mean": bf_embm_vs_rmbm_mean, "std": bf_embm_vs_rmbm_std})
"""

two_component_ml_mean = {}
racemic_mixture_ml_mean = {}
enantiomer_ml_mean = {}

for experiment in experiments:
    two_component_ml_mean[experiment] = np.mean(two_component_ml[experiment])
    racemic_mixture_ml_mean[experiment] = np.mean(racemic_mixture_ml[experiment])
    enantiomer_ml_mean[experiment] = np.mean(enantiomer_ml[experiment])

bf_rmbm_vs_2cbm = {experiment: racemic_mixture_ml_mean[experiment] / two_component_ml_mean[experiment]}
bf_rmbm_vs_2cbm = pd.Series(bf_rmbm_vs_2cbm)

bf_embm_vs_2cbm = {experiment: enantiomer_ml_mean[experiment] / two_component_ml_mean[experiment]}
bf_embm_vs_2cbm = pd.Series(bf_embm_vs_2cbm)

bf_embm_vs_rmbm = {experiment: enantiomer_ml_mean[experiment] / racemic_mixture_ml_mean[experiment]}
bf_embm_vs_rmbm = pd.Series(bf_embm_vs_rmbm)

