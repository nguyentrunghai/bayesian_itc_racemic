"""
To collect and plot Bayes factors from sequential MC results
"""

import argparse
import glob
import os
from collections import defaultdict

import numpy as np

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

two_component_bf = defaultdict(list)
racemic_mixture_bf = defaultdict(list)
enantiomer_bf = defaultdict(list)

for experiment in experiments:
    for repeat_dir in two_component_dirs:
        bf_file = os.path.join(repeat_dir, experiment, args.bayes_factor_file)
        two_component_bf[experiments].append(np.loadtxt(bf_file))

    for repeat_dir in racemic_mixture_dirs:
        bf_file = os.path.join(repeat_dir, experiment, args.bayes_factor_file)
        racemic_mixture_bf[experiments].append(np.loadtxt(bf_file))

    for repeat_dir in enantiomer_dirs:
        bf_file = os.path.join(repeat_dir, experiment, args.bayes_factor_file)
        enantiomer_bf[experiments].append(np.loadtxt(bf_file))
        