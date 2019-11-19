"""
To collect and plot Bayes factors from sequential MC results
"""

import argparse
import glob
import os

parser = argparse.ArgumentParser()

parser.add_argument("--two_component_mcmc_dir", type=str, default="twocomponent_mcmc")
parser.add_argument("--racemic_mixture_mcmc_dir", type=str, default="racemicmixture_mcmc")
parser.add_argument("--enantiomer_mcmc_dir", type=str, default="enantiomer_mcmc")

parser.add_argument("--repeat_prefix", type=str, default="repeat_")
parser.add_argument("--experiments", type=str, default="Fokkens_1_c Fokkens_1_d")

args = parser.parse_args()

experiments = args.experiments.split()
two_component_dirs = glob.glob(os.path.join(args.two_component_mcmc_dir, args.repeat_prefix + "*"))
print("two_component_dirs:", two_component_dirs)

racemic_mixture_dir = glob.glob(os.path.join(args.racemic_mixture_mcmc_dir, args.repeat_prefix + "*"))
print("racemic_mixture_dir:", racemic_mixture_dir)

enantiomer_dir = glob.glob(os.path.join(args.enantiomer_mcmc_dir, args.repeat_prefix + "*"))
print("enantiomer_dir:", enantiomer_dir)

