"""
plot actual heat versus model heat
MAP estimates were chosen for model heat.
The posterior probabilities were extracted from log priors and log likelihoods
"""


from __future__ import print_function

import argparse
import glob
import os
import pickle

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--two_component_mcmc_dir", type=str, default="/home/tnguye46/bayesian_itc_racemic/07.twocomponent_mcmc/pymc2")
parser.add_argument("--racemic_mixture_mcmc_dir", type=str, default="/home/tnguye46/bayesian_itc_racemic/08.racemicmixture_mcmc/pymc2")
parser.add_argument("--enantiomer_mcmc_dir", type=str, default="/home/tnguye46/bayesian_itc_racemic/09.enantiomer_mcmc/pymc2")

parser.add_argument("--repeat_prefix", type=str, default="repeat_")

parser.add_argument("--extracted_loglhs_file", type=str, default="log_priors_llhs.csv")
parser.add_argument("--mcmc_trace_file", type=str, default="traces.pickle")

parser.add_argument("--experiments", type=str,
                    default="Fokkens_1_a Fokkens_1_b Fokkens_1_c Fokkens_1_d Fokkens_1_e Baum_57 Baum_59 Baum_60_1 Baum_60_2 Baum_60_3 Baum_60_4")

parser.add_argument("--font_scale", type=float, default=0.75)

args = parser.parse_args()


def _load_and_combine_traces(trace_files):
    list_traces = [pickle.load(open(trace_file)) for trace_file in trace_files]
    keys = list_traces[0].keys()
    result = {}
    for key in keys:
        result[key] = np.concatenate([trace[key] for trace in list_traces])
    return result


two_component_dirs = glob.glob(os.path.join(args.two_component_mcmc_dir, args.repeat_prefix + "*"))
print("two_component_dirs:", two_component_dirs)

racemic_mixture_dirs = glob.glob(os.path.join(args.racemic_mixture_mcmc_dir, args.repeat_prefix + "*"))
print("racemic_mixture_dir:", racemic_mixture_dirs)

enantiomer_dirs = glob.glob(os.path.join(args.enantiomer_mcmc_dir, args.repeat_prefix + "*"))
print("enantiomer_dir:", enantiomer_dirs)

experiments = args.experiments.split()
print("experiments", experiments)
