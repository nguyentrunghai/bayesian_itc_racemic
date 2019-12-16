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
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--two_component_mcmc_dir", type=str, default="/home/tnguye46/bayesian_itc_racemic/07.twocomponent_mcmc/pymc2_2")
parser.add_argument("--racemic_mixture_mcmc_dir", type=str, default="/home/tnguye46/bayesian_itc_racemic/08.racemicmixture_mcmc/pymc2_2")
parser.add_argument("--enantiomer_mcmc_dir", type=str, default="/home/tnguye46/bayesian_itc_racemic/09.enantiomer_mcmc/pymc2_2")

parser.add_argument("--repeat_prefix", type=str, default="repeat_")

parser.add_argument("--extracted_loglhs_file", type=str, default="log_priors_llhs.csv")
parser.add_argument("--mcmc_trace_file", type=str, default="traces.pickle")

parser.add_argument("--experiments", type=str,
                    default="Fokkens_1_a Fokkens_1_b Fokkens_1_c Fokkens_1_d Fokkens_1_e Baum_57 Baum_59 Baum_60_1 Baum_60_2 Baum_60_3 Baum_60_4")

parser.add_argument("--font_scale", type=float, default=0.75)

args = parser.parse_args()


def _load_combine_dfs(csv_files):
    df_list = [pd.read_csv(f) for f in csv_files]
    comb_df = pd.concat(df_list, axis=0, ignore_index=True)
    return comb_df


def _load_and_combine_traces(trace_files):
    list_traces = [pickle.load(open(trace_file)) for trace_file in trace_files]
    keys = list_traces[0].keys()
    result = {}
    for key in keys:
        result[key] = np.concatenate([trace[key] for trace in list_traces])
    return result


two_component_dirs = glob.glob(os.path.join(args.two_component_mcmc_dir, args.repeat_prefix + "*"))
two_component_dirs = two_component_dirs[:3]
print("two_component_dirs:", two_component_dirs)

racemic_mixture_dirs = glob.glob(os.path.join(args.racemic_mixture_mcmc_dir, args.repeat_prefix + "*"))
racemic_mixture_dirs = racemic_mixture_dirs[:3]
print("racemic_mixture_dir:", racemic_mixture_dirs)

enantiomer_dirs = glob.glob(os.path.join(args.enantiomer_mcmc_dir, args.repeat_prefix + "*"))
enantiomer_dirs = enantiomer_dirs[:3]
print("enantiomer_dir:", enantiomer_dirs)

experiments = args.experiments.split()
print("experiments", experiments)

pr_lh_2cbm = {}
pr_lh_rmbm = {}
pr_lh_embm = {}

for exper in experiments:
    print(exper)

    loglhs_files_2cbm = [os.path.join(d, exper, args.extracted_loglhs_file) for d in two_component_dirs]
    print("loglhs_files_2cbm:\n", loglhs_files_2cbm)
    trace_files_2cbm = [os.path.join(d, exper, args.loglhs_files_2cbm) for d in two_component_dirs]
    print("loglhs_files_2cbm:\n", loglhs_files_2cbm)

    loglhs_2cbm = _load_combine_dfs(loglhs_files_2cbm)
    traces_2cbm = _load_and_combine_traces(trace_files_2cbm)
    log_posterior = loglhs_2cbm["log_lhs"] + loglhs_2cbm["log_priors"]
    log_posterior = log_posterior.to_numpy()
    max_idx_2cbm = np.argmax(log_posterior)
    map_2cbm = {param: traces_2cbm[param][max_idx_2cbm] for param in traces_2cbm}
    print("map_2cbm:", map_2cbm)

