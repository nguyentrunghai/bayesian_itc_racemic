"""
calculate and plot Bayes factors from extracted log likelihoods
"""
from __future__ import print_function

import argparse
import glob
import os

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

from _bayes_factor import log_marginal_lhs_bootstrap

parser = argparse.ArgumentParser()
parser.add_argument("--two_component_mcmc_dir", type=str, default="/home/tnguye46/bayesian_itc_racemic/07.twocomponent_mcmc/pymc2_2")
parser.add_argument("--racemic_mixture_mcmc_dir", type=str, default="/home/tnguye46/bayesian_itc_racemic/08.racemicmixture_mcmc/pymc2_2")
parser.add_argument("--enantiomer_mcmc_dir", type=str, default="/home/tnguye46/bayesian_itc_racemic/09.enantiomer_mcmc/pymc2_2")

parser.add_argument("--repeat_prefix", type=str, default="repeat_")

parser.add_argument("--extracted_loglhs_file", type=str, default="log_priors_llhs.csv")

parser.add_argument("--experiments", type=str,
                    default="Fokkens_1_a Fokkens_1_b Fokkens_1_c Fokkens_1_d Fokkens_1_e Baum_59 Baum_60_1 Baum_60_4")

parser.add_argument("--bootstrap_repeats", type=int, default=1000)

parser.add_argument("--font_scale", type=float, default=0.75)

args = parser.parse_args()


def _load_combine_dfs(csv_files):
    df_list = [pd.read_csv(f) for f in csv_files]
    comb_df = pd.concat(df_list, axis=0, ignore_index=True)
    return comb_df


two_component_dirs = glob.glob(os.path.join(args.two_component_mcmc_dir, args.repeat_prefix + "*"))
#two_component_dirs = two_component_dirs[:3]
print("two_component_dirs:", two_component_dirs)

racemic_mixture_dirs = glob.glob(os.path.join(args.racemic_mixture_mcmc_dir, args.repeat_prefix + "*"))
#racemic_mixture_dirs = racemic_mixture_dirs[:3]
print("racemic_mixture_dir:", racemic_mixture_dirs)

enantiomer_dirs = glob.glob(os.path.join(args.enantiomer_mcmc_dir, args.repeat_prefix + "*"))
#enantiomer_dirs = enantiomer_dirs[:3]
print("enantiomer_dir:", enantiomer_dirs)

experiments = args.experiments.split()
print("experiments", experiments)

marg_lh_2cbm = {}
marg_lh_rmbm = {}
marg_lh_embm = {}


for exper in experiments:
    print(exper)

    marg_lh_2cbm[exper] = {}
    marg_lh_rmbm[exper] = {}
    marg_lh_embm[exper] = {}

    # 2cbm
    loglhs_files_2cbm = [os.path.join(d, exper, args.extracted_loglhs_file) for d in two_component_dirs]
    print("loading:\n", loglhs_files_2cbm)
    loglhs_2cbm = _load_combine_dfs(loglhs_files_2cbm)["log_lhs"]
    print("Length loglhs_2cbm", len(loglhs_2cbm))
    all_samp_est_2cbm, bootstr_samp_2cbm = log_marginal_lhs_bootstrap(loglhs_2cbm, sample_size=None,
                                                                      bootstrap_repeats=args.bootstrap_repeats)
    marg_lh_2cbm[exper]["all_sample_estimate"] = all_samp_est_2cbm
    marg_lh_2cbm[exper]["bootstrap_samples"] = bootstr_samp_2cbm

    # rmbm
    loglhs_files_rmbm = [os.path.join(d, exper, args.extracted_loglhs_file) for d in racemic_mixture_dirs]
    print("loading:\n", loglhs_files_rmbm)
    loglhs_rmbm = _load_combine_dfs(loglhs_files_rmbm)["log_lhs"]
    print("Length loglhs_rmbm", len(loglhs_rmbm))
    all_samp_est_rmbm, bootstr_samp_rmbm = log_marginal_lhs_bootstrap(loglhs_rmbm, sample_size=None,
                                                                      bootstrap_repeats=args.bootstrap_repeats)
    marg_lh_rmbm[exper]["all_sample_estimate"] = all_samp_est_rmbm
    marg_lh_rmbm[exper]["bootstrap_samples"] = bootstr_samp_rmbm

    # embm
    loglhs_files_embm = [os.path.join(d, exper, args.extracted_loglhs_file) for d in enantiomer_dirs]
    print("loading:\n", loglhs_files_embm)
    loglhs_embm = _load_combine_dfs(loglhs_files_embm)["log_lhs"]
    print("Length loglhs_embm", len(loglhs_embm))
    all_samp_est_embm, bootstr_samp_embm = log_marginal_lhs_bootstrap(loglhs_embm, sample_size=None,
                                                                      bootstrap_repeats=args.bootstrap_repeats)
    marg_lh_embm[exper]["all_sample_estimate"] = all_samp_est_embm
    marg_lh_embm[exper]["bootstrap_samples"] = bootstr_samp_embm


bf_rmbm_vs_2cbm = {}
bf_embm_vs_2cbm = {}
bf_embm_vs_rmbm = {}

for exper in experiments:
    bf_rmbm_vs_2cbm[exper] = {}
    bf_embm_vs_2cbm[exper] = {}
    bf_embm_vs_rmbm[exper] = {}

    bf_rmbm_vs_2cbm[exper]["bf"] = np.exp(
        marg_lh_rmbm[exper]["all_sample_estimate"] - marg_lh_2cbm[exper]["all_sample_estimate"])
    bf_rmbm_vs_2cbm[exper]["err"] = np.std(np.exp(
        marg_lh_rmbm[exper]["bootstrap_samples"] - marg_lh_2cbm[exper]["bootstrap_samples"]))

    bf_embm_vs_2cbm[exper]["bf"] = np.exp(
        marg_lh_embm[exper]["all_sample_estimate"] - marg_lh_2cbm[exper]["all_sample_estimate"])
    bf_embm_vs_2cbm[exper]["err"] = np.std(np.exp(
        marg_lh_embm[exper]["bootstrap_samples"] - marg_lh_2cbm[exper]["bootstrap_samples"]))

    bf_embm_vs_rmbm[exper]["bf"] = np.exp(
        marg_lh_embm[exper]["all_sample_estimate"] - marg_lh_rmbm[exper]["all_sample_estimate"])
    bf_embm_vs_rmbm[exper]["err"] = np.std(np.exp(
        marg_lh_embm[exper]["bootstrap_samples"] - marg_lh_rmbm[exper]["bootstrap_samples"]))


bf_rmbm_vs_2cbm = pd.DataFrame.from_dict(bf_rmbm_vs_2cbm, orient="index")
bf_embm_vs_2cbm = pd.DataFrame.from_dict(bf_embm_vs_2cbm, orient="index")
bf_embm_vs_rmbm = pd.DataFrame.from_dict(bf_embm_vs_rmbm, orient="index")

bf_rmbm_vs_2cbm["bf_log"] = np.log10(bf_rmbm_vs_2cbm["bf"])
bf_rmbm_vs_2cbm["err_log"] = np.log10(bf_rmbm_vs_2cbm["err"])
bf_rmbm_vs_2cbm = bf_rmbm_vs_2cbm.sort_values(by="bf_log", ascending=True)

bf_embm_vs_2cbm["bf_log"] = np.log10(bf_embm_vs_2cbm["bf"])
bf_embm_vs_2cbm["err_log"] = np.log10(bf_embm_vs_2cbm["err"])
bf_embm_vs_2cbm = bf_embm_vs_2cbm.sort_values(by="bf_log", ascending=True)

bf_embm_vs_rmbm["bf_log"] = np.log10(bf_embm_vs_rmbm["bf"])
bf_embm_vs_rmbm["err_log"] = np.log10(bf_embm_vs_rmbm["err"])
bf_embm_vs_rmbm = bf_embm_vs_rmbm.sort_values(by="bf_log", ascending=True)


error_scale = 0.5
sns.set(font_scale=args.font_scale)

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(3.2, 2.4))
ax.barh(list(bf_rmbm_vs_2cbm.index), bf_rmbm_vs_2cbm["bf_log"].to_numpy(),
        xerr=error_scale*bf_rmbm_vs_2cbm["err_log"].to_numpy())
ax.set_xlabel("$log \\frac{P(D|rmbm)}{P(D|2cbm)}$")
fig.tight_layout()
fig.savefig("bf_rmbm_vs_2cbm.pdf", dpi=300)

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(3.2, 2.4))
ax.barh(list(bf_embm_vs_2cbm.index), bf_embm_vs_2cbm["bf_log"].to_numpy(),
        xerr=error_scale*bf_embm_vs_2cbm["err_log"].to_numpy())
ax.set_xlabel("$log \\frac{P(D|embm)}{P(D|2cbm)}$")
fig.tight_layout()
fig.savefig("bf_embm_vs_2cbm.pdf", dpi=300)

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(3.2, 2.4))
ax.barh(list(bf_embm_vs_rmbm.index), bf_embm_vs_rmbm["bf_log"].to_numpy(),
        xerr=error_scale*bf_embm_vs_rmbm["err_log"].to_numpy())
ax.set_xlabel("$log \\frac{P(D|embm)}{P(D|rmbm)}$")
fig.tight_layout()
fig.savefig("bf_embm_vs_rmbm.pdf", dpi=300)




