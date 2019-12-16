"""
calculate and plot bayes factors from traces files
the likelihood for each mcmc sample is calculated using manually implemented gaussian PDF
"""
from __future__ import print_function

import os
import argparse
import pickle
import glob

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from _data_io import ITCExperiment, load_heat_micro_cal
from _models import extract_loglhs_from_traces_manual
from _bayes_factor import log_marginal_lhs_bootstrap

parser = argparse.ArgumentParser()
parser.add_argument("--two_component_mcmc_dir", type=str, default="/home/tnguye46/bayesian_itc_racemic/07.twocomponent_mcmc/pymc2_2")
parser.add_argument("--racemic_mixture_mcmc_dir", type=str, default="/home/tnguye46/bayesian_itc_racemic/08.racemicmixture_mcmc/pymc2_2")
parser.add_argument("--enantiomer_mcmc_dir", type=str, default="/home/tnguye46/bayesian_itc_racemic/09.enantiomer_mcmc/pymc2_2")

parser.add_argument("--repeat_prefix", type=str, default="repeat_")

parser.add_argument("--exper_info_dir", type=str, default="/home/tnguye46/bayesian_itc_racemic/05.exper_info")
parser.add_argument("--heat_dir", type=str, default="/home/tnguye46/bayesian_itc_racemic/04.heat_in_origin_format")

parser.add_argument("--exper_info_file", type=str, default="experimental_information.pickle")
parser.add_argument("--mcmc_trace_file", type=str, default="traces.pickle")

parser.add_argument("--experiments", type=str,
                    default="Fokkens_1_a Fokkens_1_b Fokkens_1_c Fokkens_1_d Fokkens_1_e Baum_57 Baum_59 Baum_60_1 Baum_60_2 Baum_60_3 Baum_60_4")

parser.add_argument("--bootstrap_repeats", type=int, default=1000)

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

marg_lh_2cbm = {}
marg_lh_rmbm = {}
marg_lh_embm = {}

for exper in experiments:
    print(exper)
    exper_info_file = os.path.join(args.exper_info_dir, exper, args.exper_info_file)
    heat_file = os.path.join(args.heat_dir, exper + ".DAT")

    marg_lh_2cbm[exper] = {}
    marg_lh_rmbm[exper] = {}
    marg_lh_embm[exper] = {}

    # 2cbm
    traces_files_2cbm = [os.path.join(d, exper, args.mcmc_trace_file) for d in two_component_dirs]
    print("loading:\n", traces_files_2cbm)
    traces_2cbm = _load_and_combine_traces(traces_files_2cbm)
    print("Length of trace", len(traces_2cbm[traces_2cbm.keys()[0]]))
    loglhs_2cbm = extract_loglhs_from_traces_manual(traces_2cbm, "2cbm", exper_info_file, heat_file)
    all_sample_estimate, bootstrap_samples = log_marginal_lhs_bootstrap(loglhs_2cbm, sample_size=None,
                                                                        bootstrap_repeats=args.bootstrap_repeats)
    marg_lh_2cbm[exper]["all_sample_estimate"] = all_sample_estimate
    marg_lh_2cbm[exper]["bootstrap_samples"] = bootstrap_samples

    # rmbm
    traces_files_rmbm = [os.path.join(d, exper, args.mcmc_trace_file) for d in racemic_mixture_dirs]
    print("loading:\n", traces_files_rmbm)
    traces_rmbm = _load_and_combine_traces(traces_files_rmbm)
    print("Length of trace", len(traces_rmbm[traces_rmbm.keys()[0]]))
    loglhs_rmbm = extract_loglhs_from_traces_manual(traces_rmbm, "rmbm", exper_info_file, heat_file)
    all_sample_estimate, bootstrap_samples = log_marginal_lhs_bootstrap(loglhs_rmbm, sample_size=None,
                                                                        bootstrap_repeats=args.bootstrap_repeats)
    marg_lh_rmbm[exper]["all_sample_estimate"] = all_sample_estimate
    marg_lh_rmbm[exper]["bootstrap_samples"] = bootstrap_samples

    # embm
    traces_files_embm = [os.path.join(d, exper, args.mcmc_trace_file) for d in enantiomer_dirs]
    print("loading:\n", traces_files_embm)
    traces_embm = _load_and_combine_traces(traces_files_embm)
    print("Length of trace", len(traces_embm[traces_embm.keys()[0]]))
    loglhs_embm = extract_loglhs_from_traces_manual(traces_embm, "embm", exper_info_file, heat_file)
    all_sample_estimate, bootstrap_samples = log_marginal_lhs_bootstrap(loglhs_embm, sample_size=None,
                                                                        bootstrap_repeats=args.bootstrap_repeats)
    marg_lh_embm[exper]["all_sample_estimate"] = all_sample_estimate
    marg_lh_embm[exper]["bootstrap_samples"] = bootstrap_samples


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

"""
bf_rmbm_vs_2cbm = pd.Series(bf_rmbm_vs_2cbm)
bf_rmbm_vs_2cbm.sort_values(ascending=True, inplace=True)
bf_rmbm_vs_2cbm_log = np.log10(bf_rmbm_vs_2cbm)

bf_embm_vs_2cbm = pd.Series(bf_embm_vs_2cbm)
bf_embm_vs_2cbm.sort_values(ascending=True, inplace=True)
bf_embm_vs_2cbm_log = np.log10(bf_embm_vs_2cbm)

bf_embm_vs_rmbm = pd.Series(bf_embm_vs_rmbm)
bf_embm_vs_rmbm.sort_values(ascending=True, inplace=True)
bf_embm_vs_rmbm_log = np.log10(bf_embm_vs_rmbm)

# plot
sns.set(font_scale=args.font_scale)
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(3.2, 2.4))
bf_rmbm_vs_2cbm_log.plot(kind="barh", ax=ax)
ax.set_xlabel("$log \\frac{P(D|rmbm)}{P(D|2cbm)}$")
fig.tight_layout()
fig.savefig("bf_rmbm_vs_2cbm.pdf", dpi=300)


fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(3.2, 2.4))
bf_embm_vs_2cbm_log.plot(kind="barh", ax=ax)
ax.set_xlabel("$log \\frac{P(D|embm)}{P(D|2cbm)}$")
fig.tight_layout()
fig.savefig("bf_embm_vs_2cbm.pdf", dpi=300)


fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(3.2, 2.4))
bf_embm_vs_rmbm_log.plot(kind="barh", ax=ax)
ax.set_xlabel("$log \\frac{P(D|embm)}{P(D|rmbm)}$")
fig.tight_layout()
fig.savefig("bf_embm_vs_rmbm.pdf", dpi=300)

print("DONE!")
"""