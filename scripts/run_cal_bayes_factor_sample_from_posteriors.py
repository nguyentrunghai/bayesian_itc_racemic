"""
calculate bayes factor
"""
from __future__ import print_function

import os
import argparse
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from _data_io import ITCExperiment, load_heat_micro_cal
from _bayes_factor import average_likelihood_from_posterior

parser = argparse.ArgumentParser()
parser.add_argument("--two_component_mcmc_dir", type=str, default="twocomponent_mcmc")
parser.add_argument("--racemic_mixture_mcmc_dir", type=str, default="racemicmixture_mcmc")
parser.add_argument("--enantiomer_mcmc_dir", type=str, default="enantiomer_mcmc")

parser.add_argument("--exper_info_dir", type=str, default="exper_info")
parser.add_argument("--heat_dir", type=str, default="heat_in_origin_format")

parser.add_argument("--exper_info_file", type=str, default="experimental_information.pickle")
parser.add_argument("--mcmc_trace_file", type=str, default="traces.pickle")

parser.add_argument("--concentration_range_factor", type=float, default=10.)

parser.add_argument("--experiments", type=str, default="Fokkens_1_c Fokkens_1_d")
parser.add_argument("--experiments_with_unif_concen_prior", type=str, default="Fokkens_1_a Fokkens_1_b")


parser.add_argument("--font_scale", type=float, default=0.75)

args = parser.parse_args()

KB = 0.0019872041      # in kcal/mol/K

assert os.path.exists(args.two_component_mcmc_dir), args.two_component_mcmc_dir + " does not exists."
assert os.path.exists(args.racemic_mixture_mcmc_dir), args.racemic_mixture_mcmc_dir + " does not exists."
assert os.path.exists(args.enantiomer_mcmc_dir), args.enantiomer_mcmc_dir + " does not exists."
assert os.path.exists(args.exper_info_dir), args.exper_info_dir + " does not exist."
assert os.path.exists(args.heat_dir), args.heat_dir + " does not exist."

experiments = args.experiments.split()
print("experiments", experiments)
experiments_with_unif_concen_prior = args.experiments_with_unif_concen_prior.split()
print("experiments_with_unif_concen_prior", experiments_with_unif_concen_prior)


bf_rmbm_vs_2cbm = {}
bf_embm_vs_2cbm = {}
bf_embm_vs_rmbm = {}

for experiment in experiments:
    print(experiment)

    if experiment in experiments_with_unif_concen_prior:
        print("Uniform prior for P0 and Ls")
        uniform_P0 = True
        uniform_Ls = True
    else:
        print("LogNormal prior for P0 and Ls")
        uniform_P0 = False
        uniform_Ls = False

    q_actual_micro_cal = load_heat_micro_cal(os.path.join(args.heat_dir, experiment + ".DAT"))
    q_actual_cal = q_actual_micro_cal * 10**(-6)

    exper_info = ITCExperiment(os.path.join(args.exper_info_dir, experiment, args.exper_info_file))

    trace_2cbm = pickle.load(open(os.path.join(args.two_component_mcmc_dir, experiment, args.mcmc_trace_file)))
    trace_rmbm = pickle.load(open(os.path.join(args.racemic_mixture_mcmc_dir, experiment, args.mcmc_trace_file)))
    trace_embm = pickle.load(open(os.path.join(args.enantiomer_mcmc_dir, experiment, args.mcmc_trace_file)))

    llh_mean_2cbm, llh_max_log_2cbm = average_likelihood_from_posterior("2cbm", q_actual_cal, exper_info,
                                                                        trace_2cbm,
                                                                        dcell=0.1, dsyringe=0.1,
                                                                        uniform_P0=uniform_P0,
                                                                        uniform_Ls=uniform_Ls,
                                                                        concentration_range_factor=args.concentration_range_factor,
                                                                        nsamples=None)

    llh_mean_rmbm, llh_max_log_rmbm = average_likelihood_from_posterior("rmbm", q_actual_cal, exper_info,
                                                                        trace_rmbm,
                                                                        dcell=0.1, dsyringe=0.1,
                                                                        uniform_P0=uniform_P0,
                                                                        uniform_Ls=uniform_Ls,
                                                                        concentration_range_factor=args.concentration_range_factor,
                                                                        nsamples=None)

    llh_mean_embm, llh_max_log_embm = average_likelihood_from_posterior("embm", q_actual_cal, exper_info,
                                                                        trace_embm,
                                                                        dcell=0.1, dsyringe=0.1,
                                                                        uniform_P0=uniform_P0,
                                                                        uniform_Ls=uniform_Ls,
                                                                        concentration_range_factor=args.concentration_range_factor,
                                                                        nsamples=None)

    bf_rmbm_vs_2cbm[experiment] = llh_mean_rmbm / llh_mean_2cbm * np.exp(llh_max_log_rmbm - llh_max_log_2cbm)
    bf_embm_vs_2cbm[experiment] = llh_mean_embm / llh_mean_2cbm * np.exp(llh_max_log_embm - llh_max_log_2cbm)
    bf_embm_vs_rmbm[experiment] = llh_mean_embm / llh_mean_rmbm * np.exp(llh_max_log_embm - llh_max_log_rmbm)

    print("aver_likelihood_2cbm: %0.5e" % (llh_mean_2cbm * np.exp(llh_max_log_2cbm)))
    print("aver_likelihood_rmbm: %0.5e" % (llh_mean_rmbm * np.exp(llh_max_log_rmbm)))
    print("aver_likelihood_embm: %0.5e" % (llh_mean_embm * np.exp(llh_max_log_embm)))
    print("Bayes factor rmbm vs 2cbm: %0.5e" % bf_rmbm_vs_2cbm[experiment])
    print("Bayes factor embm vs 2cbm: %0.5e" % bf_embm_vs_2cbm[experiment])
    print("Bayes factor embm vs rmbm: %0.5e" % bf_embm_vs_rmbm[experiment])
    print("")
    print("")


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
