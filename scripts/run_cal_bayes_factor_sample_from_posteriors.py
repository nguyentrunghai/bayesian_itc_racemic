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
parser.add_argument("--two_component_mcmc_dir", type=str, default="5.twocomponent_mcmc")
parser.add_argument("--racemic_mixture_mcmc_dir", type=str, default="6.racemicmixture_mcmc")
parser.add_argument("--enantiomer_mcmc_dir", type=str, default="7.enantiomer")

parser.add_argument("--heat_dir", type=str, default="4.heat_in_origin_format")

parser.add_argument("--exper_info_file", type=str, default="experimental_information.pickle")
parser.add_argument("--mcmc_trace_file", type=str, default="traces.pickle")

parser.add_argument("--uniform_P0", action="store_true", default=False)
parser.add_argument("--uniform_Ls", action="store_true", default=False)
parser.add_argument("--concentration_range_factor", type=float, default=10.)

parser.add_argument("--nsamples", type=int, default=None)

parser.add_argument("--experiments", type=str, default="Fokkens_1_c Fokkens_1_d")

parser.add_argument("--font_scale", type=float, default=0.75)

args = parser.parse_args()

KB = 0.0019872041      # in kcal/mol/K

experiments = args.experiments.split()

bf_rmbm_vs_2cbm = {}
bf_embm_vs_2cbm = {}
bf_embm_vs_rmbm = {}

for experiment in experiments:
    print(experiment)
    q_actual_micro_cal = load_heat_micro_cal(os.path.join(args.heat_dir, experiment + ".DAT"))
    q_actual_cal = q_actual_micro_cal * 10**(-6)

    exper_info_2cbm = ITCExperiment(os.path.join(args.two_component_mcmc_dir, experiment, args.exper_info_file))
    exper_info_rmbm = ITCExperiment(os.path.join(args.racemic_mixture_mcmc_dir, experiment, args.exper_info_file))
    exper_info_embm = ITCExperiment(os.path.join(args.enantiomer_mcmc_dir, experiment, args.exper_info_file))

    trace_2cbm = pickle.load(open(os.path.join(args.two_component_mcmc_dir, experiment, args.mcmc_trace_file)))
    trace_rmbm = pickle.load(open(os.path.join(args.racemic_mixture_mcmc_dir, experiment, args.mcmc_trace_file)))
    trace_embm = pickle.load(open(os.path.join(args.enantiomer_mcmc_dir, experiment, args.mcmc_trace_file)))

    llh_mean_v1_2cbm, llh_mean_v2_2cbm = average_likelihood_from_posterior("2cbm", q_actual_cal, exper_info_2cbm,
                                                                           trace_2cbm,
                                                                           dcell=0.1, dsyringe=0.1,
                                                                           uniform_P0=args.uniform_P0,
                                                                           uniform_Ls=args.uniform_Ls,
                                                                           concentration_range_factor=args.concentration_range_factor,
                                                                           nsamples=None)

    llh_mean_v1_rmbm, llh_mean_v2_rmbm = average_likelihood_from_posterior("rmbm", q_actual_cal, exper_info_rmbm,
                                                                           trace_rmbm,
                                                                           dcell=0.1, dsyringe=0.1,
                                                                           uniform_P0=args.uniform_P0,
                                                                           uniform_Ls=args.uniform_Ls,
                                                                           concentration_range_factor=args.concentration_range_factor,
                                                                           nsamples=None)

    llh_mean_v1_embm, llh_mean_v2_embm = average_likelihood_from_posterior("embm", q_actual_cal, exper_info_embm,
                                                                           trace_embm,
                                                                           dcell=0.1, dsyringe=0.1,
                                                                           uniform_P0=args.uniform_P0,
                                                                           uniform_Ls=args.uniform_Ls,
                                                                           concentration_range_factor=args.concentration_range_factor,
                                                                           nsamples=None)

    bf_rmbm_vs_2cbm[experiment] = llh_mean_v1_rmbm / llh_mean_v1_2cbm
    bf_embm_vs_2cbm[experiment] = llh_mean_v1_embm / llh_mean_v1_2cbm
    bf_embm_vs_rmbm[experiment] = llh_mean_v1_embm / llh_mean_v1_rmbm

    print("aver_likelihood_2cbm: v1 = %0.5e, v2 = %0.5e" % (llh_mean_v1_2cbm, llh_mean_v2_2cbm))
    print("aver_likelihood_rmbm: v1 = %0.5e, v2 = %0.5e" % (llh_mean_v1_rmbm, llh_mean_v2_rmbm))
    print("aver_likelihood_embm: v1 = %0.5e, v2 = %0.5e" % (llh_mean_v1_rmbm, llh_mean_v2_rmbm))
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
ax.set_title("nsamples = %0.1e" % args.nsamples)
fig.tight_layout()
fig.savefig("bf_rmbm_vs_2cbm.pdf", dpi=300)


fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(3.2, 2.4))
bf_embm_vs_2cbm_log.plot(kind="barh", ax=ax)
ax.set_xlabel("$log \\frac{P(D|embm)}{P(D|2cbm)}$")
ax.set_title("nsamples = %0.1e" % args.nsamples)
fig.tight_layout()
fig.savefig("bf_embm_vs_2cbm.pdf", dpi=300)


fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(3.2, 2.4))
bf_embm_vs_rmbm_log.plot(kind="barh", ax=ax)
ax.set_xlabel("$log \\frac{P(D|embm)}{P(D|rmbm)}$")
ax.set_title("nsamples = %0.1e" % args.nsamples)
fig.tight_layout()
fig.savefig("bf_embm_vs_rmbm.pdf", dpi=300)

print("DONE!")
