"""
calculate bayes factor
"""
from __future__ import print_function

import os
import argparse
import pickle

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from _data_io import ITCExperiment, load_heat_micro_cal
from _bayes_factor import average_likelihood_from_posterior_bootstrap

parser = argparse.ArgumentParser()
parser.add_argument("--two_component_mcmc_dir", type=str, default="5.twocomponent_mcmc")
parser.add_argument("--racemic_mixture_mcmc_dir", type=str, default="6.racemicmixture_mcmc")
parser.add_argument("--enantiomer_mcmc_dir", type=str, default="7.enantiomer")

parser.add_argument("--heat_dir", type=str, default="4.heat_in_origin_format")

parser.add_argument("--exper_info_file", type=str, default="experimental_information.pickle")
parser.add_argument("--mcmc_trace_file", type=str, default="traces.pickle")

parser.add_argument("--concentration_range_factor", type=float, default=10.)

parser.add_argument("--experiments", type=str, default="Fokkens_1_c Fokkens_1_d")
parser.add_argument("--experiments_with_unif_concen_prior", type=str, default="Fokkens_1_a Fokkens_1_b")

parser.add_argument("--nsamples_list", type=str, default="100")
parser.add_argument("--repeats", type=int, default=1)

parser.add_argument("--font_scale", type=float, default=0.75)

args = parser.parse_args()

KB = 0.0019872041      # in kcal/mol/K

experiments = args.experiments.split()
print("experiments", experiments)
experiments_with_unif_concen_prior = args.experiments_with_unif_concen_prior.split()
print("experiments_with_unif_concen_prior", experiments_with_unif_concen_prior)
nsamples_list = [int(nsamples) for nsamples in args.nsamples_list.split()]
print("nsamples_list:", nsamples_list)

sns.set(font_scale=args.font_scale)

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

    exper_info_2cbm = ITCExperiment(os.path.join(args.two_component_mcmc_dir, experiment, args.exper_info_file))
    exper_info_rmbm = ITCExperiment(os.path.join(args.racemic_mixture_mcmc_dir, experiment, args.exper_info_file))
    exper_info_embm = ITCExperiment(os.path.join(args.enantiomer_mcmc_dir, experiment, args.exper_info_file))

    trace_2cbm = pickle.load(open(os.path.join(args.two_component_mcmc_dir, experiment, args.mcmc_trace_file)))
    trace_rmbm = pickle.load(open(os.path.join(args.racemic_mixture_mcmc_dir, experiment, args.mcmc_trace_file)))
    trace_embm = pickle.load(open(os.path.join(args.enantiomer_mcmc_dir, experiment, args.mcmc_trace_file)))

    bf_rmbm_vs_2cbm = []
    bf_embm_vs_2cbm = []
    bf_embm_vs_rmbm = []

    for nsamples in nsamples_list:
        print("nsamples = %d" % nsamples)

        llh_mean_2cbm, llh_max_log_2cbm = average_likelihood_from_posterior_bootstrap("2cbm", q_actual_cal, exper_info_2cbm,
                                                                        trace_2cbm,
                                                                        dcell=0.1, dsyringe=0.1,
                                                                        uniform_P0=uniform_P0,
                                                                        uniform_Ls=uniform_Ls,
                                                                        concentration_range_factor=args.concentration_range_factor,
                                                                        nsamples=nsamples,
                                                                        repeats=args.repeats)

        llh_mean_rmbm, llh_max_log_rmbm = average_likelihood_from_posterior_bootstrap("rmbm", q_actual_cal, exper_info_rmbm,
                                                                        trace_rmbm,
                                                                        dcell=0.1, dsyringe=0.1,
                                                                        uniform_P0=uniform_P0,
                                                                        uniform_Ls=uniform_Ls,
                                                                        concentration_range_factor=args.concentration_range_factor,
                                                                        nsamples=nsamples,
                                                                        repeats=args.repeats)

        llh_mean_embm, llh_max_log_embm = average_likelihood_from_posterior_bootstrap("embm", q_actual_cal, exper_info_embm,
                                                                        trace_embm,
                                                                        dcell=0.1, dsyringe=0.1,
                                                                        uniform_P0=uniform_P0,
                                                                        uniform_Ls=uniform_Ls,
                                                                        concentration_range_factor=args.concentration_range_factor,
                                                                        nsamples=nsamples,
                                                                        repeats=args.repeats)

        bf_rmbm_vs_2cbm.append(np.mean(llh_mean_rmbm / llh_mean_2cbm * np.exp(llh_max_log_rmbm - llh_max_log_2cbm)))
        bf_embm_vs_2cbm.append(np.mean(llh_mean_embm / llh_mean_2cbm * np.exp(llh_max_log_embm - llh_max_log_2cbm)))
        bf_embm_vs_rmbm.append(np.mean(llh_mean_embm / llh_mean_rmbm * np.exp(llh_max_log_embm - llh_max_log_rmbm)))

    # bf_rmbm_vs_2cbm
    fig, ax = plt.subplots(nrows=3, ncols=1, sharex=True, figsize=(4.0, 6.0))

    ax[0].plot(nsamples_list, np.log10(bf_rmbm_vs_2cbm), c="k", marker=".")
    ax[0].set_ylabel("$log \\left[ \\frac{P(D|rmbm)}{P(D|2cbm)} \\right]$")
    ax[0].locator_params(axis='y', nbins=7)

    # bf_embm_vs_2cbm

    ax[1].plot(nsamples_list, np.log10(bf_embm_vs_2cbm), c="k", marker=".")
    ax[1].set_ylabel("$log \\left[ \\frac{P(D|embm)}{P(D|2cbm)} \\right]$")
    ax[1].locator_params(axis='y', nbins=7)

    # bf_embm_vs_rmbm

    ax[2].plot(nsamples_list, np.log10(bf_embm_vs_rmbm), c="k", marker=".")
    ax[2].set_xlabel("# samples")
    ax[2].xaxis.get_major_formatter().set_powerlimits((0, 1))
    ax[2].set_ylabel("$log \\left[ \\frac{P(D|embm)}{P(D|rmbm)} \\right]$")
    ax[2].locator_params(axis='x', nbins=7)
    ax[2].locator_params(axis='y', nbins=7)

    fig.tight_layout()
    out = experiment + ".pdf"
    fig.savefig(out, dpi=300)

print("DONE!")
