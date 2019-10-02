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
from _bayes_factor import average_likelihood_TwoComponentBindingModel
from _bayes_factor import average_likelihood_RacemicMixtureBindingModel
from _bayes_factor import average_likelihood_EnantiomerBindingModel

parser = argparse.ArgumentParser()
parser.add_argument("--prior_mcmc_dir", type=str, default="8.prior_mcmc")
parser.add_argument("--exper_info_dir", type=str, default="6.racemicmixture_mcmc")

parser.add_argument("--heat_dir", type=str, default="4.heat_in_origin_format")

parser.add_argument("--exper_info_file", type=str, default="experimental_information.pickle")
parser.add_argument("--mcmc_trace_file", type=str, default="traces.pickle")

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
    actual_heat_micro_cal = load_heat_micro_cal(os.path.join(args.heat_dir, experiment + ".DAT"))

    exper_info = ITCExperiment(os.path.join(args.exper_info_dir, experiment, args.exper_info_file))

    traces = pickle.load(open(os.path.join(args.prior_mcmc_dir, experiment, args.mcmc_trace_file)))

    aver_likelihood_2cbm = average_likelihood_TwoComponentBindingModel(actual_heat_micro_cal,
                                                                       V0=exper_info.get_cell_volume_liter(),
                                                                       DeltaVn=exper_info.get_injection_volumes_liter(),
                                                                       beta=1 / KB / exper_info.get_target_temperature_kelvin(),
                                                                       n_injections=exper_info.get_number_injections(),
                                                                       mcmc_trace=traces,
                                                                       nsamples=args.nsamples)

    aver_likelihood_rmbm = average_likelihood_RacemicMixtureBindingModel(actual_heat_micro_cal,
                                                                         V0=exper_info.get_cell_volume_liter(),
                                                                         DeltaVn=exper_info.get_injection_volumes_liter(),
                                                                         beta=1/KB/exper_info.get_target_temperature_kelvin(),
                                                                         n_injections=exper_info.get_number_injections(),
                                                                         mcmc_trace=traces,
                                                                         nsamples=args.nsamples)

    aver_likelihood_embm = average_likelihood_EnantiomerBindingModel(actual_heat_micro_cal,
                                                                     V0=exper_info.get_cell_volume_liter(),
                                                                     DeltaVn=exper_info.get_injection_volumes_liter(),
                                                                     beta=1 / KB / exper_info.get_target_temperature_kelvin(),
                                                                     n_injections=exper_info.get_number_injections(),
                                                                     mcmc_trace=traces,
                                                                     nsamples=args.nsamples)

    bf_rmbm_vs_2cbm[experiment] = aver_likelihood_rmbm / aver_likelihood_2cbm
    bf_embm_vs_2cbm[experiment] = aver_likelihood_embm / aver_likelihood_2cbm
    bf_embm_vs_rmbm[experiment] = aver_likelihood_embm / aver_likelihood_rmbm

    print("aver_likelihood_2cbm: %0.5e" % aver_likelihood_2cbm)
    print("aver_likelihood_rmbm: %0.5e" % aver_likelihood_rmbm)
    print("aver_likelihood_embm: %0.5e" % aver_likelihood_embm)
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
ax.set_xlabel("log $\\frac{P(D|rmbm)}{P(D|2cbm)}$")
fig.tight_layout()
fig.savefig(bf_rmbm_vs_2cbm + ".pdf", dpi=300)


fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(3.2, 2.4))
bf_embm_vs_2cbm_log.plot(kind="barh", ax=ax)
ax.set_xlabel("log $\\frac{P(D|embm)}{P(D|2cbm)}$")
fig.tight_layout()
fig.savefig(bf_embm_vs_2cbm + ".pdf", dpi=300)


fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(3.2, 2.4))
bf_embm_vs_rmbm_log.plot(kind="barh", ax=ax)
ax.set_xlabel("log $\\frac{P(D|embm)}{P(D|rmbm)}$")
fig.tight_layout()
fig.savefig(bf_embm_vs_rmbm + ".pdf", dpi=300)

print("DONE!")
