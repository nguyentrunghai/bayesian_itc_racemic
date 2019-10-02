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

parser.add_argument("--nsamples_list", type=str, default="1000")

parser.add_argument("--experiments", type=str, default="Fokkens_1_c Fokkens_1_d")

parser.add_argument("--font_scale", type=float, default=1)

args = parser.parse_args()

KB = 0.0019872041      # in kcal/mol/K

experiments = args.experiments.split()

nsamples_list = [int(nsamples) for nsamples in args.nsamples_list.split()]
print("nsamples_list:", nsamples_list)

sns.set(font_scale=args.font_scale)

for experiment in experiments:
    print(experiment)
    actual_heat_micro_cal = load_heat_micro_cal(os.path.join(args.heat_dir, experiment + ".DAT"))

    exper_info = ITCExperiment(os.path.join(args.exper_info_dir, experiment, args.exper_info_file))

    traces = pickle.load(open(os.path.join(args.prior_mcmc_dir, experiment, args.mcmc_trace_file)))

    bf_rmbm_vs_2cbm = []
    bf_embm_vs_2cbm = []
    bf_embm_vs_rmbm = []

    for nsamples in nsamples_list:
        print("nsamples", nsamples)
        llh_2cbm = average_likelihood_TwoComponentBindingModel(actual_heat_micro_cal,
                                                               V0=exper_info.get_cell_volume_liter(),
                                                               DeltaVn=exper_info.get_injection_volumes_liter(),
                                                               beta=1 / KB / exper_info.get_target_temperature_kelvin(),
                                                               n_injections=exper_info.get_number_injections(),
                                                               mcmc_trace=traces,
                                                               nsamples=nsamples)

        llh_rmbm = average_likelihood_RacemicMixtureBindingModel(actual_heat_micro_cal,
                                                                 V0=exper_info.get_cell_volume_liter(),
                                                                 DeltaVn=exper_info.get_injection_volumes_liter(),
                                                                 beta=1/KB/exper_info.get_target_temperature_kelvin(),
                                                                 n_injections=exper_info.get_number_injections(),
                                                                 mcmc_trace=traces,
                                                                 nsamples=nsamples)

        llh_embm = average_likelihood_EnantiomerBindingModel(actual_heat_micro_cal,
                                                             V0=exper_info.get_cell_volume_liter(),
                                                             DeltaVn=exper_info.get_injection_volumes_liter(),
                                                             beta=1 / KB / exper_info.get_target_temperature_kelvin(),
                                                             n_injections=exper_info.get_number_injections(),
                                                             mcmc_trace=traces,
                                                             nsamples=nsamples)

        bf_rmbm_vs_2cbm.append(llh_rmbm / llh_2cbm)
        bf_embm_vs_2cbm.append(llh_embm / llh_2cbm)
        bf_embm_vs_rmbm.append(llh_embm / llh_rmbm)

    # bf_rmbm_vs_2cbm
    fig, ax = plt.subplots(nrows=3, ncols=2, sharex=True, figsize=(6.4, 7.2))
    ax[0][0].plot(nsamples_list, bf_rmbm_vs_2cbm, c="k")
    ax[0][0].set_ylabel("$\\frac{P(D|rmbm)}{P(D|2cbm)}$")

    ax[0][1].plot(nsamples_list, np.log10(bf_rmbm_vs_2cbm), c="k")
    ax[0][1].set_ylabel("$log \\frac{P(D|rmbm)}{P(D|2cbm)}$")

    # bf_embm_vs_2cbm
    ax[1][0].plot(nsamples_list, bf_embm_vs_2cbm, c="k")
    ax[1][0].set_ylabel("$\\frac{P(D|embm)}{P(D|2cbm)}$")

    ax[1][1].plot(nsamples_list, np.log10(bf_embm_vs_2cbm), c="k")
    ax[1][1].set_ylabel("$log \\frac{P(D|embm)}{P(D|2cbm)}$")

    # bf_embm_vs_rmbm
    ax[2][0].plot(nsamples_list, bf_embm_vs_rmbm, c="k")
    ax[2][0].set_xlabel("# samples")
    ax[2][0].set_ylabel("$\\frac{P(D|embm)}{P(D|rmbm)}$")

    ax[2][1].plot(nsamples_list, np.log10(bf_embm_vs_rmbm), c="k")
    ax[2][1].set_xlabel("# samples")
    ax[2][1].set_ylabel("$log \\frac{P(D|embm)}{P(D|rmbm)}$")

    fig.tight_layout()
    out = experiment + ".pdf"
    fig.savefig(out, dpi=300)


print("DONE")
