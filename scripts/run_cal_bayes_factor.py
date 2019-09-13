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
sns.set()

from _data_io import ITCExperiment, load_heat_micro_cal
from _models import average_likelihood_TwoComponentBindingModel, average_likelihood_RacemicMixtureBindingModel

parser = argparse.ArgumentParser()
parser.add_argument("--racemic_mixture_mcmc_dir", type=str,
                    default="/home/tnguye46/bayesian_itc_racemic/6.racemicmixture_mcmc/lognomP0_lognomLs_narrowUniformRho")

parser.add_argument("--two_component_mcmc_dir", type=str,
                    default="/home/tnguye46/bayesian_itc_racemic/5.twocomponent_mcmc/lognomP0_lognomLs")

parser.add_argument("--heat_dir", type=str, default="/home/tnguye46/bayesian_itc_racemic/4.heat_in_origin_format")

parser.add_argument("--exper_info_file", type=str, default="experimental_information.pickle")
parser.add_argument("--mcmc_trace_file", type=str, default="traces.pickle")

parser.add_argument("--experiments", type=str,
                    default="Fokkens_1_c Fokkens_1_d Fokkens_1_e Baum_60_1 Baum_60_2 Baum_60_3 Baum_60_4")


args = parser.parse_args()

KB = 0.0019872041      # in kcal/mol/K

experiments = args.experiments.split()

bayes_factors = {}
for experiment in experiments:
    print(experiment)
    actual_heat_micro_cal = load_heat_micro_cal(os.path.join(args.heat_dir, experiment + ".DAT"))

    exper_info_rmbm = ITCExperiment(os.path.join(args.racemic_mixture_mcmc_dir,
                                                 experiment, args.exper_info_file))
    exper_info_2cbm = ITCExperiment(os.path.join(args.two_component_mcmc_dir,
                                                 experiment, args.exper_info_file))

    trace_rmbm = pickle.load(open(os.path.join(args.racemic_mixture_mcmc_dir,
                                               experiment, args.mcmc_trace_file)))
    trace_2cbm = pickle.load(open(os.path.join(args.two_component_mcmc_dir,
                                                        experiment, args.mcmc_trace_file)))

    aver_likelihood_rmbm = average_likelihood_RacemicMixtureBindingModel(actual_heat_micro_cal,
                                                                         V0=exper_info_rmbm.get_cell_volume_liter(),
                                                                         DeltaVn=exper_info_rmbm.get_injection_volumes_liter(),
                                                                         beta=1/KB/exper_info_rmbm.get_target_temperature_kelvin(),
                                                                         n_injections=exper_info_rmbm.get_number_injections(),
                                                                         mcmc_trace=trace_rmbm)

    aver_likelihood_2cbm = average_likelihood_TwoComponentBindingModel(actual_heat_micro_cal,
                                                                       V0=exper_info_2cbm.get_cell_volume_liter(),
                                                                       DeltaVn=exper_info_2cbm.get_injection_volumes_liter(),
                                                                       beta=1 / KB / exper_info_2cbm.get_target_temperature_kelvin(),
                                                                       n_injections=exper_info_2cbm.get_number_injections(),
                                                                       mcmc_trace=trace_2cbm)

    bayes_factor = aver_likelihood_rmbm / aver_likelihood_2cbm
    bayes_factors[experiment] = bayes_factor

    print("aver_likelihood_rmbm: %0.5e" % aver_likelihood_rmbm)
    print("aver_likelihood_2cbm: %0.5e" % aver_likelihood_2cbm)
    print("Bayes factor: %0.5e" % bayes_factor)
    print("")


bayes_factors = pd.Series(bayes_factors)
bayes_factors.sort_values(ascending=False, inplace=True)
bayes_factors_log = np.log(bayes_factors)

# plot
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=((3.2, 2.4)))
bayes_factors.plot(kind="barh")
ax.set_xlabel("log[Bayes factor]")
plt.tight_layout()
plt.savefig("test.pdf", dpi=300)