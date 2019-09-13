"""
calculate bayes factor
"""
from __future__ import print_function

import os
import argparse
import pickle

from _data_io import ITCExperiment, load_heat_micro_cal

parser = argparse.ArgumentParser()
parser.add_argument("racemic_mixture_mcmc_dir", type=str,
                    default="/home/tnguye46/bayesian_itc_racemic/6.racemicmixture_mcmc/lognomP0_lognomLs_narrowUniformRho")

parser.add_argument("two_component_mcmc_dir", type=str,
                    default="/home/tnguye46/bayesian_itc_racemic/5.twocomponent_mcmc/lognomP0_lognomLs")

parser.add_argument("heat_dir", type=str, default="/home/tnguye46/bayesian_itc_racemic/3.dummy_itc_files")

parser.add_argument("exper_info_file", type=str, default="experimental_information.pickle")
parser.add_argument("mcmc_trace_file", type=str, default="traces.pickle")

parser.add_argument("experiments", type=str,
                    default="Fokkens_1_c Fokkens_1_d Fokkens_1_e Baum_60_1 Baum_60_2 Baum_60_3 Baum_60_4")


args = parser.parse_args()

experiments = args.experiments.split()

for experiment in experiments:
    print(experiment)
    actual_heat_micro_cal = load_heat_micro_cal(os.path.join(args.heat_dir, experiment + ".DAT"))

    exper_info_racemic_mixture = ITCExperiment(os.path.join(args.racemic_mixture_mcmc_dir,
                                                            experiment, args.exper_info_file))
    exper_info_two_component = ITCExperiment(os.path.join(args.two_component_mcmc_dir,
                                                          experiment, args.exper_info_file))

    trace_racemic_mixture = pickle.load(open(os.path.join(args.racemic_mixture_mcmc_dir,
                                                          experiment, args.mcmc_trace_file)))
    trace_two_component = pickle.load(open(os.path.join(args.two_component_mcmc_dir,
                                                        experiment, args.mcmc_trace_file)))