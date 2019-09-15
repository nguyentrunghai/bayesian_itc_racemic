"""
plot comparison between actual heats and model heats
for model heats, we use maximum a posterior estimates of parameters
"""
from __future__ import print_function

import os
import argparse
import pickle

from _data_io import ITCExperiment, load_heat_micro_cal
from _models import map_TwoComponentBindingModel, map_RacemicMixtureBindingModel

parser = argparse.ArgumentParser()

parser.add_argument("--racemic_mixture_mcmc_dir", type=str,
                    default="/home/tnguye46/bayesian_itc_racemic/6.racemicmixture_mcmc/lognomP0_lognomLs_narrowUniformRho")

parser.add_argument("--two_component_mcmc_dir", type=str,
                    default="/home/tnguye46/bayesian_itc_racemic/5.twocomponent_mcmc/lognomP0_lognomLs")

parser.add_argument("--heat_dir", type=str, default="/home/tnguye46/bayesian_itc_racemic/4.heat_in_origin_format")

parser.add_argument("--exper_info_file", type=str, default="experimental_information.pickle")
parser.add_argument("--mcmc_trace_file", type=str, default="traces.pickle")

parser.add_argument("--experiments", type=str,
                    default="Fokkens_1_c Fokkens_1_d Fokkens_1_e Baum_59 Baum_60_1 Baum_60_2 Baum_60_3 Baum_60_4")

parser.add_argument("--stated_rho", type=float, default=0.5)
parser.add_argument("--drho", type=float, default=0.1)

args = parser.parse_args()

KB = 0.0019872041      # in kcal/mol/K

experiments = args.experiments.split()

for experiment in experiments:
    print(experiment)
    actual_q_micro_cal = load_heat_micro_cal(os.path.join(args.heat_dir, experiment + ".DAT"))
    actual_q_cal = actual_q_micro_cal * 10**(-6)

    exper_info_2cbm = ITCExperiment(os.path.join(args.two_component_mcmc_dir,
                                                 experiment, args.exper_info_file))
    exper_info_rmbm = ITCExperiment(os.path.join(args.racemic_mixture_mcmc_dir,
                                                 experiment, args.exper_info_file))

    trace_2cbm = pickle.load(open(os.path.join(args.two_component_mcmc_dir,
                                               experiment, args.mcmc_trace_file)))
    trace_rmbm = pickle.load(open(os.path.join(args.racemic_mixture_mcmc_dir,
                                               experiment, args.mcmc_trace_file)))

    map_P0_2cbm, map_Ls_2cbm, map_DeltaG_2cbm, map_DeltaH_2cbm, map_DeltaH_0_2cbm = map_TwoComponentBindingModel(
                                                                                        actual_q_cal,
                                                                                        exper_info_2cbm, trace_2cbm)

    map_P0_rmbm, map_Ls_rmbm, map_rho_rmbm, map_DeltaG1_rmbm, \
    map_DeltaDeltaG_rmbm, map_DeltaH1_rmbm, map_DeltaH2_rmbm, map_DeltaH_0_rmbm = map_RacemicMixtureBindingModel(
                                                                            actual_q_cal, exper_info_rmbm, trace_rmbm,
                                                                            uniform_rho=True,
                                                                            stated_rho=args.stated_rh,
                                                                            drho=args.drho)