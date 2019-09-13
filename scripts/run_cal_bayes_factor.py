"""
calculate bayes factor
"""

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("racemic_mixture_mcmc_dir", type=str,
                    default="/home/tnguye46/bayesian_itc_racemic/6.racemicmixture_mcmc/lognomP0_lognomLs_narrowUniformRho")

parser.add_argument("two_component_mcmc_dir", type=str,
                    default="/home/tnguye46/bayesian_itc_racemic/5.twocomponent_mcmc/lognomP0_lognomLs")

parser.add_argument("heat_dir", type=str, default="/home/tnguye46/bayesian_itc_racemic/3.dummy_itc_files")

parser.add_argument("experiments", type=str,
                    default="Fokkens_1_c Fokkens_1_d Fokkens_1_e Baum_59 Baum_60_1 Baum_60_2 Baum_60_3 Baum_60_4")

parser.add_argument("exper_info_file", type=str, default="experimental_information.pickle")
parser.add_argument("mcmc_trace_file", type=str, default="traces.pickle")

args = parser.parse_args()

experiments = args.experiments.split()

for experiment in experiments:
    exper_info = 