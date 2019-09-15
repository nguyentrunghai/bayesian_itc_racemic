"""
plot comparison between actual heats and model heats
for model heats, we use maximum a posterior estimates of parameters
"""

import argparse
parser = argparse.ArgumentParser()

parser.add_argument("--racemic_mixture_mcmc_dir", type=str,
                    default="/home/tnguye46/bayesian_itc_racemic/6.racemicmixture_mcmc/lognomP0_lognomLs_narrowUniformRho")

args = parser.parse_args()
