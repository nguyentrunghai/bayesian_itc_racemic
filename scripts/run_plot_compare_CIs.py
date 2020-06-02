"""
To compare confidence intervals between model with log-normal priors for concentrations and
model with flat priors for concentrations.
"""

from __future__ import print_function

import argparse
import os
import pickle
import glob

import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

from _bayes_factor import get_values_from_trace, log_posterior_trace

parser = argparse.ArgumentParser()
parser.add_argument("--two_component_lognormal_dir", type=str,
                    default="/home/tnguye46/bayesian_itc_racemic/07.twocomponent_mcmc/pymc3_nuts_2")
parser.add_argument("--racemic_mixture_lognormal_dir", type=str,
                    default="/home/tnguye46/bayesian_itc_racemic/08.racemicmixture_mcmc/pymc3_nuts_2")
parser.add_argument("--enantiomer_lognormal_dir", type=str,
                    default="/home/tnguye46/bayesian_itc_racemic/09.enantiomer_mcmc/pymc3_nuts_2")

parser.add_argument("--two_component_flat_dir", type=str,
                    default="/home/tnguye46/bayesian_itc_racemic/07.twocomponent_mcmc/pymc3_nuts_4")
parser.add_argument("--racemic_mixture_flat_dir", type=str,
                    default="/home/tnguye46/bayesian_itc_racemic/08.racemicmixture_mcmc/pymc3_nuts_4")
parser.add_argument("--enantiomer_flat_dir", type=str,
                    default="/home/tnguye46/bayesian_itc_racemic/09.enantiomer_mcmc/pymc3_nuts_4")

parser.add_argument("--repeat_prefix", type=str, default="repeat_")

parser.add_argument("--trace_pickle", type=str, default="trace_obj.pickle")
parser.add_argument("--model_pickle", type=str, default="pm_model.pickle")

parser.add_argument("--experiments", type=str,
default="Fokkens_1_c Fokkens_1_d Fokkens_1_e")

parser.add_argument("--font_scale", type=float, default=0.75)

args = parser.parse_args()
