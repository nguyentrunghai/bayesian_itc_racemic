"""
To collect samples and logp from multiples NUTS runs
"""

from __future__ import print_function

import argparse
import os
import glob
import pickle

from _bayes_factor import get_values_from_traces
from _bayes_factor import log_posterior_trace

parser = argparse.ArgumentParser()
parser.add_argument("--mcmc_dir", type=str,
                    default="/home/tnguye46/bayesian_itc_racemic/07.twocomponent_mcmc/pymc3_nuts_2")

parser.add_argument("--repeat_prefix", type=str, default="repeat_")
parser.add_argument("--exclude_repeats", type=str, default="")

parser.add_argument("--model_pickle", type=str, default="pm_model.pickle")
parser.add_argument("--trace_pickle", type=str, default="trace_obj.pickle")

parser.add_argument("--experiments", type=str,
default="Fokkens_1_a Fokkens_1_b Fokkens_1_c Fokkens_1_d Fokkens_1_e Baum_57 Baum_59 Baum_60_1 Baum_60_2 Baum_60_3 Baum_60_4")


args = parser.parse_args()
