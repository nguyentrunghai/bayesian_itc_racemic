"""
to submit and run mcmc jobs
"""

import os
import sys
import argparse

import pickle

import pymc3
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

from _data_io import ITCExperiment, load_heat_micro_cal
from _pymc3_models import make_TwoComponentBindingModel, make_RacemicMixtureBindingModel

parser = argparse.ArgumentParser()
parser.add_argument("--exper_info_dir", type=str, default="5.twocomponent_mcmc")
parser.add_argument("--exper_info_file", type=str, default="experimental_information.pickle")

parser.add_argument("--heat_dir", type=str, default="4.heat_in_origin_format")
parser.add_argument("--heat_file", type=str, default="heat.DAT")

parser.add_argument("--dP0", type=float, default=0.1)      # cell concentration relative uncertainty
parser.add_argument("--dLs", type=float, default=0.1)      # syringe concentration relative uncertainty

parser.add_argument("--uniform_P0", action="store_true", default=False)
parser.add_argument("--uniform_Ls", action="store_true", default=False)
parser.add_argument("--concentration_range_factor", type=float, default=10.)

parser.add_argument("--nsamples", type=int, default=1000000)
parser.add_argument("--nburn", type=int, default=10000)
parser.add_argument("--nthin", type=int, default=100)

parser.add_argument("--experiments", type=str, default="Fokkens_1_c Fokkens_1_d")
args = parser.parse_args()
