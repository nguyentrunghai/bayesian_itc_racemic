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

parser.add_argument("--draws", type=int, default=10000)
parser.add_argument("--tune", type=int, default=2000)
parser.add_argument("--cores", type=int, default=4)

parser.add_argument("--experiments", type=str, default="Fokkens_1_c Fokkens_1_d")

parser.add_argument("--write_qsub_script", action="store_true", default=False)
parser.add_argument("--submit", action="store_true", default=False)
args = parser.parse_args()


if args.write_qsub_script:
    this_script = os.path.abspath(sys.argv[0])
    experiments = args.experiments.split()

    dP0 = args.dP0
    dLs = args.dLs

    uniform_P0 = " "
    if args.uniform_P0:
        uniform_P0 = " --uniform_P0 "

    uniform_Ls = " "
    if args.uniform_Ls:
        uniform_Ls = " --uniform_Ls "

    concentration_range_factor = args.concentration_range_factor

    draws = args.draws
    tune = args.tune
    cores = args.cores
    