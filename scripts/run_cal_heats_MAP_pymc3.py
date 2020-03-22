"""
Calculate heat from MAP estimate of the parameters
"""

from __future__ import print_function

import argparse
import os
import pickle

import numpy as np
import pandas as pd
import pymc3

import matplotlib.pyplot as plt
import seaborn as sns

from _data_io import ITCExperiment, load_heat_micro_cal
from _models import heats_TwoComponentBindingModel, heats_RacemicMixtureBindingModel

parser = argparse.ArgumentParser()

parser.add_argument("--two_component_mcmc_dir", type=str,
                    default="/home/tnguye46/bayesian_itc_racemic/07.twocomponent_mcmc/pymc3_met_2/repeat_0")
parser.add_argument("--racemic_mixture_mcmc_dir", type=str,
                    default="/home/tnguye46/bayesian_itc_racemic/08.racemicmixture_mcmc/pymc3_met_2/repeat_0")
parser.add_argument("--enantiomer_mcmc_dir", type=str,
                    default="/home/tnguye46/bayesian_itc_racemic/09.enantiomer_mcmc/pymc3_met_2/repeat_0")

parser.add_argument("--model_pickle", type=str, default="pm_model.pickle")

parser.add_argument("--experiments", type=str,
default="Fokkens_1_a Fokkens_1_b Fokkens_1_c Fokkens_1_d Fokkens_1_e Baum_57 Baum_59 Baum_60_1 Baum_60_2 Baum_60_3 Baum_60_4")

parser.add_argument("--font_scale", type=float, default=0.75)
parser.add_argument("--xlabel", type=str, default="# injections")
parser.add_argument("--ylabel", type=str, default="heat ($\mu$cal)")

args = parser.parse_args()

def find_MAP(model, method="L-BFGS-B"):
    return pymc3.find_MAP(model=model, method=method)

sns.set(font_scale=args.font_scale)

KB = 0.0019872041      # in kcal/mol/K

experiments = args.experiments.split()

for exper in experiments:
    print("\n\n", exper)

    model_2c = pickle.load(open(os.path.join(args.two_component_mcmc_dir, exper, args.model_pickle)))
    model_rm = pickle.load(open(os.path.join(args.racemic_mixture_mcmc_dir, exper, args.model_pickle)))
    model_em = pickle.load(open(os.path.join(args.enantiomer_mcmc_dir, exper, args.model_pickle)))

    print("Calculating MAP_2C")
    map_2c = find_MAP(model_2c)
    print("map_2c", map_2c)

    print("Calculating MAP_RM")
    map_rm = find_MAP(model_rm)
    print("map_rm", map_rm)

    print("Calculating MAP_EM")
    map_em = find_MAP(model_em)
    print("map_em", map_em)

    
