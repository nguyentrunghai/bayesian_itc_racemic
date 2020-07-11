"""
to plot convergence curve for the percentiles of posteriors
"""

from __future__ import print_function

import argparse
import os

import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

parser = argparse.ArgumentParser()

parser.add_argument("--data_dir", type=str,
                default="/home/tnguye46/bayesian_itc_racemic/11.analyses/posterior_convergence/07.twocomponent_mcmc")

parser.add_argument("--experiments", type=str,
default="Fokkens_1_a Fokkens_1_b Fokkens_1_c Fokkens_1_d Fokkens_1_e Baum_57 Baum_59 Baum_60_1 Baum_60_2 Baum_60_3 Baum_60_4")

parser.add_argument("--percentiles", type=str, default="5 25 50 75 95")

parser.add_argument("--vars", type=str, default="DeltaH DeltaG P0 Ls")

parser.add_argument("--font_scale", type=float, default=0.75)

args = parser.parse_args()

sns.set(font_scale=args.font_scale)

experiments = args.experiments.split()
print("experiments:", experiments)

vars = args.vars.split()
assert len(vars) in [4, 6], "len of vars must be 4 or 6"
print("vars:", vars)

qs = [float(s) for s in args.percentiles.split()]
data_cols = ["%0.1-th" %q for q in qs]
print("data_cols:", data_cols)
err_cols = ["%0.1--error" %q for q in qs]
print("err_cols:", err_cols)
