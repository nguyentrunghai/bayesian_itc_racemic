"""
calculate convergence of percentiles for important parameters in traces
"""

from __future__ import print_function

import argparse
import os
import pickle

import numpy as np

parser = argparse.ArgumentParser()

parser.add_argument("--mcmc_dir", type=str,
                    default="/home/tnguye46/bayesian_itc_racemic/07.twocomponent_mcmc/pymc3_nuts_2/collected_samples")

parser.add_argument("--experiments", type=str,
default="Fokkens_1_a Fokkens_1_b Fokkens_1_c Fokkens_1_d Fokkens_1_e Baum_57 Baum_59 Baum_60_1 Baum_60_2 Baum_60_3 Baum_60_4")

parser.add_argument("--vars", type=str, default="DeltaH DeltaG P0 Ls")

parser.add_argument("--sample_proportions", type=str, default="0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0")
parser.add_argument("--repeats", type=int, default=10)

args = parser.parse_args()


def percentiles(x, nsamples, repeats, q):
    perce = []
    for _ in range(repeats):
        rnd_x = np.random.choice(x, size=nsamples, replace=True)
        p = np.percentile(rnd_x, q)
        perce.append(p)

    perce = np.array(perce)
    p_mean = perce.mean(axis=0)
    p_err = perce.std(axis=0)

    return p_mean, p_err

