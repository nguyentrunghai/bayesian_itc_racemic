"""
calculate BIC and WBIC
"""

from __future__ import print_function

import argparse
import glob
import os
import pickle

import numpy as np
import pandas as pd

from _information_criterion import get_n_injections, get_n_params
from _information_criterion import load_model, get_values_from_trace_files

parser = argparse.ArgumentParser()

parser.add_argument("--two_component_mcmc_dir", type=str,
                    default="/home/tnguye46/bayesian_itc_racemic/07.twocomponent_mcmc/pymc3_nuts_2")
parser.add_argument("--racemic_mixture_mcmc_dir", type=str,
                    default="/home/tnguye46/bayesian_itc_racemic/08.racemicmixture_mcmc/pymc3_nuts_2")
parser.add_argument("--enantiomer_mcmc_dir", type=str,
                    default="/home/tnguye46/bayesian_itc_racemic/09.enantiomer_mcmc/pymc3_nuts_2")

parser.add_argument("--repeat_prefix", type=str, default="repeat_")
parser.add_argument("--exclude_repeats", type=str, default="")

parser.add_argument("--model_pickle", type=str, default="pm_model.pickle")
parser.add_argument("--trace_pickle", type=str, default="trace_obj.pickle")

parser.add_argument("--heat_data_dir", type=str, default="/home/tnguye46/bayesian_itc_racemic/04.heat_in_origin_format")

parser.add_argument("--experiments", type=str,
default="Fokkens_1_a Fokkens_1_b Fokkens_1_c Fokkens_1_d Fokkens_1_e Baum_57 Baum_59 Baum_60_1 Baum_60_2 Baum_60_3 Baum_60_4")

args = parser.parse_args()


def is_path_excluded(path, exclude_kws):
    for kw in exclude_kws:
        if kw in path:
            return True
    return False


exclude_repeats = args.exclude_repeats.split()
exclude_repeats = [args.repeat_prefix + r for r in exclude_repeats]
print("exclude_repeats:", exclude_repeats)

experiments = args.experiments.split()
print("experiments", experiments)


for exper in experiments:
    print("\n\n", exper)

    dirs_2c = glob.glob(os.path.join(args.two_component_mcmc_dir, args.repeat_prefix + "*", exper, args.trace_pickle))
    dirs_2c = [os.path.dirname(p) for p in dirs_2c]
    dirs_2c = [p for p in dirs_2c if not is_path_excluded(p, exclude_repeats)]
    print("dirs_2c:", dirs_2c)

    dirs_rm = glob.glob(os.path.join(args.racemic_mixture_mcmc_dir, args.repeat_prefix + "*", exper, args.trace_pickle))
    dirs_rm = [os.path.dirname(p) for p in dirs_rm]
    dirs_rm = [p for p in dirs_rm if not is_path_excluded(p, exclude_repeats)]
    print("dirs_rm:", dirs_rm)

    dirs_em = glob.glob(os.path.join(args.enantiomer_mcmc_dir, args.repeat_prefix + "*", exper, args.trace_pickle))
    dirs_em = [os.path.dirname(p) for p in dirs_em]
    dirs_em = [p for p in dirs_em if not is_path_excluded(p, exclude_repeats)]
    print("dirs_em:", dirs_em)

    heat_file = os.path.join(args.heat_data_dir, exper + ".DAT")
    n_injections = get_n_injections(heat_file)
    print("n_injections:", n_injections)

    model_2c = load_model(os.path.join(dirs_2c[0], args.model_pickle))
    model_rm = load_model(os.path.join(dirs_rm[0], args.model_pickle))
    model_em = load_model(os.path.join(dirs_em[0], args.model_pickle))

    trace_files_2c = [os.path.join(d, args.trace_pickle) for d in dirs_2c]
    trace_files_rm = [os.path.join(d, args.trace_pickle) for d in dirs_rm]
    trace_files_em = [os.path.join(d, args.trace_pickle) for d in dirs_em]

    traces_2c = get_values_from_trace_files(model_2c, trace_files_2c)
    traces_rm = get_values_from_trace_files(model_rm, trace_files_rm)
    traces_em = get_values_from_trace_files(model_rm, trace_files_em)
    
