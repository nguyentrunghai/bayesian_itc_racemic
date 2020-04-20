"""
calculate Bayes factors using BAR estimator
"""

from __future__ import print_function

import argparse
import os
import glob
import pickle

import numpy as np
import pandas as pd

from _bayes_factor import get_values_from_traces
from _bayes_factor import bayes_factor_v1, bayes_factor_v2

parser = argparse.ArgumentParser()

parser.add_argument("--two_component_mcmc_dir", type=str,
                    default="/home/tnguye46/bayesian_itc_racemic/07.twocomponent_mcmc/pymc3_met_2")
parser.add_argument("--racemic_mixture_mcmc_dir", type=str,
                    default="/home/tnguye46/bayesian_itc_racemic/08.racemicmixture_mcmc/pymc3_met_2")
parser.add_argument("--enantiomer_mcmc_dir", type=str,
                    default="/home/tnguye46/bayesian_itc_racemic/09.enantiomer_mcmc/pymc3_met_2")

parser.add_argument("--repeat_prefix", type=str, default="repeat_")
# inclusive
parser.add_argument("--repeat_range", type=str, default="first last")

parser.add_argument("--model_pickle", type=str, default="pm_model.pickle")
parser.add_argument("--trace_pickle", type=str, default="trace_obj.pickle")

parser.add_argument("--experiments", type=str,
default="Fokkens_1_a Fokkens_1_b Fokkens_1_c Fokkens_1_d Fokkens_1_e Baum_57 Baum_59 Baum_60_1 Baum_60_2 Baum_60_3 Baum_60_4")

parser.add_argument("--estimator_version", type=int, default=1)

# "Normal", "Uniform", "GaussMix"
parser.add_argument("--aug_with", type=str, default="GaussMix")

parser.add_argument("--n_components", type=int, default=1)
# 'full', 'tied', 'diag', 'spherical'
# read here for explanation: https://scikit-learn.org/stable/modules/generated/sklearn.mixture.GaussianMixture.html
parser.add_argument("--covariance_type", type=str, default="full")

parser.add_argument("--aug_sample_enlarge", type=int, default=1)

parser.add_argument("--burn", type=int, default=0)
parser.add_argument("--thin", type=int, default=1)

parser.add_argument("--sigma_robust", action="store_true", default=False)
parser.add_argument("--bootstrap", type=int, default=None)

parser.add_argument("--csv_out", type=str, default="bayes_factors.csv")
args = parser.parse_args()


def enlarge_sample(sample, enlarge=1):
    sample_new = sample.copy()
    for k in sample_new.keys():
        sample_new[k] = np.repeat(sample_new[k], enlarge)
    return sample_new


def is_path_in_repeat_range(path, repeat_prefix, repeat_range):
    pieces = path.split("/")
    repeat_p = None
    for p in pieces:
        if repeat_prefix in p:
            repeat_p = p

    if repeat_p is None:
        return False
    num = int(repeat_p.split("_")[-1])
    if (num >= repeat_range[0]) and num <= repeat_range[1]:
        return True
    else:
        return False


experiments = args.experiments.split()
print("experiments:", experiments)

repeat_range = [int(s) for s in args.repeat_range.split()]
assert len(repeat_range) == 2, "repeat_range must have two numbers."
assert repeat_range[1] >= repeat_range[0], "the second number must be greater than or equal to the first."

if args.estimator_version == 1:
    bayes_factor = bayes_factor_v1
elif args.estimator_version == 2:
    bayes_factor = bayes_factor_v2
else:
    raise ValueError("Unknown version: %d" % args.estimator_version)
    
bf_df = []
for exper in experiments:
    print("\n\nCalculating Bayes Factors for " + exper)

    dirs_2c = glob.glob(os.path.join(args.two_component_mcmc_dir, args.repeat_prefix + "*", exper))
    dirs_2c = [p for p in dirs_2c if is_path_in_repeat_range(p, args.repeat_prefix, repeat_range)]
    print("dirs_2c:", dirs_2c)

    dirs_rm = glob.glob(os.path.join(args.racemic_mixture_mcmc_dir, args.repeat_prefix + "*", exper))
    dirs_rm = [p for p in dirs_rm if is_path_in_repeat_range(p, args.repeat_prefix, repeat_range)]
    print("dirs_rm:", dirs_rm)

    dirs_em = glob.glob(os.path.join(args.enantiomer_mcmc_dir, args.repeat_prefix + "*", exper))
    dirs_em = [p for p in dirs_em if is_path_in_repeat_range(p, args.repeat_prefix, repeat_range)]
    print("dirs_em:", dirs_em)

    # load data for 2cbm
    model_2c = pickle.load(open(os.path.join(dirs_2c[0], args.model_pickle)))
    trace_list_2c = [pickle.load(open(os.path.join(d, args.trace_pickle))) for d in dirs_2c]
    sample_2c = get_values_from_traces(model_2c, trace_list_2c, thin=args.thin, burn=args.burn)

    # load data for rmbm
    model_rm = pickle.load(open(os.path.join(dirs_rm[0], args.model_pickle)))
    trace_list_rm = [pickle.load(open(os.path.join(d, args.trace_pickle))) for d in dirs_rm]
    sample_rm = get_values_from_traces(model_rm, trace_list_rm, thin=args.thin, burn=args.burn)

    # load data for embm
    model_em = pickle.load(open(os.path.join(dirs_em[0], args.model_pickle)))
    trace_list_em = [pickle.load(open(os.path.join(d, args.trace_pickle))) for d in dirs_em]
    sample_em = get_values_from_traces(model_em, trace_list_em, thin=args.thin, burn=args.burn)

    print("\nRM over 2C")
    result_rm_over_2c = bayes_factor(model_2c, enlarge_sample(sample_2c, enlarge=args.aug_sample_enlarge),
                                     model_rm, sample_rm,
                                     model_ini_name="2c", model_fin_name="rm",
                                     aug_with=args.aug_with,
                                     sigma_robust=args.sigma_robust,
                                     n_components=args.n_components, covariance_type=args.covariance_type,
                                     bootstrap=args.bootstrap)

    print("\nEM over 2C")
    result_em_over_2c = bayes_factor(model_2c, enlarge_sample(sample_2c, enlarge=args.aug_sample_enlarge),
                                     model_em, sample_em,
                                     model_ini_name="2c", model_fin_name="em",
                                     aug_with=args.aug_with,
                                     sigma_robust=args.sigma_robust,
                                     n_components=args.n_components, covariance_type=args.covariance_type,
                                     bootstrap=args.bootstrap)

    print("\nEM over RM")
    result_em_over_rm = bayes_factor(model_rm, enlarge_sample(sample_rm, enlarge=args.aug_sample_enlarge),
                                     model_em, sample_em,
                                     model_ini_name="rm", model_fin_name="em",
                                     aug_with=args.aug_with,
                                     sigma_robust=args.sigma_robust,
                                     n_components=args.n_components, covariance_type=args.covariance_type,
                                     bootstrap=args.bootstrap)

    if args.bootstrap is not None:
        bf_rm_over_2c, err_rm_over_2c = result_rm_over_2c
        bf_em_over_2c, err_em_over_2c = result_em_over_2c
        bf_em_over_rm, err_em_over_rm = result_em_over_rm
    else:
        bf_rm_over_2c = result_rm_over_2c
        err_rm_over_2c = None

        bf_em_over_2c = result_em_over_2c
        err_em_over_2c = None

        bf_em_over_rm = result_em_over_rm
        err_em_over_rm = None

    res_dic = {"Experiment": exper,
               "bf_rm_over_2c": bf_rm_over_2c, "err_rm_over_2c": err_rm_over_2c,
               "bf_em_over_2c": bf_em_over_2c, "err_em_over_2c": err_em_over_2c,
               "bf_em_over_rm": bf_em_over_rm, "err_em_over_rm": err_em_over_rm}
    bf_df.append(res_dic)

    print("------------------------------")

bf_df = pd.DataFrame(bf_df)
cols = ["Experiment", "bf_rm_over_2c", "err_rm_over_2c", "bf_em_over_2c", "err_em_over_2c",
        "bf_em_over_rm", "err_em_over_rm"]
bf_df = bf_df[cols]
bf_df.to_csv(args.csv_out, index=False)

print("Done")
