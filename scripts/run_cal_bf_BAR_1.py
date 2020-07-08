"""
calculate Bayes factors using BAR estimator
logp is precomputed
"""

from __future__ import print_function

import argparse
import os
import pickle

import numpy as np
import pandas as pd

from _bayes_factor import bayes_factor_v1, bayes_factor_v2

parser = argparse.ArgumentParser()

parser.add_argument("--two_component_mcmc_dir", type=str,
                    default="/home/tnguye46/bayesian_itc_racemic/07.twocomponent_mcmc/pymc3_nuts_2")
parser.add_argument("--racemic_mixture_mcmc_dir", type=str,
                    default="/home/tnguye46/bayesian_itc_racemic/08.racemicmixture_mcmc/pymc3_nuts_2")
parser.add_argument("--enantiomer_mcmc_dir", type=str,
                    default="/home/tnguye46/bayesian_itc_racemic/09.enantiomer_mcmc/pymc3_nuts_2")

parser.add_argument("--collected_trace_dir", type=str, default="collected_samples")

parser.add_argument("--model_dir", type=str, default="repeat_0")

parser.add_argument("--model_pickle", type=str, default="pm_model.pickle")

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

parser.add_argument("--sigma_robust", action="store_true", default=False)
parser.add_argument("--bootstrap", type=int, default=None)

parser.add_argument("--random_state", type=int, default=4273)

args = parser.parse_args()

np.random.seed(args.random_state)

def enlarge_sample(sample, enlarge=1):
    sample_new = sample.copy()
    for k in sample_new.keys():
        sample_new[k] = np.repeat(sample_new[k], enlarge)
    return sample_new


experiments = args.experiments.split()
print("experiments:", experiments)

if args.estimator_version == 1:
    bayes_factor = bayes_factor_v1
elif args.estimator_version == 2:
    bayes_factor = bayes_factor_v2
else:
    raise ValueError("Unknown version: %d" % args.estimator_version)

bf_df = []
for exper in experiments:
    print("\n\nCalculating Bayes Factors for " + exper)

    trace_file_2c = os.path.join(args.two_component_mcmc_dir, args.collected_trace_dir, exper+".pickle")
    print("Loading " + trace_file_2c)
    sample_2c = pickle.load(open(trace_file_2c))
    model_2c_file = os.path.join(args.two_component_mcmc_dir, args.model_dir, exper, args.model_pickle)
    print("Loading " + model_2c_file )
    model_2c = pickle.load(open(model_2c_file))

    trace_file_rm = os.path.join(args.racemic_mixture_mcmc_dir, args.collected_trace_dir, exper + ".pickle")
    print("Loading " + trace_file_rm)
    sample_rm = pickle.load(open(trace_file_rm))
    model_rm_file = os.path.join(args.racemic_mixture_mcmc_dir, args.model_dir, exper, args.model_pickle)
    print("Loading " + model_rm_file)
    model_rm = pickle.load(open(model_rm_file))

    trace_file_em = os.path.join(args.enantiomer_mcmc_dir, args.collected_trace_dir, exper + ".pickle")
    print("Loading " + trace_file_em)
    sample_em = pickle.load(open(trace_file_em))
    model_em_file = os.path.join(args.enantiomer_mcmc_dir, args.model_dir, exper, args.model_pickle)
    print("Loading " + model_em_file)
    model_em = pickle.load(open(model_em_file))


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

    if args.bootstrap is not None:
        bf_rm_over_2c, err_rm_over_2c = result_rm_over_2c
        bf_em_over_2c, err_em_over_2c = result_em_over_2c
    else:
        bf_rm_over_2c = result_rm_over_2c
        err_rm_over_2c = None

        bf_em_over_2c = result_em_over_2c
        err_em_over_2c = None

    res_dic = {"Experiment": exper,
               "bf_rm_over_2c": bf_rm_over_2c, "err_rm_over_2c": err_rm_over_2c,
               "bf_em_over_2c": bf_em_over_2c, "err_em_over_2c": err_em_over_2c}
    bf_df.append(res_dic)

    print("------------------------------")

bf_df = pd.DataFrame(bf_df)
num_cols = ["bf_rm_over_2c", "err_rm_over_2c", "bf_em_over_2c", "err_em_over_2c"]
cols = ["Experiment"] + num_cols
bf_df = bf_df[cols]
out = "bayes_factors_ln.csv"
bf_df.to_csv(out, float_format="%0.5f", index=False)

bf_df[num_cols] = bf_df[num_cols] * np.log10(np.e)
out = "bayes_factors_log10.csv"
bf_df.to_csv(out, float_format="%0.5f", index=False)
print("Done")
