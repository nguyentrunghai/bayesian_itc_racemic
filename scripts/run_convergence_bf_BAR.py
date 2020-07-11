"""
calculate convergence of Bayes factors using BAR estimator
logp is precomputed
"""

from __future__ import print_function

import argparse
import os
import pickle

import numpy as np

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

parser.add_argument("--sigma_robust", action="store_true", default=False)

parser.add_argument("--random_state", type=int, default=4273)

parser.add_argument("--sample_proportions", type=str, default="0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0")
parser.add_argument("--repeats", type=int, default=10)

args = parser.parse_args()

"""
def take_rnd_sample(sample, proportion):
    keys = sample.keys()
    total_n_samples = len(sample[keys[0]])
    n_samples = int(proportion * total_n_samples)
    n_samples = np.min([n_samples, total_n_samples])

    rnd_idx = np.random.choice(total_n_samples, size=n_samples, replace=True)
    sample_rnd = {key: sample[key][rnd_idx] for key in keys}

    return n_samples, sample_rnd


def bayes_factor_rnd(model_ini, sample_ini, model_fin, sample_fin,
                     estimator_version, sample_proportion, repeats,
                     model_ini_name="2c", model_fin_name="rm",
                     aug_with="Normal", sigma_robust=False,
                     n_components=1, covariance_type="ful"):

    if estimator_version == 1:
        bayes_factor = bayes_factor_v1
    elif estimator_version == 2:
        bayes_factor = bayes_factor_v2
    else:
        raise ValueError("Unknown version: %d" % estimator_version)

    bfs = []
    nsamples_ini, nsamples_fin = 0, 0
    for _ in range(repeats):
        nsamples_ini, sample_ini_rnd = take_rnd_sample(sample_ini, sample_proportion)
        nsamples_fin, sample_fin_rnd = take_rnd_sample(sample_fin, sample_proportion)

        bf = bayes_factor(model_ini, sample_ini_rnd,
                          model_fin, sample_fin_rnd,
                          model_ini_name=model_ini_name, model_fin_name=model_fin_name,
                          aug_with=aug_with, sigma_robust=sigma_robust,
                          n_components=n_components, covariance_type=covariance_type, bootstrap=None)
        bfs.append(bf)

    bf_mean = np.mean(bfs)
    bf_err = np.std(bfs)
    return bf_mean, bf_err
"""

if args.estimator_version == 1:
    bayes_factor = bayes_factor_v1
elif args.estimator_version == 2:
    bayes_factor = bayes_factor_v2
else:
    raise ValueError("Unknown version: %d" % args.estimator_version)

np.random.seed(args.random_state)

experiments = args.experiments.split()
print("experiments:", experiments)

sample_proportions = [float(s) for s in args.sample_proportions.split()]

bf_df = []
for exper in experiments:
    print("\n\nCalculating Bayes Factors for " + exper)

    trace_file_2c = os.path.join(args.two_component_mcmc_dir, args.collected_trace_dir, exper+".pickle")
    print("Loading " + trace_file_2c)
    sample_2c = pickle.load(open(trace_file_2c))
    model_2c_file = os.path.join(args.two_component_mcmc_dir, args.model_dir, exper, args.model_pickle)
    print("Loading " + model_2c_file)
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

    vars_2c = sample_2c.keys()
    vars_rm = sample_rm.keys()
    vars_em = sample_em.keys()

    nsamples_2c = len(sample_2c[vars_2c[0]])
    nsamples_rm = len(sample_rm[vars_rm[0]])
    nsamples_em = len(sample_em[vars_em[0]])

    print("\nRM over 2C")
    out_file_handle = open(exper + "_RM_over_2C.dat", "w")
    out_file_handle.write("# proportion   nsamples_2c    nsamples_rm   bf     bf_err\n")
    for sample_prop in sample_proportions:
        bf, fb_err = bayes_factor(model_2c, sample_2c, model_rm, sample_rm,
                                  model_ini_name="2c", model_fin_name="rm",
                                  aug_with=args.aug_with, sigma_robust=args.sigma_robust,
                                  n_components=args.n_components, covariance_type=args.covariance_type,
                                  bootstrap=args.repeats, sample_proportion=sample_prop)
        bf = bf * np.log10(np.e)
        fb_err = fb_err * np.log10(np.e)

        nsam_ini = nsamples_2c * sample_prop
        nsam_fin = nsamples_rm * sample_prop

        out_file_handle.write("%10.5f %10d %10d     %10.5f %10.5f\n" % (sample_prop, nsam_ini, nsam_fin, bf, fb_err))
    out_file_handle.close()

    print("\nEM over 2C")
    out_file_handle = open(exper + "_EM_over_2C.dat", "w")
    out_file_handle.write("# proportion   nsamples_2c    nsamples_em   bf     bf_err\n")
    

print("DONE")
