"""
calculate Bayes factors using BAR estimator
"""

from __future__ import print_function

import argparse
import os
import pickle

import pandas as pd

from _bayes_factor import get_values_from_trace
from _bayes_factor import bayes_factor

parser = argparse.ArgumentParser()

parser.add_argument("--two_component_mcmc_dir", type=str,
                    default="/home/tnguye46/bayesian_itc_racemic/07.twocomponent_mcmc/pymc3_met_2/repeat_0")
parser.add_argument("--racemic_mixture_mcmc_dir", type=str,
                    default="/home/tnguye46/bayesian_itc_racemic/08.racemicmixture_mcmc/pymc3_met_2/repeat_0")
parser.add_argument("--enantiomer_mcmc_dir", type=str,
                    default="/home/tnguye46/bayesian_itc_racemic/09.enantiomer_mcmc/pymc3_met_2/repeat_0")

parser.add_argument("--model_pickle", type=str, default="pm_model.pickle")
parser.add_argument("--trace_pickle", type=str, default="trace_obj.pickle")

parser.add_argument("--experiments", type=str,
default="Fokkens_1_a Fokkens_1_b Fokkens_1_c Fokkens_1_d Fokkens_1_e Baum_57 Baum_59 Baum_60_1 Baum_60_2 Baum_60_3 Baum_60_4")

parser.add_argument("--burns", type=int, default=0)

parser.add_argument("--sigma_robust", action="store_true", default=False)
parser.add_argument("--random_state", type=int, default=None)
parser.add_argument("--bootstrap", type=int, default=None)

parser.add_argument("--csv_out", type=str, default="bayes_factors.csv")
args = parser.parse_args()

experiments = args.experiments.split()
print("experiments:", experiments)

bf_df = []
for exper in experiments:
    print("Calculating Bayes Factors for " + exper)
    # load data for 2cbm
    model_2c = pickle.load(open(os.path.join(args.two_component_mcmc_dir, exper, args.model_pickle)))
    trace_2c = pickle.load(open(os.path.join(args.two_component_mcmc_dir, exper, args.trace_pickle)))
    sample_2c = get_values_from_trace(model_2c, trace_2c, burn=args.burns)

    # load data for rmbm
    model_rm = pickle.load(open(os.path.join(args.racemic_mixture_mcmc_dir, exper, args.model_pickle)))
    trace_rm = pickle.load(open(os.path.join(args.racemic_mixture_mcmc_dir, exper, args.trace_pickle)))
    sample_rm = get_values_from_trace(model_rm, trace_rm, burn=args.burns)

    # load data for embm
    model_em = pickle.load(open(os.path.join(args.enantiomer_mcmc_dir, exper, args.model_pickle)))
    trace_em = pickle.load(open(os.path.join(args.enantiomer_mcmc_dir, exper, args.trace_pickle)))
    sample_em = get_values_from_trace(model_em, trace_em, burn=args.burns)

    print("RM over 2C")
    result_rm_over_2c = bayes_factor(model_2c, sample_2c, model_rm, sample_rm,
                                     model_ini_name="2c", model_fin_name="rm",
                                     sigma_robust=args.sigma_robust,
                                     random_state=args.random_state,
                                     bootstrap=args.bootstrap)

    print("EM over 2C")
    result_em_over_2c = bayes_factor(model_2c, sample_2c, model_em, sample_em,
                                     model_ini_name="2c", model_fin_name="em",
                                     sigma_robust=args.sigma_robust,
                                     random_state=args.random_state,
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

bf_df = pd.DataFrame(bf_df)
cols = ["Experiment", "bf_rm_over_2c", "err_rm_over_2c", "bf_em_over_2c", "err_em_over_2c"]
bf_df = bf_df[cols]
bf_df.to_csv(args.csv_out, index=False)

print("Done")