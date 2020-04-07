"""
calculate Bayes factors using BAR estimator
"""

from __future__ import print_function

import argparse
import os
import glob
import pickle

import pandas as pd

from _bayes_factor import get_values_from_trace, get_values_from_traces
from _bayes_factor import bayes_factor

parser = argparse.ArgumentParser()

parser.add_argument("--two_component_mcmc_dir", type=str,
                    default="/home/tnguye46/bayesian_itc_racemic/07.twocomponent_mcmc/pymc3_met_2")
parser.add_argument("--racemic_mixture_mcmc_dir", type=str,
                    default="/home/tnguye46/bayesian_itc_racemic/08.racemicmixture_mcmc/pymc3_met_2")
parser.add_argument("--enantiomer_mcmc_dir", type=str,
                    default="/home/tnguye46/bayesian_itc_racemic/09.enantiomer_mcmc/pymc3_met_2")

parser.add_argument("--repeat_prefix", type=str, default="repeat_")

parser.add_argument("--model_pickle", type=str, default="pm_model.pickle")
parser.add_argument("--trace_pickle", type=str, default="trace_obj.pickle")

parser.add_argument("--experiments", type=str,
default="Fokkens_1_a Fokkens_1_b Fokkens_1_c Fokkens_1_d Fokkens_1_e Baum_57 Baum_59 Baum_60_1 Baum_60_2 Baum_60_3 Baum_60_4")

parser.add_argument("--aug_with", type=str, default="Normal")

parser.add_argument("--burn", type=int, default=0)
parser.add_argument("--thin", type=int, default=1)

parser.add_argument("--sigma_robust", action="store_true", default=False)
parser.add_argument("--bootstrap", type=int, default=None)

parser.add_argument("--csv_out", type=str, default="bayes_factors.csv")
args = parser.parse_args()

experiments = args.experiments.split()
print("experiments:", experiments)

bf_df = []
for exper in experiments:
    print("Calculating Bayes Factors for " + exper)

    dirs_2c = glob.glob(os.path.join(args.two_component_mcmc_dir, args.repeat_prefix + "*", exper))
    print("dirs_2c:", dirs_2c)
    dirs_rm = glob.glob(os.path.join(args.racemic_mixture_mcmc_dir, args.repeat_prefix + "*", exper))
    print("dirs_rm:", dirs_rm)
    dirs_em = glob.glob(os.path.join(args.enantiomer_mcmc_dir, args.repeat_prefix + "*", exper))
    print("dirs_em:", dirs_em)

    # load data for 2cbm
    model_2c = pickle.load(open(os.path.join(dirs_2c[0], args.model_pickle)))
    trace_list_2c = [pickle.load(open(os.path.join(d, args.trace_pickle))) for d in dirs_2c]
    sample_2c = get_values_from_traces(model_2c, trace_list_2c, thin=args.thin, burn=args.burns)

    # load data for rmbm
    model_rm = pickle.load(open(os.path.join(dirs_rm[0], args.model_pickle)))
    trace_list_rm = [pickle.load(open(os.path.join(d, args.trace_pickle))) for d in dirs_rm]
    sample_rm = get_values_from_traces(model_rm, trace_list_rm, thin=args.thin, burn=args.burns)

    # load data for embm
    model_em = pickle.load(open(os.path.join(dirs_em[0], args.model_pickle)))
    trace_list_em = [pickle.load(open(os.path.join(d, args.trace_pickle))) for d in dirs_em]
    sample_em = get_values_from_traces(model_em, trace_list_em, thin=args.thin, burn=args.burns)

    print("RM over 2C")
    result_rm_over_2c = bayes_factor(model_2c, sample_2c, model_rm, sample_rm,
                                     model_ini_name="2c", model_fin_name="rm",
                                     aug_with=args.aug_with,
                                     sigma_robust=args.sigma_robust,
                                     bootstrap=args.bootstrap)

    print("EM over 2C")
    result_em_over_2c = bayes_factor(model_2c, sample_2c, model_em, sample_em,
                                     model_ini_name="2c", model_fin_name="em",
                                     aug_with=args.aug_with,
                                     sigma_robust=args.sigma_robust,
                                     bootstrap=args.bootstrap)

    print("EM over RM")
    result_em_over_rm = bayes_factor(model_rm, sample_rm, model_em, sample_em,
                                     model_ini_name="rm", model_fin_name="em",
                                     aug_with=args.aug_with,
                                     sigma_robust=args.sigma_robust,
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

bf_df = pd.DataFrame(bf_df)
cols = ["Experiment", "bf_rm_over_2c", "err_rm_over_2c", "bf_em_over_2c", "err_em_over_2c",
        "bf_em_over_rm", "err_em_over_rm"]
bf_df = bf_df[cols]
bf_df.to_csv(args.csv_out, index=False)

print("Done")
