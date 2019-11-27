"""
calculate and plot Bayes factors from extracted log likelihoods
"""

import argparse

import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--two_component_mcmc_dir", type=str, default="/home/tnguye46/bayesian_itc_racemic/07.twocomponent_mcmc/pymc2")
parser.add_argument("--racemic_mixture_mcmc_dir", type=str, default="/home/tnguye46/bayesian_itc_racemic/08.racemicmixture_mcmc/pymc2")
parser.add_argument("--enantiomer_mcmc_dir", type=str, default="/home/tnguye46/bayesian_itc_racemic/09.enantiomer_mcmc/pymc2")

parser.add_argument("--repeat_prefix", type=str, default="repeat_")

parser.add_argument("--exper_info_dir", type=str, default="/home/tnguye46/bayesian_itc_racemic/05.exper_info")
parser.add_argument("--heat_dir", type=str, default="/home/tnguye46/bayesian_itc_racemic/04.heat_in_origin_format")

parser.add_argument("--exper_info_file", type=str, default="experimental_information.pickle")
parser.add_argument("--extracted_loglhs_file", type=str, default="log_priors_llhs.csv")

parser.add_argument("--experiments", type=str,
                    default="Fokkens_1_a Fokkens_1_b Fokkens_1_c Fokkens_1_d Fokkens_1_e Baum_57 Baum_59 Baum_60_1 Baum_60_2 Baum_60_3 Baum_60_4")

parser.add_argument("--bootstrap_repeats", type=int, default=1000)

parser.add_argument("--font_scale", type=float, default=0.75)

args = parser.parse_args()


def _load_combine_dfs(csv_files):
    df_list = [pd.read_csv(f) for f in csv_files]
    comb_df = pd.concat(df_list, axis=0, ignore_index=True)
    return comb_df


