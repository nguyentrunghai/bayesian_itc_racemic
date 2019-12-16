"""
plot actual heat versus model heat
MAP estimates were chosen for model heat.
The posterior probabilities were extracted from log priors and log likelihoods
"""


from __future__ import print_function

import argparse
import glob
import os
import pickle

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from _data_io import ITCExperiment, load_heat_micro_cal
from _models import heats_TwoComponentBindingModel, heats_RacemicMixtureBindingModel

parser = argparse.ArgumentParser()
parser.add_argument("--two_component_mcmc_dir", type=str, default="/home/tnguye46/bayesian_itc_racemic/07.twocomponent_mcmc/pymc2_2")
parser.add_argument("--racemic_mixture_mcmc_dir", type=str, default="/home/tnguye46/bayesian_itc_racemic/08.racemicmixture_mcmc/pymc2_2")
parser.add_argument("--enantiomer_mcmc_dir", type=str, default="/home/tnguye46/bayesian_itc_racemic/09.enantiomer_mcmc/pymc2_2")

parser.add_argument("--repeat_prefix", type=str, default="repeat_")

parser.add_argument("--exper_info_dir", type=str, default="/home/tnguye46/bayesian_itc_racemic/05.exper_info")
parser.add_argument("--heat_dir", type=str, default="/home/tnguye46/bayesian_itc_racemic/04.heat_in_origin_format")

parser.add_argument("--exper_info_file", type=str, default="experimental_information.pickle")

parser.add_argument("--extracted_loglhs_file", type=str, default="log_priors_llhs.csv")
parser.add_argument("--mcmc_trace_file", type=str, default="traces.pickle")

parser.add_argument("--experiments", type=str,
                    default="Fokkens_1_a Fokkens_1_b Fokkens_1_c Fokkens_1_d Fokkens_1_e Baum_57 Baum_59 Baum_60_1 Baum_60_2 Baum_60_3 Baum_60_4")

parser.add_argument("--font_scale", type=float, default=0.75)
parser.add_argument("--xlabel", type=str, default="# injections")
parser.add_argument("--ylabel", type=str, default="heat ($\mu$cal)")

args = parser.parse_args()

sns.set(font_scale=args.font_scale)

KB = 0.0019872041      # in kcal/mol/K


def _load_combine_dfs(csv_files):
    df_list = [pd.read_csv(f) for f in csv_files]
    comb_df = pd.concat(df_list, axis=0, ignore_index=True)
    return comb_df


def _load_and_combine_traces(trace_files):
    list_traces = [pickle.load(open(trace_file)) for trace_file in trace_files]
    keys = list_traces[0].keys()
    result = {}
    for key in keys:
        result[key] = np.concatenate([trace[key] for trace in list_traces])
    return result

# TODO: remove [:3] below
two_component_dirs = glob.glob(os.path.join(args.two_component_mcmc_dir, args.repeat_prefix + "*"))
two_component_dirs = two_component_dirs[:3]
print("two_component_dirs:", two_component_dirs)

racemic_mixture_dirs = glob.glob(os.path.join(args.racemic_mixture_mcmc_dir, args.repeat_prefix + "*"))
racemic_mixture_dirs = racemic_mixture_dirs[:3]
print("racemic_mixture_dir:", racemic_mixture_dirs)

enantiomer_dirs = glob.glob(os.path.join(args.enantiomer_mcmc_dir, args.repeat_prefix + "*"))
enantiomer_dirs = enantiomer_dirs[:3]
print("enantiomer_dir:", enantiomer_dirs)

experiments = args.experiments.split()
print("experiments", experiments)

pr_lh_2cbm = {}
pr_lh_rmbm = {}
pr_lh_embm = {}

for exper in experiments:
    print("\n\n", exper)

    # 2cbm
    loglhs_files_2cbm = [os.path.join(d, exper, args.extracted_loglhs_file) for d in two_component_dirs]
    print("loglhs_files_2cbm:\n", loglhs_files_2cbm)
    trace_files_2cbm = [os.path.join(d, exper, args.mcmc_trace_file) for d in two_component_dirs]
    print("trace_files_2cbm:\n", trace_files_2cbm)

    loglhs_2cbm = _load_combine_dfs(loglhs_files_2cbm)
    traces_2cbm = _load_and_combine_traces(trace_files_2cbm)
    log_posterior_2cbm = (loglhs_2cbm["log_lhs"] + loglhs_2cbm["log_priors"]).to_numpy()
    max_idx_2cbm = np.argmax(log_posterior_2cbm)
    print("max_idx_2cbm:", max_idx_2cbm)
    map_2cbm = {param: traces_2cbm[param][max_idx_2cbm] for param in traces_2cbm}
    print("map_2cbm:", map_2cbm)

    # rmbm
    print(" ")
    loglhs_files_rmbm = [os.path.join(d, exper, args.extracted_loglhs_file) for d in racemic_mixture_dirs]
    print("loglhs_files_rmbm:\n", loglhs_files_rmbm)
    trace_files_rmbm = [os.path.join(d, exper, args.mcmc_trace_file) for d in racemic_mixture_dirs]
    print("trace_files_rmbm:\n", trace_files_rmbm)

    loglhs_rmbm = _load_combine_dfs(loglhs_files_rmbm)
    traces_rmbm = _load_and_combine_traces(trace_files_rmbm)
    log_posterior_rmbm = (loglhs_rmbm["log_lhs"] + loglhs_rmbm["log_priors"]).to_numpy()
    max_idx_rmbm = np.argmax(log_posterior_rmbm)
    print("max_idx_rmbm:", max_idx_rmbm)
    map_rmbm = {param: traces_rmbm[param][max_idx_rmbm] for param in traces_rmbm}
    print("map_rmbm:", map_rmbm)

    # embm
    print(" ")
    loglhs_files_embm = [os.path.join(d, exper, args.extracted_loglhs_file) for d in enantiomer_dirs]
    print("loglhs_files_embm:\n", loglhs_files_embm)
    trace_files_embm = [os.path.join(d, exper, args.mcmc_trace_file) for d in enantiomer_dirs]
    print("trace_files_embm:\n", trace_files_embm)

    loglhs_embm = _load_combine_dfs(loglhs_files_embm)
    traces_embm = _load_and_combine_traces(trace_files_embm)
    log_posterior_embm = (loglhs_embm["log_lhs"] + loglhs_embm["log_priors"]).to_numpy()
    max_idx_embm = np.argmax(log_posterior_embm)
    print("max_idx_embm:", max_idx_embm)
    map_embm = {param: traces_embm[param][max_idx_embm] for param in traces_embm}
    print("map_embm:", map_embm)


    actual_q_micro_cal = load_heat_micro_cal(os.path.join(args.heat_dir, exper + ".DAT"))

    exper_info = ITCExperiment(os.path.join(args.exper_info_dir, exper, args.exper_info_file))

    # heat calculation using map parameters
    q_2cbm_cal = heats_TwoComponentBindingModel(exper_info.get_cell_volume_liter(),
                                                exper_info.get_injection_volumes_liter(),
                                                map_2cbm["P0"], map_2cbm["Ls"], map_2cbm["DeltaG"], map_2cbm["DeltaH"],
                                                map_2cbm["DeltaH_0"],
                                                beta=1 / KB / exper_info.get_target_temperature_kelvin(),
                                                N=exper_info.get_number_injections())
    q_2cbm_micro_cal = q_2cbm_cal * 10 ** 6

    q_rmbm_cal = heats_RacemicMixtureBindingModel(exper_info.get_cell_volume_liter(),
                                                  exper_info.get_injection_volumes_liter(),
                                                  map_rmbm["P0"], map_rmbm["Ls"], 0.5,
                                                  map_rmbm["DeltaH1"], map_rmbm["DeltaH2"], map_rmbm["DeltaH_0"],
                                                  map_rmbm["DeltaG1"], map_rmbm["DeltaDeltaG"],
                                                  beta=1 / KB / exper_info.get_target_temperature_kelvin(),
                                                  N=exper_info.get_number_injections())
    q_rmbm_micro_cal = q_rmbm_cal * 10 ** 6

    q_embm_cal = heats_RacemicMixtureBindingModel(exper_info.get_cell_volume_liter(),
                                                  exper_info.get_injection_volumes_liter(),
                                                  map_embm["P0"], map_embm["Ls"], map_embm["rho"],
                                                  map_embm["DeltaH1"], map_embm["DeltaH2"], map_embm["DeltaH_0"],
                                                  map_embm["DeltaG1"], map_embm["DeltaDeltaG"],
                                                  beta=1 / KB / exper_info.get_target_temperature_kelvin(),
                                                  N=exper_info.get_number_injections())
    q_embm_micro_cal = q_embm_cal * 10 ** 6

    print("actual_q_micro_cal:", actual_q_micro_cal)
    print("q_2cbm_micro_cal:", q_2cbm_micro_cal)
    print("q_rmbm_micro_cal:", q_rmbm_micro_cal)
    print("q_embm_micro_cal:", q_embm_micro_cal)

    assert len(actual_q_micro_cal) == len(q_2cbm_micro_cal) == len(q_rmbm_micro_cal) == len(
        q_embm_micro_cal), "heats do not have the same len"
    n_inj = len(actual_q_micro_cal)

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(3.2, 2.4))
    ax.scatter(range(1, n_inj + 1), actual_q_micro_cal, s=20, c="k", marker="o", label="observed")
    ax.plot(range(1, n_inj + 1), q_2cbm_micro_cal, c="r", linestyle="-", label="2cbm")
    ax.plot(range(1, n_inj + 1), q_rmbm_micro_cal, c="b", linestyle="-", label="rmbm")
    ax.plot(range(1, n_inj + 1), q_embm_micro_cal, c="g", linestyle="-", label="embm")

    ax.set_xlabel(args.xlabel)
    ax.set_ylabel(args.ylabel)
    ax.legend(loc="lower right")

    fig.tight_layout()
    fig.savefig(exper + ".pdf", dpi=300)

print("DONE!")