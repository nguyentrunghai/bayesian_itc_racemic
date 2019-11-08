"""
plot comparison between actual heats and model heats
for model heats, we use maximum a posterior estimates of parameters
"""
from __future__ import print_function

import os
import argparse
import pickle

import matplotlib.pyplot as plt
import seaborn as sns

from _data_io import ITCExperiment, load_heat_micro_cal
from _models import heats_TwoComponentBindingModel, heats_RacemicMixtureBindingModel
from _map_estimation import map_TwoComponentBindingModel, map_RacemicMixtureBindingModel, map_EnantiomerBindingModel

parser = argparse.ArgumentParser()

parser.add_argument("--two_component_mcmc_dir", type=str, default="twocomponent_mcmc")
parser.add_argument("--racemic_mixture_mcmc_dir", type=str, default="racemicmixture_mcmc")
parser.add_argument("--enantiomer_mcmc_dir", type=str, default="enantiomer")

parser.add_argument("--exper_info_dir", type=str, default="exper_info")
parser.add_argument("--heat_dir", type=str, default="heat_in_origin_format")

parser.add_argument("--exper_info_file", type=str, default="experimental_information.pickle")
parser.add_argument("--mcmc_trace_file", type=str, default="trace.pkl")

parser.add_argument("--experiments", type=str, default="Fokkens_1_c Fokkens_1_d Fokkens_1_e")
parser.add_argument("--experiments_unif_conc_prior", type=str, default="Fokkens_1_a Fokkens_1_b")

parser.add_argument("--concentration_range_factor", type=float, default=10.)

parser.add_argument("--font_scale", type=float, default=0.75)
parser.add_argument("--xlabel", type=str, default="# injections")
parser.add_argument("--ylabel", type=str, default="heat ($\mu$cal)")

args = parser.parse_args()

KB = 0.0019872041      # in kcal/mol/K

experiments = args.experiments.split()
experiments_unif_conc_prior = args.experiments_unif_conc_prior.split()

sns.set(font_scale=args.font_scale)

for experiment in experiments:
    print(experiment)
    actual_q_micro_cal = load_heat_micro_cal(os.path.join(args.heat_dir, experiment + ".DAT"))
    actual_q_cal = actual_q_micro_cal * 10**(-6)

    exper_info = ITCExperiment(os.path.join(args.exper_info_dir, experiment, args.exper_info_file))

    trace_2cbm = pickle.load(open(os.path.join(args.two_component_mcmc_dir, experiment, args.mcmc_trace_file)))
    trace_rmbm = pickle.load(open(os.path.join(args.racemic_mixture_mcmc_dir, experiment, args.mcmc_trace_file)))
    trace_embm = pickle.load(open(os.path.join(args.enantiomer_mcmc_dir, experiment, args.mcmc_trace_file)))

    if experiment in experiments_unif_conc_prior:
        uniform_P0 = True
        uniform_Ls = True
        print("Uniform prior for concentration")
    else:
        uniform_P0 = False
        uniform_Ls = False
        print("LogNormal prior for concentration")

    (map_P0_2cbm, map_Ls_2cbm, map_DeltaG_2cbm, map_DeltaH_2cbm,
     map_DeltaH_0_2cbm) = map_TwoComponentBindingModel(actual_q_cal, exper_info, trace_2cbm,
                                                       dcell=0.1, dsyringe=0.1,
                                                       uniform_P0=uniform_P0, uniform_Ls=uniform_Ls,
                                                       concentration_range_factor=args.concentration_range_factor)

    (map_P0_rmbm, map_Ls_rmbm, map_DeltaG1_rmbm, map_DeltaDeltaG_rmbm, map_DeltaH1_rmbm, map_DeltaH2_rmbm,
     map_DeltaH_0_rmbm) = map_RacemicMixtureBindingModel(actual_q_cal, exper_info, trace_rmbm,
                                                         dcell=0.1, dsyringe=0.1,
                                                         uniform_P0=uniform_P0, uniform_Ls=uniform_Ls,
                                                         concentration_range_factor=args.concentration_range_factor)

    (map_P0_embm, map_Ls_embm, map_rho_embm, map_DeltaG1_embm, map_DeltaDeltaG_embm, map_DeltaH1_embm, map_DeltaH2_embm,
     map_DeltaH_0_embm) = map_EnantiomerBindingModel(actual_q_cal, exper_info, trace_embm,
                                                     dcell=0.1, dsyringe=0.1,
                                                     uniform_P0=uniform_P0, uniform_Ls=uniform_Ls,
                                                     concentration_range_factor=args.concentration_range_factor)

    # heat calculation using map parameters
    q_2cbm_cal = heats_TwoComponentBindingModel(exper_info.get_cell_volume_liter(),
                                                exper_info.get_injection_volumes_liter(),
                                                map_P0_2cbm, map_Ls_2cbm, map_DeltaG_2cbm, map_DeltaH_2cbm,
                                                map_DeltaH_0_2cbm,
                                                beta=1 / KB / exper_info.get_target_temperature_kelvin(),
                                                N=exper_info.get_number_injections())
    q_2cbm_micro_cal = q_2cbm_cal * 10**6

    q_rmbm_cal = heats_RacemicMixtureBindingModel(exper_info.get_cell_volume_liter(),
                                                  exper_info.get_injection_volumes_liter(),
                                                  map_P0_rmbm, map_Ls_rmbm, 0.5,
                                                  map_DeltaH1_rmbm, map_DeltaH2_rmbm, map_DeltaH_0_rmbm,
                                                  map_DeltaG1_rmbm, map_DeltaDeltaG_rmbm,
                                                  beta=1 / KB / exper_info.get_target_temperature_kelvin(),
                                                  N=exper_info.get_number_injections())
    q_rmbm_micro_cal = q_rmbm_cal * 10**6

    q_embm_cal = heats_RacemicMixtureBindingModel(exper_info.get_cell_volume_liter(),
                                                  exper_info.get_injection_volumes_liter(),
                                                  map_P0_embm, map_Ls_embm, map_rho_embm,
                                                  map_DeltaH1_embm, map_DeltaH2_embm, map_DeltaH_0_embm,
                                                  map_DeltaG1_embm, map_DeltaDeltaG_embm,
                                                  beta=1 / KB / exper_info.get_target_temperature_kelvin(),
                                                  N=exper_info.get_number_injections())
    q_embm_micro_cal = q_embm_cal * 10 ** 6

    print("actual_q_micro_cal:", actual_q_micro_cal)
    print("q_2cbm_micro_cal:", q_2cbm_micro_cal)
    print("q_rmbm_micro_cal:", q_rmbm_micro_cal)
    print("q_embm_micro_cal:", q_embm_micro_cal)

    assert len(actual_q_micro_cal) == len(q_2cbm_micro_cal) == len(q_rmbm_micro_cal) == len(q_embm_micro_cal), "heats do not have the same len"
    n_inj = len(actual_q_micro_cal)

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(3.2, 2.4))
    ax.scatter(range(1, n_inj+1), actual_q_micro_cal, s=20, c="k", marker="o", label="observed")
    ax.plot(range(1, n_inj+1), q_2cbm_micro_cal, c="r", linestyle="-", label="2cbm")
    ax.plot(range(1, n_inj + 1), q_rmbm_micro_cal, c="b", linestyle="-", label="rmbm")
    ax.plot(range(1, n_inj + 1), q_embm_micro_cal, c="g", linestyle="-", label="embm")

    ax.set_xlabel(args.xlabel)
    ax.set_ylabel(args.ylabel)
    ax.legend(loc="lower right")

    fig.tight_layout()
    fig.savefig(experiment + ".pdf", dpi=300)

print("DONE!")
