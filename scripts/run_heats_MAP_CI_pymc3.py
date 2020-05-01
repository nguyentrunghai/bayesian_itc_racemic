"""
Make prediction of heat. MAP and confidence interval
"""

from __future__ import print_function

import argparse
import os
import glob
import pickle

import numpy as np
import pymc3

import matplotlib.pyplot as plt
import seaborn as sns

from _bayes_factor import get_values_from_trace, log_posterior_trace
from _data_io import ITCExperiment, load_heat_micro_cal
from _models import heats_TwoComponentBindingModel, heats_RacemicMixtureBindingModel

parser = argparse.ArgumentParser()

parser.add_argument("--two_component_mcmc_dir", type=str,
                    default="/home/tnguye46/bayesian_itc_racemic/07.twocomponent_mcmc/pymc3_met_2")
parser.add_argument("--racemic_mixture_mcmc_dir", type=str,
                    default="/home/tnguye46/bayesian_itc_racemic/08.racemicmixture_mcmc/pymc3_met_2")
parser.add_argument("--enantiomer_mcmc_dir", type=str,
                    default="/home/tnguye46/bayesian_itc_racemic/09.enantiomer_mcmc/pymc3_met_2")

parser.add_argument("--repeat_prefix", type=str, default="repeat_")
# inclusive
parser.add_argument("--exclude_repeats", type=str, default="")

parser.add_argument("--model_pickle", type=str, default="pm_model.pickle")
parser.add_argument("--trace_pickle", type=str, default="trace_obj.pickle")

parser.add_argument("--exper_info_dir", type=str, default="/home/tnguye46/bayesian_itc_racemic/05.exper_info")
parser.add_argument("--exper_info_file", type=str, default="experimental_information.pickle")

parser.add_argument("--heat_dir", type=str, default="/home/tnguye46/bayesian_itc_racemic/04.heat_in_origin_format")


parser.add_argument("--experiments", type=str,
default="Fokkens_1_a Fokkens_1_b Fokkens_1_c Fokkens_1_d Fokkens_1_e Baum_57 Baum_59 Baum_60_1 Baum_60_2 Baum_60_3 Baum_60_4")

# "optimization" or "mcmc_sampling"
parser.add_argument("--how_to_find_MAP", type=str, default="optimization")

parser.add_argument("--font_scale", type=float, default=0.75)
parser.add_argument("--xlabel", type=str, default="# injections")
parser.add_argument("--ylabel", type=str, default="heat ($\mu$cal)")

args = parser.parse_args()


def is_path_in_excluded_dirs(path, exclude_kws):
    for kw in exclude_kws:
        if kw in path:
            return True
    return False


def find_MAP_maximize(model, method="L-BFGS-B"):
    return pymc3.find_MAP(model=model, method=method)


def find_MAP_trace(model, trace):
    tr_val = get_values_from_trace(model, trace)
    logp = log_posterior_trace(model, tr_val)

    idx_max = np.argmax(logp)
    map_logp = logp[idx_max]

    free_vars = [name for name in trace.varnames if not name.endswith("__")]
    map_val = {name: trace.get_values(name)[idx_max] for name in free_vars}
    return map_val, map_logp


def find_MAP_traces(model, traces):
    map_results = [find_MAP_trace(model, trace) for trace in traces]
    map_results.sort(key=lambda i: i[1])
    map_val = map_results[-1][0]
    return map_val


assert args.how_to_find_MAP in ["optimization", "mcmc_sampling"], "Unknown how_to_find_MAP: " + args.how_to_find_MAP
print("how_to_find_MAP: ", args.how_to_find_MAP)

exclude_repeats = args.exclude_repeats.split()
exclude_repeats = [args.repeat_prefix + r  for r in exclude_repeats]
print("exclude_repeats:", exclude_repeats)

sns.set(font_scale=args.font_scale)

KB = 0.0019872041      # in kcal/mol/K

experiments = args.experiments.split()

heat_data = {}
map_params = {}
for exper in experiments:
    print("\n\n", exper)
    heat_data[exper] = {}
    map_params[exper] = {}

    dirs_2c = glob.glob(os.path.join(args.two_component_mcmc_dir, args.repeat_prefix + "*", exper))
    dirs_2c = [p for p in dirs_2c if is_path_in_repeat_range(p, args.repeat_prefix, repeat_range)]
    print("dirs_2c:", dirs_2c)

    dirs_rm = glob.glob(os.path.join(args.racemic_mixture_mcmc_dir, args.repeat_prefix + "*", exper))
    dirs_rm = [p for p in dirs_rm if is_path_in_repeat_range(p, args.repeat_prefix, repeat_range)]
    print("dirs_rm:", dirs_rm)

    dirs_em = glob.glob(os.path.join(args.enantiomer_mcmc_dir, args.repeat_prefix + "*", exper))
    dirs_em = [p for p in dirs_em if is_path_in_repeat_range(p, args.repeat_prefix, repeat_range)]
    print("dirs_em:", dirs_em)

    model_2c = pickle.load(open(os.path.join(dirs_2c[0], args.model_pickle)))
    model_rm = pickle.load(open(os.path.join(dirs_rm[0], args.model_pickle)))
    model_em = pickle.load(open(os.path.join(dirs_em[0], args.model_pickle)))

    if args.how_to_find_MAP == "optimization":
        print("Optimizing MAP_2C")
        map_2c = find_MAP_maximize(model_2c)
        print("map_2c", map_2c)

        print("Optimizing MAP_RM")
        map_rm = find_MAP_maximize(model_rm)
        print("map_rm", map_rm)

        print("Optimizing MAP_EM")
        map_em = find_MAP_maximize(model_em)
        print("map_em", map_em)

    else:
        traces_2c = [pickle.load(open(os.path.join(d, args.trace_pickle))) for d in dirs_2c]
        traces_rm = [pickle.load(open(os.path.join(d, args.trace_pickle))) for d in dirs_rm]
        traces_em = [pickle.load(open(os.path.join(d, args.trace_pickle))) for d in dirs_em]

        print("Searching for MAP_2C in mcmc trace")
        map_2c = find_MAP_traces(model_2c, traces_2c)
        print("map_2c", map_2c)

        print("Searching for MAP_RM in mcmc trace")
        map_rm = find_MAP_traces(model_rm, traces_rm)
        print("map_rm", map_rm)

        print("Searching for MAP_EM in mcmc trace")
        map_em = find_MAP_traces(model_em, traces_em)
        print("map_em", map_em)

    map_params[exper]["2c"] = map_2c
    map_params[exper]["rm"] = map_rm
    map_params[exper]["em"] = map_em

    exper_info = ITCExperiment(os.path.join(args.exper_info_dir, exper, args.exper_info_file))
    actual_q_micro_cal = load_heat_micro_cal(os.path.join(args.heat_dir, exper + ".DAT"))

    # heat calculation using map parameters
    q_2c_cal = heats_TwoComponentBindingModel(exper_info.get_cell_volume_liter(),
                                              exper_info.get_injection_volumes_liter(),
                                              map_2c["P0"], map_2c["Ls"], map_2c["DeltaG"], map_2c["DeltaH"],
                                              map_2c["DeltaH_0"],
                                              beta=1 / KB / exper_info.get_target_temperature_kelvin(),
                                              N=exper_info.get_number_injections())
    q_2c_micro_cal = q_2c_cal * 10 ** 6

    q_rm_cal = heats_RacemicMixtureBindingModel(exper_info.get_cell_volume_liter(),
                                                exper_info.get_injection_volumes_liter(),
                                                map_rm["P0"], map_rm["Ls"], 0.5,
                                                map_rm["DeltaH1"], map_rm["DeltaH2"], map_rm["DeltaH_0"],
                                                map_rm["DeltaG1"], map_rm["DeltaDeltaG"],
                                                beta=1 / KB / exper_info.get_target_temperature_kelvin(),
                                                N=exper_info.get_number_injections())
    q_rm_micro_cal = q_rm_cal * 10 ** 6

    q_em_cal = heats_RacemicMixtureBindingModel(exper_info.get_cell_volume_liter(),
                                                exper_info.get_injection_volumes_liter(),
                                                map_em["P0"], map_em["Ls"], map_em["rho"],
                                                map_em["DeltaH1"], map_em["DeltaH2"], map_em["DeltaH_0"],
                                                map_em["DeltaG1"], map_em["DeltaDeltaG"],
                                                beta=1 / KB / exper_info.get_target_temperature_kelvin(),
                                                N=exper_info.get_number_injections())
    q_em_micro_cal = q_em_cal * 10 ** 6

    assert len(actual_q_micro_cal) == len(q_2c_micro_cal) == len(q_rm_micro_cal) == len(
        q_em_micro_cal), "heats do not have the same len"
    n_inj = len(actual_q_micro_cal)

    heat_data[exper]["actual_q_micro_cal"] = actual_q_micro_cal
    heat_data[exper]["q_2c_micro_cal"] = q_2c_micro_cal
    heat_data[exper]["q_rm_micro_cal"] = q_rm_micro_cal
    heat_data[exper]["q_em_micro_cal"] = q_em_micro_cal

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(3.2, 2.4))
    ax.scatter(range(1, n_inj + 1), actual_q_micro_cal, s=20, c="k", marker="o", label="observed")
    ax.plot(range(1, n_inj + 1), q_2c_micro_cal, c="r", linestyle="-", label="2cbm")
    ax.plot(range(1, n_inj + 1), q_rm_micro_cal, c="b", linestyle="-", label="rmbm")
    ax.plot(range(1, n_inj + 1), q_em_micro_cal, c="g", linestyle="-", label="embm")

    ax.set_xlabel(args.xlabel)
    ax.set_ylabel(args.ylabel)
    ax.legend(loc="lower right")

    fig.tight_layout()
    fig.savefig(exper + ".pdf", dpi=300)

# pickle data
pickle.dump(heat_data, open("heats.pickle", "wb"))
pickle.dump(map_params, open("map_params.pickle", "wb"))

print("DONE")
