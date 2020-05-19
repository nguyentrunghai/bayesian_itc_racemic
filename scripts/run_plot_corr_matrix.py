"""
Plot correlation matrix of parameters
"""

from __future__ import print_function

import argparse
import os
import pickle
import glob

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

parser = argparse.ArgumentParser()
parser.add_argument("--two_component_mcmc_dir", type=str,
                    default="/home/tnguye46/bayesian_itc_racemic/07.twocomponent_mcmc/pymc3_nuts_2")
parser.add_argument("--racemic_mixture_mcmc_dir", type=str,
                    default="/home/tnguye46/bayesian_itc_racemic/08.racemicmixture_mcmc/pymc3_nuts_2")
parser.add_argument("--enantiomer_mcmc_dir", type=str,
                    default="/home/tnguye46/bayesian_itc_racemic/09.enantiomer_mcmc/pymc3_nuts_2")

parser.add_argument("--repeat_prefix", type=str, default="repeat_")
parser.add_argument("--exclude_repeats", type=str, default="")

parser.add_argument("--trace_pickle", type=str, default="trace_obj.pickle")

parser.add_argument("--experiments", type=str,
default="Fokkens_1_a Fokkens_1_b Fokkens_1_c Fokkens_1_d Fokkens_1_e Baum_57 Baum_59 Baum_60_1 Baum_60_2 Baum_60_3 Baum_60_4")

parser.add_argument("--font_scale", type=float, default=0.75)

args = parser.parse_args()


def is_path_excluded(path, exclude_kws):
    for kw in exclude_kws:
        if kw in path:
            return True
    return False

def value_from_trace(trace):
    free_vars = [name for name in trace.varnames if not name.endswith("__")]
    tr_val = {name: trace.get_values(name) for name in free_vars}
    return tr_val


def value_from_traces(traces):
    trace_value_list = [value_from_trace(t) for t in traces]
    keys = trace_value_list[0].keys()
    trace_values = {}
    for key in keys:
        trace_values[key] = np.concatenate([tr_val[key] for tr_val in trace_value_list])
    return trace_values


exclude_repeats = args.exclude_repeats.split()
exclude_repeats = [args.repeat_prefix + r for r in exclude_repeats]
print("exclude_repeats:", exclude_repeats)

experiments = args.experiments.split()
print("experiments", experiments)

colors = ("r", "b", "g")
ylabel = "Probability density"

font_scale = args.font_scale

exclude_vars = ["DeltaH_0", "log_sigma"]

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

    traces_2c = [pickle.load(open(os.path.join(d, args.trace_pickle))) for d in dirs_2c]
    traces_rm = [pickle.load(open(os.path.join(d, args.trace_pickle))) for d in dirs_rm]
    traces_em = [pickle.load(open(os.path.join(d, args.trace_pickle))) for d in dirs_em]

    tr_val_2c = pd.DataFrame(value_from_traces(traces_2c))
    tr_val_2c = tr_val_2c.drop(exclude_vars, axis=1)
    tr_val_2c = tr_val_2c.sort_index(axis=1)

    tr_val_rm = pd.DataFrame(value_from_traces(traces_rm))
    tr_val_rm["DeltaG2"] = tr_val_rm["DeltaG1"] + tr_val_rm["DeltaDeltaG"]
    tr_val_rm = tr_val_rm.drop(exclude_vars + ["DeltaDeltaG"], axis=1)
    tr_val_rm = tr_val_rm.sort_index(axis=1)

    tr_val_em = pd.DataFrame(value_from_traces(traces_em))
    tr_val_em["DeltaG2"] = tr_val_em["DeltaG1"] + tr_val_em["DeltaDeltaG"]
    tr_val_em["Ls1"] = tr_val_em["Ls"] * tr_val_em["rho"]
    tr_val_em["Ls2"] = tr_val_em["Ls"] * (1 - tr_val_em["rho"])
    tr_val_em = tr_val_em.drop(exclude_vars + ["DeltaDeltaG", "Ls", "rho"], axis=1)
    tr_val_em = tr_val_em.sort_index(axis=1)

    corr_2c = tr_val_2c.corr()
    print("corr_2c", corr_2c)
    corr_rm = tr_val_rm.corr()
    print("corr_rm", corr_rm)
    corr_em = tr_val_em.corr()
    print("corr_em", corr_em)

    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))
    plt.subplots_adjust(wspace=0.3)
    sns.set(font_scale=font_scale)

    sns.heatmap(corr_2c, annot=True, fmt="0.2f", ax=axes[0])
    axes[0].set_title("Two-Component")

    sns.heatmap(corr_rm, annot=True, fmt="0.2f", ax=axes[1])
    axes[1].set_title("Racemic Mixture")

    sns.heatmap(corr_em, annot=True, fmt="0.2f", ax=axes[2])
    axes[2].set_title("Enantiomer")

    out = exper + ".pdf"
    fig.savefig(out, dpi=300)



