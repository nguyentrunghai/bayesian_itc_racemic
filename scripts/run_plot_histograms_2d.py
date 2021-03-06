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

parser.add_argument("--axis_lims_dir", type=str, default="axis_lims")

parser.add_argument("--sample_frac", type=float, default=1.)

parser.add_argument("--font_scale", type=float, default=1.)

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


def conf_interv(x, conf_level=95.):
    alpha = 100 - conf_level
    lower = np.percentile(x, alpha/2)
    upper = np.percentile(x, 100 - (alpha/2))
    return lower, upper


def filter_outliers(df, thres=90):
    for col in df.columns:
        lower, upper = conf_interv(df[col], conf_level=thres)
        df = df[(lower < df[col]) & (df[col] < upper)]
    return df


def pairplot(df, out, figsize):
    plt.figure(figsize=figsize)

    g = sns.PairGrid(df, diag_sharey=False)
    g.map_upper(sns.kdeplot)
    #g.map_upper(sns.scatterplot)
    g.map_lower(sns.kdeplot)
    g.map_diag(sns.kdeplot, lw=2)
    plt.tight_layout()
    plt.savefig(out, dpi=300)
    return None


def read_axis_lims(filename, label_maps):
    lims = []
    with open(filename) as handle:
        for line in handle:
            words = line.split()
            assert len(words) == 6, line + " does not have correct number of entries"
            for i in range(2, 6):
                if words[i].lower() == "none":
                    words[i] = None
                else:
                    words[i] = float(words[i])

            words[1] = label_maps[words[1]]
            lims.append(tuple(words))
    return lims


def pairplot_lims(df, model_abb, lims=None, out="out.pdf", figsize=(10, 10)):

    plt.figure(figsize=figsize)

    g = sns.PairGrid(df, diag_sharey=False)
    g.map_upper(sns.kdeplot)
    # g.map_upper(sns.scatterplot)
    g.map_lower(sns.kdeplot)
    g.map_diag(sns.kdeplot, lw=2)
    axes = g.axes

    if lims is None:
        plt.tight_layout()
        plt.savefig(out, dpi=300)
        return None

    # set lims
    params = list(df.columns)
    param_to_idx = {param: idx for idx, param in enumerate(params)}
    for lim in lims:
        model_abb_lim = lim[0]
        if model_abb_lim == model_abb:
            param = lim[1]
            print("set lims for", model_abb_lim, param, lim)
            idx = param_to_idx[param]
            axes[idx, 0].set_xlim([lim[2], lim[3]])
            axes[0, idx].set_ylim([lim[2], lim[3]])

            axes[idx, 0].set_ylim([lim[4], lim[5]])
            axes[0, idx].set_xlim([lim[4], lim[5]])

    plt.tight_layout()
    plt.savefig(out, dpi=300)
    return None


exclude_repeats = args.exclude_repeats.split()
exclude_repeats = [args.repeat_prefix + r for r in exclude_repeats]
print("exclude_repeats:", exclude_repeats)

experiments = args.experiments.split()
print("experiments", experiments)

colors = ("r", "b", "g")
ylabel = "Probability density"

font_scale = args.font_scale

exclude_vars = ["DeltaH_0", "log_sigma"]

sample_frac = args.sample_frac
print("sample_frac:", sample_frac)

axis_lims_files = glob.glob(os.path.join(args.axis_lims_dir, "*"))
axis_lims_expers = [os.path.basename(f) for f in axis_lims_files]

LABEL_MAPS = {}
LABEL_MAPS["DeltaG"] = "$\Delta G$"
LABEL_MAPS["DeltaH"] = "$\Delta H$"
LABEL_MAPS["Ls"] = "$[L]_s$"
LABEL_MAPS["P0"] = "$[R]_0$"

LABEL_MAPS["DeltaDeltaG"] = "$\Delta \Delta G$"
LABEL_MAPS["DeltaG1"] = "$\Delta G_1$"
LABEL_MAPS["DeltaH1"] = "$\Delta H_1$"
LABEL_MAPS["DeltaH2"] = "$\Delta H_2$"

LABEL_MAPS["rho"] = "rho"

axis_lims = {}
for exper, f in zip(axis_lims_expers, axis_lims_files):
    lims = read_axis_lims(f, LABEL_MAPS)
    axis_lims[exper] = lims

print("axis_lims", axis_lims)

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
    tr_val_2c = tr_val_2c.sample(frac=sample_frac, replace=False)
    tr_val_2c = tr_val_2c.drop(exclude_vars, axis=1)
    tr_val_2c = tr_val_2c.sort_index(axis=1)
    tr_val_2c = filter_outliers(tr_val_2c)

    tr_val_rm = pd.DataFrame(value_from_traces(traces_rm))
    tr_val_rm = tr_val_rm.sample(frac=sample_frac, replace=False)
    #tr_val_rm["DeltaG2"] = tr_val_rm["DeltaG1"] + tr_val_rm["DeltaDeltaG"]
    #tr_val_rm = tr_val_rm.drop(exclude_vars + ["DeltaDeltaG"], axis=1)
    tr_val_rm = tr_val_rm.drop(exclude_vars, axis=1)
    tr_val_rm = tr_val_rm.sort_index(axis=1)
    tr_val_rm = filter_outliers(tr_val_rm)

    tr_val_em = pd.DataFrame(value_from_traces(traces_em))
    tr_val_em = tr_val_em.sample(frac=sample_frac, replace=False)
    #tr_val_em["DeltaG2"] = tr_val_em["DeltaG1"] + tr_val_em["DeltaDeltaG"]
    #tr_val_em["Ls1"] = tr_val_em["Ls"] * tr_val_em["rho"]
    #tr_val_em["Ls2"] = tr_val_em["Ls"] * (1 - tr_val_em["rho"])
    #tr_val_em = tr_val_em.drop(exclude_vars + ["DeltaDeltaG", "Ls"], axis=1)
    tr_val_em = tr_val_em.drop(exclude_vars, axis=1)
    tr_val_em = tr_val_em.sort_index(axis=1)
    tr_val_em = filter_outliers(tr_val_em)

    # rename columns
    tr_val_2c = tr_val_2c.rename(LABEL_MAPS, axis="columns")
    tr_val_rm = tr_val_rm.rename(LABEL_MAPS, axis="columns")
    tr_val_em = tr_val_em.rename(LABEL_MAPS, axis="columns")

    print("tr_val_2c.columns:", tr_val_2c.columns)
    print("tr_val_rm.columns:", tr_val_rm.columns)
    print("tr_val_em.columns:", tr_val_em.columns)

    figsize = (20, 20)

    if exper in axis_lims:
        lims = axis_lims[exper]
    else:
        lims = None

    out = exper + "_2C.pdf"
    print("Plotinng " + out)
    pairplot_lims(tr_val_2c, "2C", lims=lims, out=out, figsize=figsize)

    out = exper + "_RM.pdf"
    print("Plotinng " + out)
    pairplot_lims(tr_val_rm, "RM", lims=lims, out=out, figsize=figsize)

    out = exper + "_EM.pdf"
    print("Plotinng " + out)
    pairplot_lims(tr_val_em, "EM", lims=lims, out=out, figsize=figsize)

print("DONE")



