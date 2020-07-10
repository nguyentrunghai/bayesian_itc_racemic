"""
To collect samples and logp from multiples NUTS runs
"""

from __future__ import print_function

import argparse
import os
import glob
import pickle

import numpy as np
import pandas as pd

from _bayes_factor import log_posterior_trace

parser = argparse.ArgumentParser()
parser.add_argument("--mcmc_dir", type=str,
                    default="/home/tnguye46/bayesian_itc_racemic/07.twocomponent_mcmc/pymc3_nuts_2")

parser.add_argument("--repeat_prefix", type=str, default="repeat_")
parser.add_argument("--exclude_repeats", type=str, default="")

parser.add_argument("--model_pickle", type=str, default="pm_model.pickle")
parser.add_argument("--trace_pickle", type=str, default="trace_obj.pickle")

parser.add_argument("--experiments", type=str,
default="Fokkens_1_a Fokkens_1_b Fokkens_1_c Fokkens_1_d Fokkens_1_e Baum_57 Baum_59 Baum_60_1 Baum_60_2 Baum_60_3 Baum_60_4")

parser.add_argument("--burn", type=int, default=0)
parser.add_argument("--thin", type=int, default=1)

parser.add_argument("--logp_shift_file", type=str, default=None)

args = parser.parse_args()


def is_path_excluded(path, exclude_kws):
    for kw in exclude_kws:
        if kw in path:
            return True
    return False


def get_values_from_trace(model, trace, thin=1, burn=0):
    """
    :param model: pymc3 model
    :param trace: pymc3 trace object
    :param thin: int
    :param burn: int, number of steps to exclude
    :return: dict: varname --> ndarray
    """
    varnames = [var.name for var in model.unobserved_RVs]

    if isinstance(trace, dict):
        trace_values = {var: trace[var][burn::thin] for var in varnames}
        return trace_values

    trace_values = {var: trace.get_values(var, thin=thin, burn=burn) for var in varnames}
    return trace_values


def get_values_from_traces(model, traces, thin=1, burn=0):
    trace_value_list = [get_values_from_trace(model, trace, thin=thin, burn=burn) for trace in traces]
    keys = trace_value_list[0].keys()
    trace_values = {}
    for key in keys:
        trace_values[key] = np.concatenate([tr_val[key] for tr_val in trace_value_list])
    return trace_values


def load_logp_shift(csv_file):
    df = pd.read_csv(csv_file)
    df = df.set_index("experiment")
    ser = df["val"]
    return ser


experiments = args.experiments.split()
print("experiments:", experiments)

exclude_repeats = args.exclude_repeats.split()
exclude_repeats = [args.repeat_prefix + r for r in exclude_repeats]
print("exclude_repeats:", exclude_repeats)

for exper in experiments:
    print("\n\nProcessing " + exper)

    dirs = glob.glob(os.path.join(args.mcmc_dir, args.repeat_prefix + "*", exper, args.trace_pickle))
    dirs = [os.path.dirname(p) for p in dirs]
    dirs = [p for p in dirs if not is_path_excluded(p, exclude_repeats)]
    print("dirs:", dirs)

    model_file = os.path.join(dirs[0], args.model_pickle)
    print("Loading model: " + model_file)

    pm_model = pickle.load(open(model_file))
    trace_list = [pickle.load(open(os.path.join(d, args.trace_pickle))) for d in dirs]

    samples = get_values_from_traces(pm_model, trace_list, thin=args.thin, burn=args.burn)
    del trace_list

    logp = log_posterior_trace(pm_model, samples)

    samples["logp"] = logp

    out_file = exper + ".pickle"
    print("Saving " + out_file)
    with open(out_file, "wb") as handle:
        pickle.dump(samples, handle)

print("DONE")
