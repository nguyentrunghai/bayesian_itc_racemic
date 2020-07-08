"""
To collect samples and logp from multiples NUTS runs
"""

from __future__ import print_function

import argparse
import os
import glob
import pickle

from _bayes_factor import get_values_from_traces
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

args = parser.parse_args()


def is_path_excluded(path, exclude_kws):
    for kw in exclude_kws:
        if kw in path:
            return True
    return False


def get_values_org_var_from_trace(model, trace, thin=1, burn=0):
    """
    :param model: pymc3 model
    :param trace: pymc3 trace object
    :param thin: int
    :param burn: int, number of steps to exclude
    :return: dict: varname --> ndarray
    """
    varnames = [name for name in model.named_vars.keys() if not name.endswith("__")]

    if isinstance(trace, dict):
        trace_values = {var: trace[var][burn::thin] for var in varnames}
        return trace_values

    trace_values = {var: trace.get_values(var, thin=thin, burn=burn) for var in varnames}
    return trace_values


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

    model = pickle.load(open(model_file))
    trace_list = [pickle.load(open(os.path.join(d, args.trace_pickle))) for d in dirs]
    sample_free_vars = get_values_from_traces(model, trace_list)
    del trace_list

