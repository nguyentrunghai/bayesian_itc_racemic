"""
calculate convergence of percentiles for important parameters in traces
"""

from __future__ import print_function

import argparse
import os
import pickle

import numpy as np

parser = argparse.ArgumentParser()

parser.add_argument("--mcmc_dir", type=str,
                    default="/home/tnguye46/bayesian_itc_racemic/07.twocomponent_mcmc/pymc3_nuts_2/collected_samples")

parser.add_argument("--experiments", type=str,
default="Fokkens_1_a Fokkens_1_b Fokkens_1_c Fokkens_1_d Fokkens_1_e Baum_57 Baum_59 Baum_60_1 Baum_60_2 Baum_60_3 Baum_60_4")

parser.add_argument("--percentiles", type=str, default="5 25 50 75 95")

parser.add_argument("--vars", type=str, default="DeltaH DeltaG P0 Ls")

parser.add_argument("--sample_proportions", type=str, default="0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0")
parser.add_argument("--repeats", type=int, default=10)

parser.add_argument("--random_state", type=int, default=4273)

args = parser.parse_args()


def percentiles(x, q, nsamples, repeats):
    perce = []
    for _ in range(repeats):
        rnd_x = np.random.choice(x, size=nsamples, replace=True)
        p = np.percentile(rnd_x, q)
        perce.append(p)

    perce = np.array(perce)
    p_mean = perce.mean(axis=0)
    p_err = perce.std(axis=0)

    return p_mean, p_err


def print_percentiles(p_mean, p_err):
    if isinstance(p_mean, float) and isinstance(p_err, float):
        return "%10.5f%10.5f" % (p_mean, p_err)
    else:
        p_str = "".join(["%10.5f%10.5f" % (p_m, p_e) for p_m, p_e in zip(p_mean, p_err)])
        return p_str


np.random.seed(args.random_state)

experiments = args.experiments.split()
print("experiments:", experiments)

qs = [float(s) for s in args.percentiles]
qs_str = "".join(["%10.2f" % q for q in qs])
print("qs:", qs_str)

vars = args.vars.split()
print("vars:", vars)

sample_proportions = [float(s) for s in args.sample_proportions.split()]
print("sample_proportions:", sample_proportions)

for exper in experiments:
    print("\n\nCalculating CIs for " + exper)

    trace_file = os.path.join(args.mcmc_dir, exper+".pickle")
    print("Loading " + trace_file)
    sample = pickle.load(open(trace_file))

    all_vars = sample.keys()
    for v in vars:
        if v not in all_vars:
            raise KeyError(v + " not a valid var name.")

    for var in vars:
        print("var:", var)

        x = sample[var]
        nsamples = len(x)
        out_file_handle = open(exper + "_" + var + ".dat", "w")
        out_file_handle.write("#proportion   nsamples" + qs_str + "\n")

        for samp_pro in sample_proportions:
            nsamp_pro = int(nsamples * samp_pro)
            p_mean, p_err = percentiles(x, qs, nsamp_pro, args.repeats)

            out_str = "%10.5f%10d" % (samp_pro, nsamp_pro) + print_percentiles(p_mean, p_err) + "\n"

            out_file_handle.write(out_str)

        out_file_handle.close()

print("DONE")
