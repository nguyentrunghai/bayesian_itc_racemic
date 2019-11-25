"""
To submit and run jobs to extract log priors and likelihoods from traces
"""
from __future__ import print_function

import os
import glob
import sys
import argparse
import pickle

import pandas as pd

from _pymc3_models import extract_loglhs_from_traces_pymc3

parser = argparse.ArgumentParser()
parser.add_argument("--exper_info_dir", type=str, default="twocomponent_mcmc")
parser.add_argument("--exper_info_file", type=str, default="experimental_information.pickle")

# models "2cbm", "rmbm", "embm"
parser.add_argument("--model", type=str, default="2cbm")

parser.add_argument("--heat_dir", type=str, default="heat_in_origin_format")
parser.add_argument("--heat_file", type=str, default="heat.DAT")

parser.add_argument("--traces_file", type=str, default="traces.pickle")

parser.add_argument("--dP0", type=float, default=0.1)      # cell concentration relative uncertainty
parser.add_argument("--dLs", type=float, default=0.1)      # syringe concentration relative uncertainty

parser.add_argument("--uniform_P0", action="store_true", default=False)
parser.add_argument("--uniform_Ls", action="store_true", default=False)
parser.add_argument("--concentration_range_factor", type=float, default=10.)

parser.add_argument("--experiments_unif_conc_prior", type=str, default="Fokkens_1_a Fokkens_1_b")

parser.add_argument("--nsamples", type=int, default=-1)

parser.add_argument("--out_dir", type=str, default="out")
parser.add_argument("--write_qsub_script", action="store_true", default=False)
parser.add_argument("--submit", action="store_true", default=False)
args = parser.parse_args()

assert args.model in ["2cbm", "rmbm", "embm"], "Unknown model:" + args.model

if args.write_qsub_script:
    assert os.path.exists(args.exper_info_dir), args.exper_info_dir + " does not exist."
    assert os.path.exists(args.heat_dir), args.heat_dir + " does not exist."

    this_script = os.path.abspath(sys.argv[0])

    model = args.model

    dP0 = args.dP0
    dLs = args.dLs

    concentration_range_factor = args.concentration_range_factor

    nsamples = args.nsamples

    experiments_unif_conc_prior = args.experiments_unif_conc_prior.split()
    print("xperiments_unif_conc_prior", experiments_unif_conc_prior)

    traces_files = glob.glob(os.path.join("*", args.traces_file))
    print("traces_files", traces_files)

    experiments = [os.path.basename(path) for path in traces_files]
    print("experiments", experiments)

    mcmc_dirs = [os.path.abspath(e) for e in experiments]
    print("mcmc_dirs", mcmc_dirs)

    for mcmc_dir, exper, traces_file in zip(mcmc_dirs, experiments, traces_files):
        print("Working on " + mcmc_dir)
        print("with", exper)
        print("using", traces_file)

        exper_info_file = os.path.join(args.exper_info_dir, exper, args.exper_info_file)
        assert os.path.exists(exper_info_file), exper_info_file + " does not exist."

        heat_file = os.path.join(args.heat_dir, exper + ".DAT")
        assert os.path.exists(heat_file), heat_file + " does not exist."

        out_dir = mcmc_dir

        if exper in experiments_unif_conc_prior:
            uniform_P0 = " --uniform_P0 "
            uniform_Ls = " --uniform_Ls "
        else:
            uniform_P0 = " "
            uniform_Ls = " "

        qsub_file = os.path.join(out_dir, exper + "_logllhs.job")
        log_file = os.path.join(out_dir, exper + "_logllhs.log")
        qsub_script = '''#!/bin/bash
#PBS -S /bin/bash
#PBS -o %s ''' % log_file + '''
#PBS -j oe
#PBS -l nodes=1:ppn=1,walltime=300:00:00

source /home/tnguye46/opt/module/anaconda2019.10.sh
date
python ''' + this_script + \
        ''' --exper_info_file ''' + exper_info_file + \
        ''' --heat_file ''' + heat_file + \
        ''' --traces_file ''' + traces_file + \
        ''' --model ''' + model + \
        ''' --dP0 %0.5f''' % dP0 + \
        ''' --dLs %0.5f''' % dLs + \
        uniform_P0 + uniform_Ls + \
        ''' --concentration_range_factor %0.5f''' % concentration_range_factor + \
        ''' --nsamples %d''' % nsamples + \
        ''' --out_dir ''' + out_dir

        print("Writing qsub file", qsub_file)
        open(qsub_file, "w").write(qsub_script)
        if args.submit:
            print("Submitting " + exper)
            os.system("qsub %s" % qsub_file)

else:
    exper_info_file = args.exper_info_file
    print("exper_info_file", exper_info_file)
    heat_file = args.heat_file
    print("heat_file", heat_file)

    model_name = args.model
    print("model_name", model_name)

    traces_file = args.traces_file
    print("traces_file", traces_file)

    dcell = args.dP0
    print("dcell", dcell)

    dsyringe = args.dLs
    print("dsyringe", dsyringe)

    uniform_P0 = args.uniform_P0
    print("uniform_P0", uniform_P0)

    uniform_Ls = args.uniform_Ls
    print("uniform_Ls", uniform_Ls)

    concentration_range_factor = args.concentration_range_factor
    print("concentration_range_factor", concentration_range_factor)

    nsamples = args.nsamples
    print("nsamples", nsamples)

    our_dir = args.our_dir

    traces = pickle.load(open(traces_file))
    if nsamples > 0:
        for key in traces:
            traces[key] = traces[key][:nsamples]

    log_priors, log_lhs = extract_loglhs_from_traces_pymc3(traces, model_name, exper_info_file, heat_file,
                                                           dcell=dcell, dsyringe=dsyringe,
                                                           uniform_P0=uniform_P0, uniform_Ls=uniform_Ls,
                                                           concentration_range_factor=concentration_range_factor,
                                                           auto_transform=False)

    data_df = pd.DataFrame(data={"log_priors": log_priors, "log_lhs": log_lhs})
    out_file = os.path.join(our_dir, "log_priors_llhs.csv")
    print("Writing results to", out_file)
    data_df.to_csv(out_file, sep=",", float_format="%0.10f", header=True, index=False)

print("DONE")
