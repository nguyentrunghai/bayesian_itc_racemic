"""
run this script to maximize the posterior
"""

from __future__ import print_function

import argparse
import os
import sys

import numpy as np

from _data_io import ITCExperiment, load_heat_micro_cal
from _optimization import posterior_maximizer
from _optimization import generate_bounds

parser = argparse.ArgumentParser()
parser.add_argument("--mcmc_dir", type=str, default="5.twocomponent_mcmc")
parser.add_argument("--model", type=str, default="2cbm")

parser.add_argument("--heat_dir", type=str, default="4.heat_in_origin_format")
parser.add_argument("--heat_file", type=str, default="/home/tnguye46/bayesian_itc_racemic/4.heat_in_origin_format/Baum_59.DAT")

parser.add_argument("--exper_info_file", type=str, default="/home/tnguye46/bayesian_itc_racemic/5.twocomponent_mcmc/nsamples_5k/Baum_59/experimental_information.pickle")

parser.add_argument("--DeltaG_bound", type=str, default="-20 0")
parser.add_argument("--DeltaDeltaG_bound", type=str, default="0 15")
parser.add_argument("--DeltaH_bound", type=str, default="-40 40")
parser.add_argument("--rho_bound", type=str, default="0.45 0.55")

parser.add_argument("--dP0", type=float, default=0.1)      # cell concentration relative uncertainty
parser.add_argument("--dLs", type=float, default=0.1)      # syringe concentration relative uncertainty

parser.add_argument("--uniform_P0", action="store_true", default=False)
parser.add_argument("--uniform_Ls", action="store_true", default=False)

parser.add_argument("--maxiter", type=int, default=1000)
parser.add_argument("--repeats", type=int, default=10)

parser.add_argument("--experiments", type=str, default="Fokkens_1_c Fokkens_1_d")

parser.add_argument("--write_qsub_script",   action="store_true", default=False)
parser.add_argument("--submit",   action="store_true", default=False)

args = parser.parse_args()

if args.write_qsub_script:
    this_script = os.path.abspath(sys.argv[0])
    experiments = args.experiments.split()

    model = args.model
    DeltaG_bound = args.DeltaG_bound
    DeltaDeltaG_bound = args.DeltaDeltaG_bound
    DeltaH_bound = args.DeltaH_bound
    rho_bound = args.rho_bound

    dP0 = args.dP0
    dLs = args.dLs

    uniform_P0 = " "
    if args.uniform_P0:
        uniform_P0 = " --uniform_P0 "

    uniform_Ls = " "
    if args.uniform_Ls:
        uniform_Ls = " --uniform_Ls "

    maxiter = args.maxiter
    repeats = args.repeats

    for experiment in experiments:
        exper_info_file = os.path.join(args.exper_info_dir, experiment, args.exper_info_file)
        heat_file = os.path.join(args.heat_dir, experiment + ".DAT")

        out_dir = experiment
        if not os.path.isdir(out_dir):
            os.makedirs(out_dir)
        out_dir = os.path.abspath(out_dir)

        qsub_file = os.path.join(out_dir, experiment + "_optimize.job")
        log_file = os.path.join(out_dir, experiment + "_optimize.log")

        qsub_script = '''#!/bin/bash
#PBS -S /bin/bash
#PBS -o %s ''' % log_file + '''
#PBS -j oe
#PBS -l nodes=1:ppn=4,walltime=300:00:00

source /home/tnguye46/opt/module/anaconda.sh
date
python ''' + this_script + \
        ''' --model ''' + model + \
        ''' --exper_info_file ''' + exper_info_file + \
        ''' --heat_file ''' + heat_file + \
        ''' --DeltaG_bound ''' + DeltaG_bound + \
        ''' --DeltaDeltaG_bound ''' + DeltaDeltaG_bound + \
        ''' --DeltaH_bound ''' + DeltaH_bound + \
        ''' --rho_bound ''' + rho_bound + \
        ''' --dP0 %0.5f ''' % dP0 + \
        ''' --dLs %0.5f ''' % dLs + \
        uniform_P0 + uniform_Ls + \
        ''' --maxiter %d ''' % maxiter + \
        ''' --repeats %d ''' % repeats + \
        '''\ndate\n'''

        open(qsub_file, "w").write(qsub_script)

        if args.submit:
            print("Submitting " + experiment)
            os.system("qsub %s" % qsub_file)

else:
    model = args.model
    heat_file = args.heat_file
    exper_info_file = args.exper_info_file

    q_actual_micro_cal = load_heat_micro_cal(heat_file)
    q_actual_cal = q_actual_micro_cal * 10 ** (-6)

    exper_info = ITCExperiment(exper_info_file)

    DeltaG_bound = [np.float(s) for s in args.DeltaG_bound.split()]
    DeltaDeltaG_bound = [np.float(s) for s in args.DeltaDeltaG_bound.split()]
    DeltaH_bound = [np.float(s) for s in args.DeltaH_bound.split()]
    rho_bound = [np.float(s) for s in args.rho_bound.split()]

    dcell = args.dP0
    dsyringe = args.dLs
    uniform_P0 = args.uniform_P0
    uniform_Ls = args.uniform_Ls

    maxiter = args.maxiter
    repeats = args.repeats

    bounds = generate_bounds(model, q_actual_cal, exper_info,
                             DeltaG_bound, DeltaDeltaG_bound, DeltaH_bound, rho_bound,
                             dcell=dcell, dsyringe=dsyringe)
    print("Bounds: ", bounds)

    results = posterior_maximizer(model, q_actual_cal, exper_info,
                                  DeltaG_bound, DeltaDeltaG_bound, DeltaH_bound, rho_bound,
                                  dcell=dcell, dsyringe=dsyringe,
                                  uniform_P0=uniform_P0, uniform_Ls=uniform_P0,
                                  maxiter=maxiter, repeats=repeats)
