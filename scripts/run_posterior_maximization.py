"""
run this script to maximize the posterior
"""

from __future__ import print_function

import argparse
import os
import sys
import pickle

import numpy as np

from _data_io import ITCExperiment, load_heat_micro_cal
from _optimization import maximizer
from _optimization import generate_bounds
from _optimization import create_dict_from_optimize_results
from _optimization import plot_heat_actual_vs_model

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="2cbm")
parser.add_argument("--objective", type=str, default="posterior")

parser.add_argument("--heat_dir", type=str, default="4.heat_in_origin_format")
parser.add_argument("--heat_file", type=str, default="heat.DAT")

parser.add_argument("--exper_info_dir", type=str, default="5.exper_info")
parser.add_argument("--exper_info_file", type=str, default="experimental_information.pickle")

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

parser.add_argument("--out_dir", type=str, default="./")

parser.add_argument("--font_scale", type=float, default=1.)

parser.add_argument("--write_qsub_script",   action="store_true", default=False)
parser.add_argument("--submit",   action="store_true", default=False)

args = parser.parse_args()

assert args.model in ["2cbm", "rmbm", "embm"], "unkown model: " + args.model
assert args.objective in ["posterior", "mse"], "unknown objective: " + args.objective

if args.write_qsub_script:
    this_script = os.path.abspath(sys.argv[0])
    experiments = args.experiments.split()

    model = args.model
    objective = args.objective

    DeltaG_bound = '"%s"' % args.DeltaG_bound
    DeltaDeltaG_bound = '"%s"' % args.DeltaDeltaG_bound
    DeltaH_bound = '"%s"' % args.DeltaH_bound
    rho_bound = '"%s"' % args.rho_bound

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

    font_scale = args.font_scale

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

source /home/tnguye46/opt/module/anaconda2019.10.sh
date
python ''' + this_script + \
        ''' --model ''' + model + \
        ''' --objective ''' + objective + \
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
        ''' --font_scale %0.2f ''' % font_scale + \
        ''' --out_dir ''' + out_dir + \
        '''\ndate\n'''

        print("Writing " + qsub_file)
        open(qsub_file, "w").write(qsub_script)

        if args.submit:
            print("Submitting " + experiment)
            os.system("qsub %s" % qsub_file)

else:
    model = args.model
    print("model:", model)

    objective = args.objective
    print("objective:", objective)

    heat_file = args.heat_file
    print("heat_file:", heat_file)

    exper_info_file = args.exper_info_file
    print("exper_info_file:", exper_info_file)

    q_actual_micro_cal = load_heat_micro_cal(heat_file)
    q_actual_cal = q_actual_micro_cal * 10 ** (-6)

    exper_info = ITCExperiment(exper_info_file)

    DeltaG_bound = tuple([np.float(s) for s in args.DeltaG_bound.split()])
    DeltaDeltaG_bound = tuple([np.float(s) for s in args.DeltaDeltaG_bound.split()])
    DeltaH_bound = tuple([np.float(s) for s in args.DeltaH_bound.split()])
    rho_bound = tuple([np.float(s) for s in args.rho_bound.split()])

    dcell = args.dP0
    dsyringe = args.dLs
    uniform_P0 = args.uniform_P0
    uniform_Ls = args.uniform_Ls

    maxiter = args.maxiter
    repeats = args.repeats

    font_scale = args.font_scale

    out_dir = args.out_dir

    bounds = generate_bounds(model, objective, q_actual_cal, exper_info,
                             DeltaG_bound, DeltaDeltaG_bound, DeltaH_bound, rho_bound,
                             dcell=dcell, dsyringe=dsyringe)

    bounds_str = [("%0.5e" % lower, "%0.5e" % upper) for lower, upper in bounds]
    print("Bounds: ", bounds_str)

    results = maximizer(model, objective, q_actual_cal, exper_info,
                        DeltaG_bound, DeltaDeltaG_bound, DeltaH_bound, rho_bound,
                        dcell=dcell, dsyringe=dsyringe,
                        uniform_P0=uniform_P0, uniform_Ls=uniform_P0,
                        maxiter=maxiter, repeats=repeats)

    # write out results
    results_dict = create_dict_from_optimize_results(results, objective)
    print("Lowest function value %0.5e" % results_dict["global"]["fun"])
    print("Global minimizer: ", results_dict["global"]["x"])

    results_out_file = os.path.join(out_dir, "results.pickle")
    pickle.dump(results_dict, open(results_out_file, "w"))

    # plot
    fig_file_name = os.path.join(out_dir, "heat.pdf")
    xlabel = "# injections"
    ylabel = "heat ($\mu$cal)"
    plot_heat_actual_vs_model(q_actual_micro_cal, model, exper_info, results_dict["global"]["x"], fig_file_name,
                              xlabel=xlabel, ylabel=ylabel, font_scale=font_scale)

print("DONE!")
