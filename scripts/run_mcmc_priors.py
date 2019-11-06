"""
run mcmc sampling of priors
"""

from __future__ import print_function

import os
import sys
import argparse
import pickle

from _models import sample_priors
from _data_io import ITCExperiment, load_heat_micro_cal

parser = argparse.ArgumentParser()
parser.add_argument("--exper_info_dir", type=str, default="twocomponent_mcmc")
parser.add_argument("--exper_info_file", type=str, default="experimental_information.pickle")

parser.add_argument("--heat_dir", type=str, default="heat_in_origin_format")
parser.add_argument("--heat_file", type=str, default="heat.DAT")

parser.add_argument("--dP0", type=float, default=0.1)      # cell concentration relative uncertainty
parser.add_argument("--dLs", type=float, default=0.1)      # syringe concentration relative uncertainty

parser.add_argument("--uniform_P0", action="store_true", default=False)
parser.add_argument("--uniform_Ls", action="store_true", default=False)
parser.add_argument("--concentration_range_factor", type=float, default=10.)

parser.add_argument("--nsamples", type=int, default=1000000)
parser.add_argument("--nburn", type=int, default=10000)
parser.add_argument("--nthin", type=int, default=100)

parser.add_argument("--experiments", type=str, default="Fokkens_1_c Fokkens_1_d")

parser.add_argument("--out", type=str, default="traces.pickle")

parser.add_argument("--submit",   action="store_true", default=False)

args = parser.parse_args()

assert os.path.exists(args.exper_info_dir), args.exper_info_dir + " does not exist."
assert os.path.exists(args.heat_dir), args.heat_dir + " does not exist."

if args.submit:
    this_script = os.path.abspath(sys.argv[0])
    experiments = args.experiments.split()

    dP0 = args.dP0
    dLs = args.dLs

    uniform_P0 = " "
    if args.uniform_P0:
        uniform_P0 = " --uniform_P0 "

    uniform_Ls = " "
    if args.uniform_Ls:
        uniform_Ls = " --uniform_Ls "

    concentration_range_factor = args.concentration_range_factor

    nsamples = args.nsamples
    nburn = args.nburn
    nthin = args.nthin

    for experiment in experiments:

        exper_info_file = os.path.join(args.exper_info_dir, experiment, args.exper_info_file)
        heat_file = os.path.join(args.heat_dir, experiment + ".DAT")

        out_dir = experiment
        if not os.path.isdir(out_dir):
            os.makedirs(out_dir)
        out_dir = os.path.abspath(out_dir)

        out = os.path.join(out_dir, args.out)

        qsub_file = os.path.join(out_dir, experiment + "_mcmc.job")
        log_file = os.path.join(out_dir, experiment + "_mcmc.log")
        qsub_script = '''#!/bin/bash
#PBS -S /bin/bash
#PBS -o %s ''' % log_file + '''
#PBS -j oe
#PBS -l nodes=1:ppn=1,walltime=300:00:00

source /home/tnguye46/opt/module/anaconda.sh
date
python ''' + this_script + \
        ''' --exper_info_file ''' + exper_info_file + \
        ''' --heat_file ''' + heat_file + \
        ''' --dP0 %0.5f''' % dP0 + \
        ''' --dLs %0.5f''' % dLs + \
        uniform_P0 + uniform_Ls + \
        ''' --concentration_range_factor %0.5f''' % concentration_range_factor + \
        ''' --nsamples %d''' % nsamples + \
        ''' --nburn %d''' % nburn + \
        ''' --nthin %d''' % nthin + \
        ''' --out ''' + out + \
        '''\ndate\n'''

        open(qsub_file, "w").write(qsub_script)
        print("Submitting " + experiment)
        os.system("qsub %s" % qsub_file)

else:
    exper_info_file = args.exper_info_file
    print("Experimental info file: " + exper_info_file)
    heat_file = args.heat_file
    print("Heat file: " + heat_file)

    exper_info = ITCExperiment(exper_info_file)
    actual_q_micro_cal = load_heat_micro_cal(heat_file)
    actual_q_cal = actual_q_micro_cal * 10**(-6)

    stated_P0 = exper_info.get_cell_concentration_milli_molar()
    print("Stated P0: %0.5f" % stated_P0)
    stated_Ls = exper_info.get_syringe_concentration_milli_molar()
    print("Stated Ls: %0.5f" % stated_Ls)

    dP0 = args.dP0
    dLs = args.dLs

    uniform_P0 = args.uniform_P0
    uniform_Ls = args.uniform_Ls
    concentration_range_factor = args.concentration_range_factor

    nsamples = args.nsamples
    nburn = args.nburn
    nthin = args.nthin

    out = args.out

    all_traces = sample_priors(nsamples, nburn, nthin,
                               actual_q_cal,
                               stated_P0, stated_Ls, dP0=dP0, dLs=dLs,
                               uniform_P0=uniform_P0, uniform_Ls=uniform_Ls,
                               concentration_range_factor=concentration_range_factor)

    pickle.dump(all_traces, open(out, "w"))

    print("DONE")
