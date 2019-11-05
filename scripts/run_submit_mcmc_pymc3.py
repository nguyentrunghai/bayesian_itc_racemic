"""
to submit and run mcmc jobs
"""

import os
import sys
import argparse

import pickle

import pymc3
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

from _data_io import ITCExperiment, load_heat_micro_cal
from _pymc3_models import make_TwoComponentBindingModel, make_RacemicMixtureBindingModel

parser = argparse.ArgumentParser()
parser.add_argument("--exper_info_dir", type=str, default="5.twocomponent_mcmc")
parser.add_argument("--exper_info_file", type=str, default="experimental_information.pickle")

# models "2cbm", "rmbm", "embm"
parser.add_argument("--model", type=str, default="2cbm")

parser.add_argument("--heat_dir", type=str, default="4.heat_in_origin_format")
parser.add_argument("--heat_file", type=str, default="heat.DAT")

parser.add_argument("--dP0", type=float, default=0.1)      # cell concentration relative uncertainty
parser.add_argument("--dLs", type=float, default=0.1)      # syringe concentration relative uncertainty

parser.add_argument("--uniform_P0", action="store_true", default=False)
parser.add_argument("--uniform_Ls", action="store_true", default=False)
parser.add_argument("--concentration_range_factor", type=float, default=10.)

# Metropolis, HamiltonianMC, NUTS, SMC
parser.add_argument("--step_method", type=str, default="SMC")
parser.add_argument("--draws", type=int, default=10000)
parser.add_argument("--tune", type=int, default=2000)
parser.add_argument("--cores", type=int, default=1)

parser.add_argument("--experiments", type=str, default="Fokkens_1_c Fokkens_1_d")
parser.add_argument("--experiments_unif_conc_prior", type=str, default="Fokkens_1_a Fokkens_1_b")

parser.add_argument("--out_dir", type=str, default="out")

parser.add_argument("--write_qsub_script", action="store_true", default=False)
parser.add_argument("--submit", action="store_true", default=False)
args = parser.parse_args()

assert args.model in ["2cbm", "rmbm", "embm"], "Unknown model:" + args.model
assert args.step_method in ["Metropolis", "HamiltonianMC", "NUTS", "SMC"], "Unknown step method: " + args.step_method

if args.write_qsub_script:
    this_script = os.path.abspath(sys.argv[0])
    experiments = args.experiments.split()
    experiments_unif_conc_prior = args.experiments_unif_conc_prior.split()

    model = args.model

    dP0 = args.dP0
    dLs = args.dLs

    concentration_range_factor = args.concentration_range_factor

    step_method = args.step_method
    draws = args.draws
    tune = args.tune
    cores = args.cores

    for experiment in experiments:
        exper_info_file = os.path.join(args.exper_info_dir, experiment, args.exper_info_file)
        heat_file = os.path.join(args.heat_dir, experiment + ".DAT")

        out_dir = experiment
        if not os.path.isdir(out_dir):
            os.makedirs(out_dir)

        out_dir = os.path.abspath(out_dir)

        if experiment in experiments_unif_conc_prior:
            uniform_P0 = " --uniform_P0 "
            uniform_Ls = " --uniform_Ls "
        else:
            uniform_P0 = " "
            uniform_Ls = " "

        qsub_file = os.path.join(out_dir, experiment + "_mcmc.job")
        log_file = os.path.join(out_dir, experiment + "_mcmc.log")
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
        ''' --model ''' + model + \
        ''' --dP0 %0.5f''' % dP0 + \
        ''' --dLs %0.5f''' % dLs + \
        uniform_P0 + uniform_Ls + \
        ''' --concentration_range_factor %0.5f''' % concentration_range_factor + \
        ''' --step_method ''' + step_method + \
        ''' --draws %d''' % draws + \
        ''' --tune %d''' % tune + \
        ''' --cores %d''' % cores + \
        ''' --out_dir ''' + out_dir + \
        '''\ndate\n'''

        open(qsub_file, "w").write(qsub_script)
        if args.submit:
            print("Submitting " + experiment)
            os.system("qsub %s" % qsub_file)


else:
    exper_info_file = args.exper_info_file
    print(exper_info_file)
    heat_file = args.heat_file
    print(heat_file)

    model_name = args.model
    print(model_name)

    dcell = args.dP0
    dsyringe = args.dLs
    uniform_P0 = args.uniform_P0
    uniform_Ls = args.uniform_Ls
    concentration_range_factor = args.concentration_range_factor

    draws = args.draws
    tune = args.tune
    cores = args.cores

    out_dir = args.out_dir

    exper_info = ITCExperiment(exper_info_file)
    q_actual_micro_cal = load_heat_micro_cal(heat_file)
    q_actual_cal = q_actual_micro_cal * 10. ** (-6)

    if model_name == "2cbm":
        pm_model = make_TwoComponentBindingModel(q_actual_cal, exper_info,
                                                 dcell=dcell, dsyringe=dsyringe,
                                                 uniform_P0=uniform_P0, uniform_Ls=uniform_Ls,
                                                 concentration_range_factor=concentration_range_factor)

    elif model_name == "rmbm":
        make_RacemicMixtureBindingModel(q_actual_cal, exper_info,
                                        dcell=dcell, dsyringe=dsyringe,
                                        uniform_P0=uniform_P0, uniform_Ls=uniform_Ls,
                                        concentration_range_factor=concentration_range_factor,
                                        is_rho_free_param=False)

    elif model_name == "embm":
        make_RacemicMixtureBindingModel(q_actual_cal, exper_info,
                                        dcell=dcell, dsyringe=dsyringe,
                                        uniform_P0=uniform_P0, uniform_Ls=uniform_Ls,
                                        concentration_range_factor=concentration_range_factor,
                                        is_rho_free_param=True)
    else:
        raise ValueError("Unknown model: " + model_name)
