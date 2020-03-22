"""
to submit and run mcmc jobs
"""
from __future__ import print_function

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
parser.add_argument("--exper_info_dir", type=str, default="twocomponent_mcmc")
parser.add_argument("--exper_info_file", type=str, default="experimental_information.pickle")

# models "2cbm", "rmbm", "embm"
parser.add_argument("--model", type=str, default="2cbm")

parser.add_argument("--heat_dir", type=str, default="heat_in_origin_format")
parser.add_argument("--heat_file", type=str, default="heat.DAT")

parser.add_argument("--dP0", type=float, default=0.1)      # cell concentration relative uncertainty
parser.add_argument("--dLs", type=float, default=0.1)      # syringe concentration relative uncertainty

parser.add_argument("--uniform_P0", action="store_true", default=False)
parser.add_argument("--uniform_Ls", action="store_true", default=False)
parser.add_argument("--concentration_range_factor", type=float, default=10.)

# Metropolis, HamiltonianMC, NUTS, SMC
parser.add_argument("--step_method", type=str, default="NUTS")
parser.add_argument("--draws", type=int, default=10000)
parser.add_argument("--chains", type=int, default=4)
parser.add_argument("--thin", type=int, default=1)
# Initialization method to use for auto-assigned NUTS samplers.
# "auto", "adapt_diag", "jitter+adapt_diag", "advi+adapt_diag", "advi+adapt_diag_grad", "advi", "advi_map", "map", "nuts"
parser.add_argument("--init", type=str, default="auto")
parser.add_argument("--tune", type=int, default=2000)
parser.add_argument("--cores", type=int, default=1)

parser.add_argument("--experiments", type=str, default=" ")
parser.add_argument("--experiments_flat_prior_P0", type=str, default="")
parser.add_argument("--experiments_flat_prior_Ls", type=str, default="")

parser.add_argument("--out_dir", type=str, default="out")

parser.add_argument("--write_qsub_script", action="store_true", default=False)
parser.add_argument("--submit", action="store_true", default=False)
args = parser.parse_args()

assert args.model in ["2cbm", "rmbm", "embm"], "Unknown model:" + args.model
assert args.step_method in ["Metropolis", "HamiltonianMC", "NUTS", "SMC"], "Unknown step method: " + args.step_method

if args.write_qsub_script:
    assert os.path.exists(args.exper_info_dir), args.exper_info_dir + " does not exist."
    assert os.path.exists(args.heat_dir), args.heat_dir + " does not exist."

    this_script = os.path.abspath(sys.argv[0])
    experiments = args.experiments.split()
    experiments_flat_prior_P0 = args.experiments_flat_prior_P0.split()
    experiments_flat_prior_Ls = args.experiments_flat_prior_Ls.split()

    model = args.model

    dP0 = args.dP0
    dLs = args.dLs

    concentration_range_factor = args.concentration_range_factor

    step_method = args.step_method
    draws = args.draws
    init = args.init
    tune = args.tune
    cores = args.cores
    chains = args.chains

    for experiment in experiments:
        exper_info_file = os.path.join(args.exper_info_dir, experiment, args.exper_info_file)
        heat_file = os.path.join(args.heat_dir, experiment + ".DAT")

        out_dir = experiment
        if not os.path.isdir(out_dir):
            os.makedirs(out_dir)

        out_dir = os.path.abspath(out_dir)

        if experiment in experiments_flat_prior_P0:
            uniform_P0 = " --uniform_P0 "
        else:
            uniform_P0 = " "

        if experiment in experiments_flat_prior_Ls:
            uniform_Ls = " --uniform_Ls "
        else:
            uniform_Ls = " "

        thin = args.thin

        qsub_file = os.path.join(out_dir, experiment + "_mcmc.job")
        log_file = os.path.join(out_dir, experiment + "_mcmc.log")
        qsub_script = '''#!/bin/bash
#PBS -S /bin/bash
#PBS -o %s ''' % log_file + '''
#PBS -j oe
#PBS -l nodes=1:ppn=1,mem=4096mb,walltime=300:00:00

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
        ''' --init ''' + init + \
        ''' --tune %d''' % tune + \
        ''' --cores %d''' % cores + \
        ''' --chains %d''' % chains + \
        ''' --thin %d''' % thin + \
        ''' --out_dir ''' + out_dir + \
        '''\ndate\n'''

        print("Writing qsub file", qsub_file)
        open(qsub_file, "w").write(qsub_script)
        if args.submit:
            print("Submitting " + experiment)
            os.system("qsub %s" % qsub_file)


else:
    exper_info_file = args.exper_info_file
    print("exper_info_file", exper_info_file)
    heat_file = args.heat_file
    print("heat_file", heat_file)

    model_name = args.model
    print("model_name", model_name)

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

    step_method = args.step_method
    print("step_method", step_method)

    draws = args.draws
    print("draws", draws)

    init = args.init
    print("init", init)

    tune = args.tune
    print("tune", tune)

    cores = args.cores
    print("cores", cores)

    chains = args.chains
    print("chains", chains)

    thin = args.thin
    print("thin", thin)

    out_dir = args.out_dir
    print("out_dir", out_dir)

    exper_info = ITCExperiment(exper_info_file)
    q_actual_micro_cal = load_heat_micro_cal(heat_file)
    q_actual_cal = q_actual_micro_cal * 10. ** (-6)

    if model_name == "2cbm":
        pm_model = make_TwoComponentBindingModel(q_actual_cal, exper_info,
                                                 dcell=dcell, dsyringe=dsyringe,
                                                 uniform_P0=uniform_P0, uniform_Ls=uniform_Ls,
                                                 concentration_range_factor=concentration_range_factor,
                                                 auto_transform=True)

    elif model_name == "rmbm":
        pm_model = make_RacemicMixtureBindingModel(q_actual_cal, exper_info,
                                                   dcell=dcell, dsyringe=dsyringe,
                                                   uniform_P0=uniform_P0, uniform_Ls=uniform_Ls,
                                                   concentration_range_factor=concentration_range_factor,
                                                   is_rho_free_param=False,
                                                   auto_transform=True)

    elif model_name == "embm":
        pm_model = make_RacemicMixtureBindingModel(q_actual_cal, exper_info,
                                                   dcell=dcell, dsyringe=dsyringe,
                                                   uniform_P0=uniform_P0, uniform_Ls=uniform_Ls,
                                                   concentration_range_factor=concentration_range_factor,
                                                   is_rho_free_param=True,
                                                   auto_transform=True)
    else:
        raise ValueError("Unknown model: " + model_name)

    with pm_model:
        print("Finding MAP")
        start = pymc3.find_MAP()
        print("MAP:\n", start)

        # Metropolis, HamiltonianMC, NUTS, SMC
        if step_method == "Metropolis":
            step = pymc3.Metropolis()

        elif step_method == "HamiltonianMC":
            step = pymc3.HamiltonianMC()

        elif step_method == "NUTS":
            step = pymc3.NUTS()

        elif step_method == "SMC":
            step = pymc3.SMC()

        else:
            raise ValueError("Unknown step method", step_method)

        print("Running sampling")
        trace = pymc3.sample(draws=draws, init=init, tune=tune,
                             step=step, cores=cores, chains=chains,
                             start=start,
                             progressbar=False)

    trace = trace[::thin]
    out_model = os.path.join(out_dir, "pm_model.pickle")
    pickle.dump(pm_model, open(out_model, "w"))

    out_trace_obj = os.path.join(out_dir, "trace_obj.pickle")
    pickle.dump(trace, open(out_trace_obj, "w"))

    free_vars = [name for name in trace.varnames if not name.endswith("__")]
    trace_vars = {name: trace.get_values(name) for name in free_vars}
    out_trace = os.path.join(out_dir, "traces.pickle")
    pickle.dump(trace_vars, open(out_trace, "w"))

    if step_method == "SMC":
        marg_llh = pm_model.marginal_likelihood
        out_marg_llh = os.path.join(out_dir, "marginal_likelihood.dat")
        open(out_marg_llh, "w").write("%20.10e" % marg_llh)

    plt.figure()
    pymc3.traceplot(trace)
    plt.tight_layout()
    figure_out = os.path.join(out_dir, "trace_plot.pdf")
    plt.savefig(figure_out)

