"""
plot AIC (Akaike information criterion), BIC (Bayesian nformation criterion) and
DIC (deviance information criterion)
"""
from __future__ import print_function

import argparse
import glob
import os
import pickle

import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

import numpy as np
import pandas as pd

from _data_io import load_heat_micro_cal
from _models import extract_loglhs_from_traces_manual

parser = argparse.ArgumentParser()
parser.add_argument("--heat_data_dir", type=str, default="/home/tnguye46/bayesian_itc_racemic/04.heat_in_origin_format")
parser.add_argument("--exper_info_dir", type=str, default="twocomponent_mcmc")

parser.add_argument("--traces_file", type=str, default="traces.pickle")
parser.add_argument("--exper_info_file", type=str, default="experimental_information.pickle")

parser.add_argument("--two_component_mcmc_dir", type=str, default="/home/tnguye46/bayesian_itc_racemic/07.twocomponent_mcmc/pymc2_2")
parser.add_argument("--racemic_mixture_mcmc_dir", type=str, default="/home/tnguye46/bayesian_itc_racemic/08.racemicmixture_mcmc/pymc2_2")
parser.add_argument("--enantiomer_mcmc_dir", type=str, default="/home/tnguye46/bayesian_itc_racemic/09.enantiomer_mcmc/pymc2_2")

parser.add_argument("--repeat_prefix", type=str, default="repeat_")

parser.add_argument("--experiments", type=str,
                    default="Fokkens_1_a Fokkens_1_b Fokkens_1_c Fokkens_1_d Fokkens_1_e Baum_59 Baum_60_1 Baum_60_4")
parser.add_argument("--experiments_flat_prior_P0", type=str, default="")
parser.add_argument("--experiments_flat_prior_Ls", type=str, default="")

parser.add_argument("--font_scale", type=float, default=0.5)

args = parser.parse_args()


def aic(log_llhs, k):
    """
    aic = -2 \ln p(y | \theta_{MLE}) + 2k
    Ref: Andrew Gelman et al., "Bayesian Data Analysis", 3rd Ed., CRC Press, page 169
    :param log_llhs: array-like of float, log likelihood values
    :param k: number of parameters
    :return: float, Akaike information criterion
    """
    return -2 * np.max(log_llhs) + 2 * k


def bic(log_llhs, n, k):
    """
    bic = -2 \ln p(y | \theta_{MLE}) + \ln(n)k
    Ref: Andrew Gelman et al., "Bayesian Data Analysis", 3rd Ed., CRC Press, page 169
    :param log_llhs: array-like of float, log likelihood values
    :param n: int, number of samples
    :param k: number of parameters
    :return: float, Bayesian information criterion
    """
    return -2 * np.max(log_llhs) + np.log(n) * k


def dic_1(traces, log_llhs, model_name, exper_info_file, heat_file):
    """
    D(\theta) = -2 \ln p(y|\theta)
    dic = \overline{D(\theta)} + \overline{D(\theta)} - D(\overline{\theta})
    \overline{D(\theta)} is averge over the poterior
    \overline{\theta} is mean of the posterior

    :param traces: dict, variable_name -> 1d array
    :param log_llhs: array-like of float, log likelihood values
    :param model_name: str, in ["2cbm", "rmbm", "embm"]
    :param exper_info_file: str
    :param heat_file: str
    :return: devian infomation criterion
    """
    devian = -2 * log_llhs
    devian_mean = np.mean(devian)

    posterior_mean = {var_name: [np.mean(var_trace)] for var_name, var_trace in traces.items()}
    log_llh_bayes = extract_loglhs_from_traces_manual(posterior_mean, model_name, exper_info_file, heat_file)
    log_llh_bayes = log_llh_bayes[0]
    return 2 * devian_mean + 2 * log_llh_bayes


def dic_2(log_llhs):
    """
    D(\theta) = -2 \ln p(y|\theta)
    dic = \overline{D(\theta)} + frac{1}{2} \var(D(\theta))
    \overline{D(\theta)} is averge over the poterior
    \var(D(\theta)) is variance over the poterior

    :param log_llhs: array-like of float, log likelihood values
    :return: deviance infomation criterion
    """
    devian = -2 * log_llhs
    devian_mean = np.mean(devian)
    return devian_mean + 0.5 * np.var(log_llhs)

# TODO remove shortened lists below
two_component_dirs = glob.glob(os.path.join(args.two_component_mcmc_dir, args.repeat_prefix + "*"))
two_component_dirs = two_component_dirs[:4]
print("two_component_dirs:", two_component_dirs)

racemic_mixture_dirs = glob.glob(os.path.join(args.racemic_mixture_mcmc_dir, args.repeat_prefix + "*"))
racemic_mixture_dirs = racemic_mixture_dirs[:4]
print("racemic_mixture_dir:", racemic_mixture_dirs)

enantiomer_dirs = glob.glob(os.path.join(args.enantiomer_mcmc_dir, args.repeat_prefix + "*"))
enantiomer_dirs = enantiomer_dirs[:4]
print("enantiomer_dir:", enantiomer_dirs)

experiments = args.experiments.split()
print("experiments", experiments)

experiments_flat_prior_P0 = args.experiments_flat_prior_P0.split()
experiments_flat_prior_Ls = args.experiments_flat_prior_Ls.split()
print("experiments_flat_prior_P0:", experiments_flat_prior_P0)
print("experiments_flat_prior_Ls:", experiments_flat_prior_Ls)

models = ["2cbm", "rmbm", "embm"]
list_data_dirs = [two_component_dirs, racemic_mixture_dirs, enantiomer_dirs]

info_criteria = []
for exper in experiments:
    print("")
    print(exper)

    heat_file = os.path.join(args.heat_data_dir, exper + ".DAT")
    print("heat_file:", heat_file)
    n_samples = load_heat_micro_cal(heat_file).shape[0]
    print("n_samples:", n_samples)

    if exper in experiments_flat_prior_P0:
        uniform_P0 = True
    else:
        uniform_P0 = False

    if exper in experiments_flat_prior_Ls:
        uniform_Ls = True
    else:
        uniform_Ls = False

    for model, data_dirs in zip(models, list_data_dirs):
        print("")
        print("model", model)
        print("list_data_dirs", list_data_dirs)

        for repeat, data_dir in enumerate(data_dirs):
            traces_file = os.path.join(data_dir, exper, args.traces_file)
            exper_info_file = os.path.join(data_dir, exper, args.exper_info_file)
            print("")
            print("traces_file:", traces_file)
            print("exper_info_file", exper_info_file)

            traces = pickle.load(open(traces_file))
            log_llhs = extract_loglhs_from_traces_manual(traces, model, exper_info_file, heat_file)
            n_params = len(traces.keys())

            a = aic(log_llhs, n_params)
            b = bic(log_llhs, n_samples, n_params)

            d1 = dic_1(traces, log_llhs, model, exper_info_file, heat_file)
            d2 = dic_2(log_llhs)

            info_criteria.append({"exper": exper, "model": model, "repeat": repeat,
                                  "aic": a, "bic": b, "dic_1": d1, "dic_2": d2})

info_criteria = pd.DataFrame(info_criteria)
info_criteria.to_csv("info_criteria_post_traces.csv")

# plot
sns.set(font_scale=args.font_scale)

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(3.2, 2.4))
sns.barplot(data=info_criteria, x="aic", y="exper", hue="model", ax=ax)
ax.set_xlabel("AIC")
ax.set_ylabel("Experiment")
fig.tight_layout()
fig.savefig("aic.pdf", dpi=300)


fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(3.2, 2.4))
sns.barplot(data=info_criteria, x="bic", y="exper", hue="model", ax=ax)
ax.set_xlabel("BIC")
ax.set_ylabel("Experiment")
fig.tight_layout()
fig.savefig("bic.pdf", dpi=300)


fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(3.2, 2.4))
sns.barplot(data=info_criteria, x="dic_2", y="exper", hue="model", ax=ax)
ax.set_xlabel("DIC")
ax.set_ylabel("Experiment")
fig.tight_layout()
fig.savefig("dic_2.pdf", dpi=300)

