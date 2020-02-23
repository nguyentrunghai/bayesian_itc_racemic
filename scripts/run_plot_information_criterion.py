"""
plot AIC (Akaike information criterion), BIC (Bayesian nformation criterion) and
DIC (deviance information criterion)
"""
from __future__ import print_function

import argparse
import glob
import os
import pickle

import numpy as np
import pandas as pd

from _data_io import load_heat_micro_cal
from _pymc3_models import extract_loglhs_from_traces_pymc3

parser = argparse.ArgumentParser()
parser.add_argument("--heat_data_dir", type=str, default="04.heat_in_origin_format")
parser.add_argument("--exper_info_dir", type=str, default="twocomponent_mcmc")

parser.add_argument("--traces_file", type=str, default="traces.pickle")
parser.add_argument("--exper_info_file", type=str, default="experimental_information.pickle")
parser.add_argument("--extracted_loglhs_file", type=str, default="log_priors_llhs.csv")

parser.add_argument("--two_component_mcmc_dir", type=str, default="/home/tnguye46/bayesian_itc_racemic/07.twocomponent_mcmc/pymc2_2")
parser.add_argument("--racemic_mixture_mcmc_dir", type=str, default="/home/tnguye46/bayesian_itc_racemic/08.racemicmixture_mcmc/pymc2_2")
parser.add_argument("--enantiomer_mcmc_dir", type=str, default="/home/tnguye46/bayesian_itc_racemic/09.enantiomer_mcmc/pymc2_2")

parser.add_argument("--repeat_prefix", type=str, default="repeat_")

parser.add_argument("--experiments", type=str,
                    default="Fokkens_1_a Fokkens_1_b Fokkens_1_c Fokkens_1_d Fokkens_1_e Baum_59 Baum_60_1 Baum_60_4")
parser.add_argument("--experiments_flat_prior_P0", type=str, default="")
parser.add_argument("--experiments_flat_prior_Ls", type=str, default="")

parser.add_argument("--dP0", type=float, default=0.1)      # cell concentration relative uncertainty
parser.add_argument("--dLs", type=float, default=0.1)      # syringe concentration relative uncertainty

parser.add_argument("--concentration_range_factor", type=float, default=10.)

parser.add_argument("--font_scale", type=float, default=0.75)

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


def dic_1(traces, log_llhs,
          model_name, exper_info_file, heat_file,
          dcell=0.1, dsyringe=0.1,
          uniform_P0=False, uniform_Ls=False,
          concentration_range_factor=10.,
          auto_transform=False):
    """
    dic_1 = 2 \ln p(y|\hat{\theta}_{Bayes}) - 2 \frac{1}{S} \sum_{s=1}^S \ln p(y|\theta^s)
    where \hat{\theta}_{Bayes} is mean of the posterior

    :param traces: dict, variable_name -> 1d array
    :param log_llhs: array-like of float, log likelihood values
    :param model_name: str, in ["2cbm", "rmbm", "embm"]
    :param exper_info_file: str
    :param heat_file: str
    :param dcell: float in (0, 1)
    :param dsyringe: float in (0, 1)
    :param uniform_P0: bool
    :param uniform_Ls: bool
    :param concentration_range_factor: float
    :param auto_transform: bool
    :return: llhs, 1d array
    """
    posterior_mean = {var_name: [np.mean(var_trace)] for var_name, var_trace in traces.items()}
    _, log_llh_bayes = extract_loglhs_from_traces_pymc3(posterior_mean, model_name, exper_info_file, heat_file,
                                                        dcell=dcell, dsyringe=dsyringe,
                                                        uniform_P0=uniform_P0, uniform_Ls=uniform_Ls,
                                                        concentration_range_factor=concentration_range_factor,
                                                        auto_transform=auto_transform)
    log_llh_bayes = log_llh_bayes[0]
    return 2 * log_llh_bayes - 2 * np.mean(log_llhs)


def dic_2(traces, log_llhs,
          model_name, exper_info_file, heat_file,
          dcell=0.1, dsyringe=0.1,
          uniform_P0=False, uniform_Ls=False,
          concentration_range_factor=10.,
          auto_transform=False):
    """
    dic_2 = -2 \ln p(y|\hat{\theta}_{Bayes}) + 4 \var \ln p(y|\theta)
    where \hat{\theta}_{Bayes} is mean of the posterior

    :param traces: dict, variable_name -> 1d array
    :param log_llhs: array-like of float, log likelihood values
    :param model_name: str, in ["2cbm", "rmbm", "embm"]
    :param exper_info_file: str
    :param heat_file: str
    :param dcell: float in (0, 1)
    :param dsyringe: float in (0, 1)
    :param uniform_P0: bool
    :param uniform_Ls: bool
    :param concentration_range_factor: float
    :param auto_transform: bool
    :return: llhs, 1d array
    """
    posterior_mean = {var_name: [np.mean(var_trace)] for var_name, var_trace in traces.items()}
    _, log_llh_bayes = extract_loglhs_from_traces_pymc3(posterior_mean, model_name, exper_info_file, heat_file,
                                                        dcell=dcell, dsyringe=dsyringe,
                                                        uniform_P0=uniform_P0, uniform_Ls=uniform_Ls,
                                                        concentration_range_factor=concentration_range_factor,
                                                        auto_transform=auto_transform)
    log_llh_bayes = log_llh_bayes[0]
    return -2 * log_llh_bayes + 4 * np.var(log_llhs)


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

dP0 = args.dP0
dLs = args.dLs
concentration_range_factor = args.concentration_range_factor

models = ["2cbm", "rmbm", "embm"]
list_data_dirs = [two_component_dirs, racemic_mixture_dirs, enantiomer_dirs]

info_criteria = []
for exper in experiments:
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
        aic_s = []
        bic_s = []
        dic_1_s = []
        dic_2_s = []

        for data_dir in data_dirs:
            traces_file = os.path.join(data_dir, exper, args.traces_file)
            loglhs_file = os.path.join(data_dir, exper, args.extracted_loglhs_file)
            exper_info_file = os.path.join(data_dir, exper, args.exper_info_file)
            print("traces_file:", traces_file)
            print("loglhs_file:", loglhs_file)
            print("exper_info_file", exper_info_file)

            traces = pickle.load(open(traces_file))
            log_llhs = pd.read_csv(loglhs_file)["log_lhs"].values
            n_params = len(traces.keys())

            aic_s.append(aic(log_llhs, n_params))
            bic_s.append(bic(log_llhs, n_samples, n_params))

            d1 = dic_1(traces, log_llhs,
                       model, exper_info_file, heat_file,
                       dcell=dP0, dsyringe=dLs,
                       uniform_P0=uniform_P0, uniform_Ls=uniform_Ls,
                       concentration_range_factor=concentration_range_factor,
                       auto_transform=False)

            dic_1_s.append(d1)

            d2 = dic_2(traces, log_llhs,
                       model, exper_info_file, heat_file,
                       dcell=dP0, dsyringe=dLs,
                       uniform_P0=uniform_P0, uniform_Ls=uniform_Ls,
                       concentration_range_factor=concentration_range_factor,
                       auto_transform=False)
            dic_2_s.append(d2)

        info_criteria.append({"exper": exper, "model": model,
                              "aci": np.mean(aic_s), "aci_err": np.std(aic_s),
                              "bci": np.mean(bic_s), "bic_err": np.std(bic_s),
                              "dic_1": np.mean(dic_1_s), "dic_1_err": np.std(dic_1_s),
                              "dic_2": np.mean(dic_2_s), "dic_2_err": np.std(dic_2_s)})

info_criteria = pd.DataFrame(info_criteria)
