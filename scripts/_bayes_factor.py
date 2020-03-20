"""
define function to calculate bayes factor
"""

from __future__ import print_function

import numpy as np
from scipy.stats import norm
from scipy.stats import iqr

from _models import log_marginal_likelihood


NAME_MATCH = [("DeltaH", "DeltaH1"), ("DeltaH2", "DeltaH2"), ("DeltaG", "DeltaG1"),
              ("DeltaDeltaG", "DeltaDeltaG"), ("P0", "P0"), ("Ls", "Ls"), ]


def log_marginal_lhs_bootstrap(extracted_loglhs, sample_size=None, bootstrap_repeats=1):
    """
    :param extracted_loglhs: 1d array
    :param sample_size: int
    :param bootstrap_repeats: int
    :return: all_sample_estimate, bootstrap_samples
    """
    if sample_size is None:
        sample_size = len(extracted_loglhs)
    all_sample_estimate = log_marginal_likelihood(extracted_loglhs)

    bootstrap_samples = []
    for _ in range(bootstrap_repeats):
        drawn_loglhs = np.random.choice(extracted_loglhs, size=sample_size, replace=True)

        bootstrap_samples.append(log_marginal_likelihood(drawn_loglhs))

    bootstrap_samples = np.array(bootstrap_samples)

    return all_sample_estimate, bootstrap_samples


def std_from_iqr(data):
    return iqr(data) / 1.35


def fit_normal(x, sigma_robust=False):
    mu, sigma = norm.fit(x)
    if sigma_robust:
        sigma = std_from_iqr(x)
    res = {"mu": mu, "sigma": sigma}
    return res


def fit_normal_trace(trace_values, sigma_robust=False):
    """
    :param trace_values: dict: varname --> ndarray
    :return: dict: varname --> dict: {mu, sigma} -> {float, float}
    """
    res = {varname: fit_normal(trace_values[varname], sigma_robust=sigma_robust) for varname in trace_values}
    return res


def log_normal_pdf(mu, sigma, y):
    """
    :param mu: float
    :param sigma: float
    :param y: float
    :return: float
    """
    sigma2 = sigma * sigma
    res = - 0.5 * np.log(2 * np.pi * sigma2) - (0.5 / sigma2) * (y - mu) ** 2
    return res


def log_normal_trace(trace_val, mu_sigma_dict):
    """
    :param trace_val: dict: varname --> ndarray
    :param mu_sigma_dict: dict: varname --> dict: {"mu", "sigma"} -> {float, float}
    :return: ndarray
    """
    keys = trace_val.keys()
    k0 = keys[0]
    for k in keys[1:]:
        assert len(trace_val[k0]) == len(trace_val[k]), k0 + " and " + k + " do not have same len."

    nsamples = len(trace_val[k0])
    logp = np.zeros(nsamples, dtype=float)
    for k in keys:
        mu = mu_sigma_dict[k]["mu"]
        sigma = mu_sigma_dict[k]["sigma"]
        y = trace_val[k]
        logp += log_normal_pdf(mu, sigma, y)

    return logp


def draw_normal_samples(mu_sigma_dict, nsamples, random_state=None):
    rand = np.random.RandomState(random_state)
    keys = mu_sigma_dict.keys()
    samples = {k: rand.normal(loc=mu_sigma_dict[k]["mu"], scale=mu_sigma_dict[k]["sigma"], size=nsamples)
               for k in keys}
    return samples


def dict_to_list(dict_of_list):
    """
    :param dict_of_list: dict: varname --> ndarray
    :return: list of dic: [ {varname: float, ...}, ...  ]
    """
    keys = dict_of_list.keys()
    key0 = keys[0]
    for key in keys[1:]:
        assert len(dict_of_list[key0]) == len(dict_of_list[key]), key0 + " and " + key + " do not have same len."

    n = len(dict_of_list[key0])
    ls_of_dic = []
    for i in range(n):
        dic = {key: dict_of_list[key][i] for key in keys}
        ls_of_dic.append(dic)
    return ls_of_dic


def get_values_from_trace(model, trace):
    varnames = [var.name for var in model.vars]
    trace_values = {var: trace.get_values(var) for var in varnames}
    return trace_values


def log_posterior_trace(model, trace_values):
    model_vars = set([var.name for var in model.vars])
    trace_vars = set(trace_values.keys())
    print("model_vars:", model_vars)
    print("trace_vars:", trace_vars)
    assert model_vars == trace_vars, "model_vars and trace_vars are not the same set"

    trace_values = dict_to_list(trace_values)
    get_logp = np.vectorize(model.logp)
    logp = get_logp(trace_values)
    return logp


def u_rmbm_rmbm(model_rmbm, tr_val_rmbm):
    """
    calculate potential energy for samples drawn from rmbm using model rmbm
    :param model_rmbm: pymc3 model
    :param tr_val_rmbm: dict: varname --> ndarray
    :return: ndarray
    """
    return - log_posterior_trace(model_rmbm, tr_val_rmbm)


def u_rmbm_2cbm(model_2cbm, tr_val_rmbm, sigma_robust=False):
    """
    calculate potential energy for samples drawn from rmbm using model 2cbm
    :param model_2cbm: pymc3 model
    :param tr_val_rmbm: dict: varname --> ndarray
    :param sigma_robust: bool
    :return: ndarray
    """
    mu_sigma_rmbm = fit_normal_trace(tr_val_rmbm, sigma_robust=sigma_robust)

    # the first element is key for the model, the second is key of trace
    pair_2cbm_rmbm = [("P0_interval__", "P0_interval__"),
                      ("Ls_interval__", "Ls_interval__"),
                      ("DeltaG_interval__", "DeltaG1_interval__"),
                      ("DeltaH_interval__", "DeltaH1_interval__"),
                      ("DeltaH_0_interval__", "DeltaH_0_interval__"),
                      ("log_sigma_interval__", "log_sigma_interval__")]
    print("pair_2cbm_rmbm", pair_2cbm_rmbm)
    redundant_var_rmbm = ["DeltaDeltaG_interval__", "DeltaH2_interval__"]
    print("redundant_var_rmbm", redundant_var_rmbm)

    # tr_val sampled at rmbm, used to estimate logp with model 2cbm
    tr_val_rmbm_4_2cbm = {k0: tr_val_rmbm[k1] for k0, k1 in pair_2cbm_rmbm}
    log_post = log_posterior_trace(model_2cbm, tr_val_rmbm_4_2cbm)

    # tr_val sampled at rmbm, but redundant for 2cbm
    tr_val_rmbm_redun = {k: tr_val_rmbm[k] for k in redundant_var_rmbm}
    logp_norm = log_normal_trace(tr_val_rmbm_redun, mu_sigma_rmbm)
    u = -log_post - logp_norm
    return u


def augment_2cbm_tr_for_rmbm_model(model_rmbm, tr_val_rmbm, tr_val_2cbm, sigma_robust=False):
    """
    trace drawn from model 2cbm has less vars than required by model rmbm.
    So we need to augment by sampling from normal distribution in order to be able to estimate with rmbm.
    :param model_rmbm: pymc3 model
    :param tr_val_rmbm: dict: varname --> ndarray
    :param tr_val_2cbm: dict: varname --> ndarray
    :param sigma_robust: bool
    :return: (tr_val_4_rmbm, aug_tr_val), (dict, dict)
    """
    mu_sigma_rmbm = fit_normal_trace(tr_val_rmbm, sigma_robust=sigma_robust)

    pair_rmbm_2cbm = [("P0_interval__", "P0_interval__"),
                      ("Ls_interval__", "Ls_interval__"),
                      ("DeltaG1_interval__", "DeltaG_interval__"),
                      ("DeltaH1_interval__", "DeltaH_interval__"),
                      ("DeltaH_0_interval__", "DeltaH_0_interval__"),
                      ("log_sigma_interval__", "log_sigma_interval__")]
    aug_vars = ["DeltaDeltaG_interval__", "DeltaH2_interval__"]

    # trace sampled from 2cbm used for rmbm model
    tr_val_4_rmbm = {k0: tr_val_2cbm[k1] for k0, k1 in pair_rmbm_2cbm}


def bfact_rmbm_over_2cbm(model_rmbm, model_2cbm,
                         trace_rmbm, trace_2cbm,
                         sigma_robust=False):
    """
    :param model_rmbm: pymc3 model
    :param model_2cbm: pymc3 model
    :param trace_rmbm: pymc3 trace object
    :param trace_2cbm: pymc3 trace object
    :return: float
    """
    tr_val_rmbm = get_values_from_trace(model_rmbm, trace_rmbm)
    tr_val_2cbm = get_values_from_trace(model_2cbm, trace_2cbm)

    mu_sigma_rmbm = fit_normal_trace(tr_val_rmbm, sigma_robust=sigma_robust)
    mu_sigma_2cbm = fit_normal_trace(tr_val_2cbm, sigma_robust=sigma_robust)

    # u_sample_model
    print("Calculating u_rmbm_rmbm: drawn at rmbm, estimated at rmbm")
    u_rmbm_rmbm = - log_posterior(model_rmbm, tr_val_rmbm)

    print("Calculating u_rmbm_2cbm: drawn at rmbm, estimated at 2cbm")
    pair_2cbm_rmbm = [("P0_interval__", "P0_interval__"),
                      ("Ls_interval__", "Ls_interval__"),
                      ("DeltaG_interval__", "DeltaG1_interval__"),
                      ("DeltaH_interval__", "DeltaH1_interval__"),
                      ("DeltaH_0_interval__", "DeltaH_0_interval__"),
                      ("log_sigma_interval__"), "log_sigma_interval__"]
    print("pair_2cbm_rmbm", pair_2cbm_rmbm)
    redundant_var_rmbm = ["DeltaDeltaG_interval__", "DeltaH2_interval__"]
    print("redundant_var_rmbm", redundant_var_rmbm)
    # tr_val sampled at rmbm, used to estimate logp with model 2cbm
    tr_val_rmbm_4_2cbm = {k1: tr_val_rmbm[k2] for k1, k2 in pair_2cbm_rmbm}
    logp_rmbm_2cbm = log_posterior(model_2cbm, tr_val_rmbm_4_2cbm)
