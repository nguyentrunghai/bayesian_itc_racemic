"""
define function to calculate bayes factor
"""

from __future__ import print_function

import numpy as np
from scipy.stats import norm
from scipy.stats import iqr

import pymbar

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


def get_values_from_trace(model, trace, burn=0):
    varnames = [var.name for var in model.vars]
    trace_values = {var: trace.get_values(var, burn=burn) for var in varnames}
    return trace_values


def log_posterior_trace(model, trace_values):
    model_vars = set([var.name for var in model.vars])
    trace_vars = set(trace_values.keys())
    if model_vars != trace_vars:
        print("model_vars:", model_vars)
        print("trace_vars:", trace_vars)
        raise ValueError("model_vars and trace_vars are not the same set")

    trace_values = dict_to_list(trace_values)
    get_logp = np.vectorize(model.logp)
    logp = get_logp(trace_values)
    return logp


def pot_ener(sample, model):
    """
    :param sample: dict: varname --> ndarray
    :param model: pymc3 model
    :return: ndarray
    """
    u = -log_posterior_trace(model, sample)
    return u


def pot_ener_normal_aug(sample, model, sample_aug, mu_sigma):
    """
    use model to calculate potential energy for sample
    use normal distribution to calculate potential energy for sample_aug
    :param sample: dict: varname --> ndarray
    :param model: pymc3 model
    :param sample_aug: dict: varname --> ndarray
    :param mu_sigma: dict: varname --> dict: {"mu", "sigma"} --> {float, float}
    :return: ndarray
    """
    u1 = -log_posterior_trace(model, sample)
    u2 = -log_normal_trace(sample_aug, mu_sigma)
    u = u1 + u2
    return u


def split_complex_vars(sample_complex, split_type):
    """
    split more complex set of vars to be used for simpler model
    :param sample_complex: dict: varname --> ndarray, samples drawn from more complex model
    :param split_type: str, either or "em_for_2c" or "em_for_rm"
    :return: (sample_main, sample_aug), (dict: varname --> ndarray, dict: varname --> ndarray)
    """
    assert split_type in ["rm_for_2c", "em_for_2c", "em_for_rm"], "Unknown split_type:" + split_type

    if split_type in ["rm_for_2c", "em_for_2c"]:
        common_vars = ["P0_interval__", "Ls_interval__", "DeltaH_0_interval__", "log_sigma_interval__"]
        sample_main = {var: sample_complex[var] for var in common_vars}
        sample_main["DeltaG_interval__"] = sample_complex["DeltaG1_interval__"]
        sample_main["DeltaH_interval__"] = sample_complex["DeltaH1_interval__"]

        if split_type == "rm_for_2c":
            aug_vars = ["DeltaDeltaG_interval__", "DeltaH2_interval__"]
        else:
            aug_vars = ["DeltaDeltaG_interval__", "DeltaH2_interval__", "rho_interval__"]
        sample_aug = {var: sample_complex[var] for var in aug_vars}

        return sample_main, sample_aug

    if split_type == "em_for_rm":
        common_vars = [var for var in sample_complex.keys() if var != "rho_interval__"]
        sample_main = {var: sample_complex[var] for var in common_vars}

        agu_vars = ["rho_interval__"]
        sample_aug = {var: sample_complex[var] for var in agu_vars}

        return sample_main, sample_aug


def augment_simpler_vars(sample_simpler, mu_sigma_complex, aug_type, random_state=None):
    """
    :param sample_simpler: dict: varname --> ndarray, samples drawn from simpler model
    :param mu_sigma_complex: dict: varname --> dict: {"mu", "sigma"} --> {float, float}
                                  mu and sigma estimated from samples drawn from more complex model
    :param aug_type: str, either "2c_for_rm", "2c_for_em", or "rm_for_em"
    :param random_state: int
    :return: (sample_main, sample_aug), (dict: varname --> ndarray, dict: varname --> ndarray)
    """
    assert aug_type in ["2c_for_rm", "2c_for_em", "rm_for_em"], "Unknown aug_type:" + aug_type

    # make sure we get correct mu_sigma_complex
    if aug_type == "2c_for_rm":
        assert "DeltaDeltaG_interval__" in mu_sigma_complex, "DeltaDeltaG_interval__ not in mu_sigma_complex"
        assert "rho_interval__" not in mu_sigma_complex, "rho_interval__ in mu_sigma_complex"

    if aug_type in ["2c_for_em", "rm_for_em"]:
        assert "rho_interval__" in mu_sigma_complex, "rho_interval__ not in mu_sigma_complex"

    nsamples = len(sample_simpler["P0_interval__"])

    if aug_type in ["2c_for_rm", "2c_for_em"]:
        common_vars = ["P0_interval__", "Ls_interval__", "DeltaH_0_interval__", "log_sigma_interval__"]
        sample_main = {var: sample_simpler[var] for var in common_vars}
        sample_main["DeltaG1_interval__"] = sample_simpler["DeltaG_interval__"]
        sample_main["DeltaH1_interval__"] = sample_simpler["DeltaH_interval__"]

        if aug_type == "2c_for_rm":
            aug_vars = ["DeltaDeltaG_interval__", "DeltaH2_interval__"]
        else:
            aug_vars = ["DeltaDeltaG_interval__", "DeltaH2_interval__", "rho_interval__"]

        mu_sigma_aug = {k: mu_sigma_complex[k] for k in aug_vars}
        sample_aug = draw_normal_samples(mu_sigma_aug, nsamples, random_state=random_state)

        return sample_main, sample_aug

    if aug_type == "rm_for_em":
        common_vars = [var for var in sample_simpler.keys() if var != "rho_interval__"]
        sample_main = {var: sample_simpler[var] for var in common_vars}

        aug_vars = ["rho_interval__"]
        mu_sigma_aug = {k: mu_sigma_complex[k] for k in aug_vars}
        sample_aug = draw_normal_samples(mu_sigma_aug, nsamples, random_state=random_state)

        return sample_main, sample_aug


def bootstrap_BAR(w_F, w_R, repeats):
    """
    :param w_F: ndarray
    :param w_R: ndarray
    :param repeats: int
    :return: std, float
    """
    n_F = len(w_F)
    n_R = len(w_R)
    delta_Fs = []
    for _ in range(repeats):
        w_F_rand = np.random.choice(w_F, size=n_F, replace=True)
        w_R_rand = np.random.choice(w_R, size=n_R, replace=True)

        df = pymbar.BAR(w_F_rand, w_R_rand, compute_uncertainty=False, relative_tolerance=1e-6, verbose=False)
        delta_Fs.append(df)

    delta_Fs = np.asarray(delta_Fs)
    delta_Fs = delta_Fs[~np.isnan(delta_Fs)]
    delta_Fs = delta_Fs[~np.isinf(delta_Fs)]

    return delta_Fs.std()


def bayes_factor(model_ini, sample_ini, model_fin, sample_fin,
                 model_ini_name, model_fin_name,
                 sigma_robust=False, random_state=None,
                 bootstrap=None):
    """
    :param model_ini: pymc3 model
    :param sample_ini: dict: varname --> ndarray, samples drawn from initial (simpler) state
    :param model_fin: pymc3 model
    :param sample_fin: dict: varname --> ndarray, samples drawn from final (more complex) state
    :param model_ini_name: str
    :param model_fin_name: str
    :param sigma_robust: bool
    :param random_state: int
    :param bootstrap: int
    :return: float
    """
    mu_sigma_fin = fit_normal_trace(sample_fin, sigma_robust=sigma_robust)

    split_type = model_fin_name + "_for_" + model_ini_name
    aug_type = model_ini_name + "_for_" + model_fin_name

    # augment initial sample
    sample_i_for_f, sample_ini_aug = augment_simpler_vars(sample_ini, mu_sigma_fin, aug_type,
                                                           random_state=random_state)
    # split final sample
    sample_f_for_i, sample_fin_aug = split_complex_vars(sample_fin, split_type)

    # potential for sample drawn from i estimated at state i
    print("Calculate u_i_i: drawn from i, estimated at i")
    u_i_i = pot_ener_normal_aug(sample_ini, model_ini, sample_ini_aug, mu_sigma_fin)

    # potential for sample drawn from i estimated at state f
    sample_ini_comb = sample_i_for_f.copy()
    sample_ini_comb.update(sample_ini_aug)
    print("Calculate u_i_f: drawn from i, estimated at f")
    u_i_f = pot_ener(sample_ini_comb, model_fin)

    #
    # potential for sample drawn from f estimated at state f
    print("Calculate u_f_f: drawn from f, estimated at f")
    u_f_f = pot_ener(sample_fin, model_fin)

    # potential for sample drawn from f estimated at state i
    print("Calculate u_f_i: drawn from f, estimated at i")
    u_f_i = pot_ener_normal_aug(sample_f_for_i, model_ini, sample_fin_aug, mu_sigma_fin)

    w_F = u_i_f - u_i_i
    w_R = u_f_i - u_f_f

    delta_F = pymbar.BAR(w_F, w_R, compute_uncertainty=False, relative_tolerance=1e-12, verbose=True)
    bf = -delta_F

    if bootstrap is None:
        print("ln(bf) = %0.5f" % bf)
        return bf
    else:
        print("Running %d bootstraps to estimate error." % bootstrap)
        bf_err = bootstrap_BAR(w_F, w_R, bootstrap)
        print("ln(bf) = %0.5f +/- %0.5f" % (bf, bf_err))
        return bf, bf_err

