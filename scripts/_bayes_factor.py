"""
define function to calculate bayes factor
"""

from __future__ import print_function

import numpy as np
from scipy.stats import norm
from scipy.stats import iqr

from sklearn.mixture import GaussianMixture

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


def dict_to_list(dict_of_list):
    """
    :param dict_of_list: dict: varname --> ndarray
    :return: list of dic: [ {varname: float, ...}, ...  ]
    """
    keys = list(dict_of_list.keys())
    key0 = keys[0]
    for key in keys[1:]:
        assert len(dict_of_list[key0]) == len(dict_of_list[key]), key0 + " and " + key + " do not have same len."

    n = len(dict_of_list[key0])
    ls_of_dic = []
    for i in range(n):
        dic = {key: dict_of_list[key][i] for key in keys}
        ls_of_dic.append(dic)
    return ls_of_dic


def get_values_from_trace(model, trace, thin=1, burn=0):
    """
    :param model: pymc3 model
    :param trace: pymc3 trace object
    :param thin: int
    :param burn: int, number of steps to exclude
    :return: dict: varname --> ndarray
    """
    varnames = [var.name for var in model.vars]
    trace_values = {var: trace.get_values(var, thin=thin, burn=burn) for var in varnames}
    return trace_values


def get_values_from_traces(model, traces, thin=1, burn=0):
    trace_value_list = [get_values_from_trace(model, trace, thin=thin, burn=burn) for trace in traces]
    keys = trace_value_list[0].keys()
    trace_values = {}
    for key in keys:
        trace_values[key] = np.concatenate([tr_val[key] for tr_val in trace_value_list])
    return trace_values


def std_from_iqr(data):
    return iqr(data) / 1.35


def fit_normal(x, sigma_robust=False):
    mu, sigma = norm.fit(x)
    if sigma_robust:
        sigma = std_from_iqr(x)
    res = {"mu": mu, "sigma": sigma}
    return res


def fit_normal_trace(trace_values, sigma_robust=False):
    res = {varname: fit_normal(trace_values[varname], sigma_robust=sigma_robust) for varname in trace_values}
    return res


def draw_normal_samples(mu_sigma_dict, nsamples, random_state=None):
    rand = np.random.RandomState(random_state)
    keys = mu_sigma_dict.keys()
    samples = {k: rand.normal(loc=mu_sigma_dict[k]["mu"], scale=mu_sigma_dict[k]["sigma"], size=nsamples)
               for k in keys}
    return samples


def log_normal_pdf(mu, sigma, y):
    sigma2 = sigma * sigma
    res = - 0.5 * np.log(2 * np.pi * sigma2) - (0.5 / sigma2) * (y - mu) ** 2
    return res


def log_normal_trace(trace_val, mu_sigma_dict):
    keys = list(trace_val.keys())
    if len(keys) == 0:
        return 0.

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


def fit_gaussian_mixture(x, n_components=2, covariance_type="spherical"):
    gm = GaussianMixture(n_components=n_components, covariance_type=covariance_type)
    gm.fit(x.reshape([-1, 1]))
    weights = gm.weights_
    means = gm.means_
    covariances = gm.covariances_
    results = []
    for i in range(n_components):
        params = {"weight": weights[i], "mean": means[i][0], "sigma": np.sqrt(covariances[i])}
        results.append(params)
    return results


def fit_uniform(x, d=1e-100):
    lower = x.min() - d
    upper = x.max() + d
    res = {"lower": lower, "upper": upper}
    return res


def fit_uniform_trace(trace_values):
    res = {varname: fit_uniform(trace_values[varname]) for varname in trace_values}
    return res


def draw_uniform_samples(lower_upper_dict, nsamples, random_state=None):
    rand = np.random.RandomState(random_state)
    keys = lower_upper_dict.keys()
    samples = {k: rand.uniform(low=lower_upper_dict[k]["lower"],
                               high=lower_upper_dict[k]["upper"],
                               size=nsamples)
               for k in keys}
    return samples


def log_uniform_pdf(lower, upper, y):
    logp = np.zeros_like(y)
    logp[:] = -np.inf
    logp[(y > lower) & (y < upper)] = - np.log(upper - lower)
    return logp


def log_uniform_trace(trace_val, lower_upper_dict):
    keys = list(trace_val.keys())
    k0 = keys[0]
    for k in keys[1:]:
        assert len(trace_val[k0]) == len(trace_val[k]), k0 + " and " + k + " do not have same len."

    nsamples = len(trace_val[k0])
    logp = np.zeros(nsamples, dtype=float)
    for k in keys:
        lower = lower_upper_dict[k]["lower"]
        upper = lower_upper_dict[k]["upper"]
        y = trace_val[k]
        logp += log_uniform_pdf(lower, upper, y)

    return logp


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
    u = -log_posterior_trace(model, sample)
    return u


def pot_ener_normal_aug(sample, model, sample_aug, mu_sigma):
    u1 = -log_posterior_trace(model, sample)
    u2 = -log_normal_trace(sample_aug, mu_sigma)
    u = u1 + u2
    return u


def pot_ener_uniform_aug(sample, model, sample_aug, lower_upper):
    u1 = -log_posterior_trace(model, sample)
    u2 = -log_uniform_trace(sample_aug, lower_upper)
    u = u1 + u2
    return u


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


def is_var_starts_with(probe_str, str_to_be_probed):
    if str_to_be_probed.endswith("__"):
        suffix = str_to_be_probed.split("__")[0].split("_")[-1]
        suffix = "_" + suffix + "__"
        base_var_name = str_to_be_probed.split(suffix)[0]
        return probe_str == base_var_name
    else:
        return probe_str == str_to_be_probed


def var_starts_with(start_str, list_of_strs):
    """
    :param start_str: str
    :param list_of_strs: list
    :return: str
    """
    found = [s for s in list_of_strs if is_var_starts_with(start_str, s)]
    if len(found) == 0:
        raise ValueError("Found none")
    if len(found) > 1:
        raise ValueError("Found many: " + ", ".join(found))
    return found[0]


def split_complex_vars(sample_complex, vars_simple, split_type):
    """
    split set of more complex vars to be used for simpler model
    :param sample_complex: dict: varname --> ndarray, samples drawn from more complex model
    :param vars_simple: list of str, names of vars in simple model
    :param split_type: str, either or "em_for_2c" or "em_for_rm"
    :return: (sample_main, sample_aug), (dict: varname --> ndarray, dict: varname --> ndarray)
    """
    assert split_type in ["rm_for_2c", "em_for_2c", "em_for_rm"], "Unknown split_type:" + split_type

    vars_complex = sample_complex.keys()

    if split_type in ["rm_for_2c", "em_for_2c"]:
        common_var_prefixes = ["P0", "Ls", "DeltaH_0", "log_sigma"]
        sample_main = {}
        for var_prefix in common_var_prefixes:
            var_s = var_starts_with(var_prefix, vars_simple)
            var_c = var_starts_with(var_prefix, vars_complex)
            sample_main[var_s] = sample_complex[var_c]

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

    vars_simple = sample_simpler.keys()
    vars_complex = mu_sigma_complex.keys()
    nsamples = len(sample_simpler[vars_simple[0]])

    if aug_type in ["2c_for_rm", "2c_for_em"]:
        common_var_prefixes = ["P0", "Ls", "DeltaH_0", "log_sigma"]
        sample_main = {}
        for var_prefix in common_var_prefixes:
            var_s = var_starts_with(var_prefix, vars_simple)
            var_c = var_starts_with(var_prefix, vars_complex)
            sample_main[var_c] = sample_simpler[var_s]

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


def bayes_factor(model_ini, sample_ini, model_fin, sample_fin,
                 model_ini_name="2c", model_fin_name="rm",
                 aug_with="Normal", sigma_robust=False, bootstrap=None):
    """
    :param model_ini:
    :param sample_ini:
    :param model_fin:
    :param sample_fin:
    :param model_ini_name:
    :param model_fin_name:
    :param aug_with:
    :param sigma_robust:
    :param bootstrap:
    :return:
    """
    ini_fin_name = model_ini_name + "_" + model_fin_name
    assert ini_fin_name in ["2c_rm", "2c_em", "rm_em"], "Unknown ini_fin_name: " + ini_fin_name

    lower_upper_fin = fit_uniform_trace(sample_fin)
    mu_sigma_fin = fit_normal_trace(sample_fin, sigma_robust=sigma_robust)

    vars_ini = sample_ini.keys()
    print("vars_ini:", vars_ini)
    vars_fin = sample_fin.keys()
    print("vars_fin:", vars_fin)

    if ini_fin_name == "2c_rm":
        vars_redundant = ["DeltaDeltaG", "DeltaH2"]
    elif ini_fin_name == "2c_em":
        vars_redundant = ["DeltaDeltaG", "DeltaH2", "rho"]
    elif ini_fin_name == "rm_em":
        vars_redundant = ["rho"]
    else:
        raise ValueError("Unknown ini_fin_name: " + ini_fin_name)

    vars_redundant = [var_starts_with(var, vars_fin) for var in vars_redundant]
    print("vars_redundant:", vars_redundant)

    var_match_common = [("P0", "P0"), ("Ls", "Ls"), ("DeltaH_0", "DeltaH_0"), ("log_sigma", "log_sigma")]
    if ini_fin_name in ["2c_rm", "2c_em"]:
        ini_final_var_match = [("DeltaG", "DeltaG1"), ("DeltaH", "DeltaH1")] + var_match_common

    elif ini_fin_name == "rm_em":
        ini_final_var_match = [("DeltaG1", "DeltaG1"), ("DeltaDeltaG", "DeltaDeltaG"),
                               ("DeltaH1", "DeltaH1"), ("DeltaH2", "DeltaH2")] + var_match_common
    else:
        raise ValueError("Unknown ini_fin_name: " + ini_fin_name)
    print("ini_final_var_match:", ini_final_var_match)

    lower_upper_fin = {var: lower_upper_fin[var] for var in vars_redundant}
    mu_sigma_fin = {var: mu_sigma_fin[var] for var in vars_redundant}

    nsamples_ini = len(sample_ini[vars_ini[0]])
    print("nsamples_ini = %d" % nsamples_ini)
    nsamples_fin = len(sample_fin[vars_fin[0]])
    print("nsamples_fin = %d" % nsamples_fin)

    # potential for sample drawn from i estimated at state i
    if aug_with == "Normal":
        sample_aug_ini = draw_normal_samples(mu_sigma_fin, nsamples_ini)
        u_i_i = pot_ener_normal_aug(sample_ini, model_ini, sample_aug_ini, mu_sigma_fin)

    elif aug_with == "Uniform":
        sample_aug_ini = draw_uniform_samples(lower_upper_fin, nsamples_ini)
        u_i_i = pot_ener_uniform_aug(sample_ini, model_ini, sample_aug_ini, lower_upper_fin)

    else:
        raise ValueError("Unknown aug_with:" + aug_with)

    # potential for sample drawn from i estimated at state f
    sample_ini_comb = {}
    for ki, kf in ini_final_var_match:
        var_ini = var_starts_with(ki, vars_ini)
        var_fin = var_starts_with(kf, vars_fin)
        sample_ini_comb[var_fin] = sample_ini[var_ini]
    sample_ini_comb.update(sample_aug_ini)
    u_i_f = pot_ener(sample_ini_comb, model_fin)

    # potential for sample drawn from f estimated at state f
    u_f_f = pot_ener(sample_fin, model_fin)

    # potential for sample drawn from f estimated at state i
    sample_fin_split = {}
    for ki, kf in ini_final_var_match:
        var_ini = var_starts_with(ki, vars_ini)
        var_fin = var_starts_with(kf, vars_fin)
        sample_fin_split[var_ini] = sample_fin[var_fin]

    sample_aug_fin = {var: sample_fin[var] for var in vars_redundant}
    if aug_with == "Normal":
        u_f_i = pot_ener_normal_aug(sample_fin_split, model_ini, sample_aug_fin, mu_sigma_fin)

    elif aug_with == "Uniform":
        u_f_i = pot_ener_uniform_aug(sample_fin_split, model_ini, sample_aug_fin, lower_upper_fin)

    else:
        raise ValueError("Unknown aug_with:" + aug_with)

    w_F = u_i_f - u_i_i
    w_R = u_f_i - u_f_f

    delta_F = pymbar.BAR(w_F, w_R, compute_uncertainty=False, relative_tolerance=1e-12, verbose=True)
    bf = -delta_F

    if bootstrap is None:
        print("log10(bf) = %0.5f" % (bf * np.log10(np.e)))
        return bf
    else:
        print("Running %d bootstraps to estimate error." % bootstrap)
        bf_err = bootstrap_BAR(w_F, w_R, bootstrap)
        print("log10(bf) = %0.5f +/- %0.5f" % (bf * np.log10(np.e), bf_err * np.log10(np.e)))
        return bf, bf_err
