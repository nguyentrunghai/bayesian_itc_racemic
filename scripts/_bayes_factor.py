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
    print("model_vars:", model_vars)
    print("trace_vars:", trace_vars)
    assert model_vars == trace_vars, "model_vars and trace_vars are not the same set"

    trace_values = dict_to_list(trace_values)
    get_logp = np.vectorize(model.logp)
    logp = get_logp(trace_values)
    return logp


def u_rmbm_rmbm(tr_val_rmbm, model_rmbm):
    """
    calculate potential energy for samples drawn from rmbm using model rmbm
    :param tr_val_rmbm: dict: varname --> ndarray
    :param model_rmbm: pymc3 model
    :return: ndarray
    """
    return - log_posterior_trace(model_rmbm, tr_val_rmbm)


def u_rmbm_2cbm(tr_val_rmbm, model_2cbm, mu_sigma_rmbm):
    """
    calculate potential energy for samples drawn from rmbm using model 2cbm
    :param tr_val_rmbm: dict: varname --> ndarray
    :param model_2cbm: pymc3 model
    :param mu_sigma_rmbm: dict: varname --> dict: {"mu", "sigma"} --> {float, float}
    :return: ndarray
    """
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


def u_2cbm_2cbm(tr_val_2cbm, aug_tr_2cbm, model_2cbm, mu_sigma_rmbm):
    """
    calculate potential energy for samples drawn from 2cbm using model 2cbm
    :param tr_val_2cbm: dict: varname --> ndarray
    :param aug_tr_2cbm: dict: varname --> ndarray
    :param model_2cbm: pymc3 model
    :param mu_sigma_rmbm: dict: varname --> dict: {"mu", "sigma"} --> {float, float}
    :return: ndarray
    """
    u = - log_posterior_trace(model_2cbm, tr_val_2cbm) - log_normal_trace(aug_tr_2cbm, mu_sigma_rmbm)
    return u


def u_2cbm_rmbm(tr_2cbm_4_rmbm, aug_tr_2cbm, model_rmbm):
    """
    calculate potential energy for samples drawn from 2cbm using model rmbm
    :param tr_2cbm_4_rmbm: dict: varname --> ndarray
    :param aug_tr_2cbm: dict: varname --> ndarray
    :param model_rmbm: pymc3 model
    :return: ndarray
    """
    tr_2cbm_4_rmbm.update(aug_tr_2cbm)
    u = - log_posterior_trace(model_rmbm, tr_2cbm_4_rmbm)
    return u


def augment_2cbm_tr_for_rmbm_model(tr_val_2cbm, mu_sigma_rmbm, random_state=None):
    """
    trace drawn from model 2cbm has less vars than required by model rmbm.
    So we need to augment by sampling from normal distribution in order to be able to estimate with rmbm.
    :param tr_val_2cbm: dict: varname --> ndarray
    :param mu_sigma_rmbm: dict: varname --> dict: {"mu", "sigma"} --> {float, float}
    :param random_state: int
    :return: (tr_2cbm_4_rmbm, aug_tr_val), (dict, dict)
    """
    pair_rmbm_2cbm = [("P0_interval__", "P0_interval__"),
                      ("Ls_interval__", "Ls_interval__"),
                      ("DeltaG1_interval__", "DeltaG_interval__"),
                      ("DeltaH1_interval__", "DeltaH_interval__"),
                      ("DeltaH_0_interval__", "DeltaH_0_interval__"),
                      ("log_sigma_interval__", "log_sigma_interval__")]
    aug_vars = ["DeltaDeltaG_interval__", "DeltaH2_interval__"]

    # trace sampled from 2cbm used for rmbm model
    tr_2cbm_4_rmbm = {k0: tr_val_2cbm[k1] for k0, k1 in pair_rmbm_2cbm}

    mu_sigma_aug = {k: mu_sigma_rmbm[k] for k in aug_vars}
    nsamples = len(tr_2cbm_4_rmbm["P0_interval__"])
    aug_tr_2cbm = draw_normal_samples(mu_sigma_aug, nsamples, random_state=random_state)

    return tr_2cbm_4_rmbm, aug_tr_2cbm


def bfact_rmbm_over_2cbm(model_rmbm, model_2cbm,
                         trace_rmbm, trace_2cbm,
                         sigma_robust=False,
                         random_state=None):
    """
    :param model_rmbm: pymc3 model
    :param model_2cbm: pymc3 model
    :param trace_rmbm: pymc3 trace object
    :param trace_2cbm: pymc3 trace object
    :param sigma_robust: bool
    :param random_state: int
    :return: float
    """
    tr_val_rmbm = get_values_from_trace(model_rmbm, trace_rmbm)
    tr_val_2cbm = get_values_from_trace(model_2cbm, trace_2cbm)

    mu_sigma_rmbm = fit_normal_trace(tr_val_rmbm, sigma_robust=sigma_robust)

    u_rm_rm = u_rmbm_rmbm(tr_val_rmbm, model_rmbm)
    u_rm_2c = u_rmbm_2cbm(tr_val_rmbm, model_2cbm, mu_sigma_rmbm)

    tr_2cbm_4_rmbm, aug_tr_2cbm = augment_2cbm_tr_for_rmbm_model(tr_val_2cbm, mu_sigma_rmbm, random_state=random_state)
    u_2c_2c = u_2cbm_2cbm(tr_val_2cbm, aug_tr_2cbm, model_2cbm, mu_sigma_rmbm)
    u_2c_rm = u_2cbm_rmbm(tr_2cbm_4_rmbm, aug_tr_2cbm, model_rmbm)

    w_F = u_rm_2c - u_rm_rm
    w_R = u_2c_rm - u_2c_2c

    delta_f = pymbar.BAR(w_F, w_R, compute_uncertainty=False, relative_tolerance=1e-12, verbose=True)
    return delta_f


#-----------------------


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

        df = pymbar.BAR(w_F_rand, w_R_rand, compute_uncertainty=False, relative_tolerance=1e-12, verbose=False)
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
    sample_ini_main, sample_ini_aug = augment_simpler_vars(sample_ini, mu_sigma_fin, aug_type,
                                                           random_state=random_state)
    # split final sample
    sample_fin_main, sample_fin_aug = split_complex_vars(sample_fin, split_type)

    # potential for sample drawn from i estimated at state i
    u_i_i = pot_ener_normal_aug(sample_ini_main, model_ini, sample_ini_aug, mu_sigma_fin)

    # potential for sample drawn from i estimated at state f
    sample_ini_comb = sample_ini_main.copy()
    sample_ini_comb.update(sample_ini_aug)
    u_i_f = pot_ener(sample_ini_comb, model_fin)

    #
    # potential for sample drawn from f estimated at state f
    sample_fin_comb = sample_fin_main.copy()
    sample_fin_comb.update(sample_fin_aug)
    u_f_f = pot_ener(sample_fin_comb, model_fin)

    # potential for sample drawn from f estimated at state i
    u_f_i = pot_ener_normal_aug(sample_fin_main, model_ini, sample_fin_aug, mu_sigma_fin)

    w_F = u_i_f - u_i_i
    w_R = u_f_i - u_f_f

    delta_F = pymbar.BAR(w_F, w_R, compute_uncertainty=False, relative_tolerance=1e-12, verbose=True)
    bf = -delta_F

    if bootstrap is None:
        return bf
    else:
        bf_err = bootstrap_BAR(w_F, w_R, bootstrap)
        return bf, bf_err

