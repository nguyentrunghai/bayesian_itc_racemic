"""
define function to optimize the posterior
"""

import numpy as np
from scipy import optimize

import matplotlib.pyplot as plt
import seaborn as sns

from _models import heats_TwoComponentBindingModel, heats_RacemicMixtureBindingModel
from _models import logsigma_guesses, deltaH0_guesses

KB = 0.0019872041      # in kcal/mol/K


def log_likelihood(q_actual, q_model, sigma):
    """
    :param q_actual: 1d ndarray, actual or observed values of heats
    :param q_model: heat calculated from a model
    :param sigma: standard deviation
    :return: log_likelihood, float

    log_likelihood = -(N/2)\ln(2 \pi \sigma^2) - 1/(2 \sigma^2) \sum_{i=1}^N \epsilon^2
    """
    sum_e_squared = np.sum((q_model - q_actual)**2)

    n_injections = len(q_actual)
    sigma_2 = sigma**2
    log_llh = - n_injections / 2. * np.log(2 * np.pi * sigma_2) - sum_e_squared / 2. / sigma_2

    return log_llh


def log_lognormal(x, stated_center, uncertainty):
    """
    :param x: float
    :param stated_center: float
    :param uncertainty: float
    :return: log_pdf, float
    """
    if x <= 0:
        return 0.

    m = stated_center
    v = uncertainty**2

    mu = np.log(m / np.sqrt(1 + (v / (m ** 2))))
    sigma_2 = np.log(1 + (v / (m**2)))

    pdf = 1 / x / np.sqrt(2 * np.pi * sigma_2) * np.exp(-0.5 / sigma_2 * (np.log(x) - mu)**2)

    return np.log(pdf)


def minus_log_posterior_2cbm(q_actual_cal, exper_info,
                             DeltaG, DeltaH, P0, Ls, DeltaH_0, log_sigma,
                             dcell=0.1, dsyringe=0.1,
                             uniform_P0=False, uniform_Ls=False):
    """
    :param q_actual_cal: observed heats in calorie
    :param exper_info: an object of _data_io.ITCExperiment class
    :param DeltaG: float, free energy of binding (kcal/mol)
    :param DeltaH: float, enthalpy of binding (kcal/mol)
    :param P0: float, Cell concentration (millimolar)
    :param Ls: float, Syringe concentration (millimolar)
    :param DeltaH_0: float, heat of injection (cal)
    :param log_sigma: float, log of sigma, sigma is in cal
    :param dcell: float, relative uncertainty in cell concentration (0 < dcell < 1)
    :param dsyringe: float, relative uncertainty in syringe concentration (0 < dcell < 1)
    :param uniform_P0: bool
    :param uniform_Ls: bool
    :return: minus log posterior
    """

    V0 = exper_info.get_cell_volume_liter()
    DeltaVn = exper_info.get_injection_volumes_liter()
    beta = 1 / KB / exper_info.get_target_temperature_kelvin()
    n_injections = exper_info.get_number_injections()

    q_model_cal = heats_TwoComponentBindingModel(V0, DeltaVn, P0, Ls, DeltaG, DeltaH, DeltaH_0, beta, n_injections)

    sigma_cal = np.exp(log_sigma)

    log_posterior = log_likelihood(q_actual_cal, q_model_cal, sigma_cal)

    if not uniform_P0:
        stated_P0 = exper_info.get_cell_concentration_milli_molar()
        uncertainty_P0 = dcell * stated_P0
        log_posterior += log_lognormal(P0, stated_P0, uncertainty_P0)

    if not uniform_Ls:
        stated_Ls = exper_info.get_syringe_concentration_milli_molar()
        uncertainty_Ls = dsyringe * stated_Ls
        log_posterior += log_lognormal(Ls, stated_Ls, uncertainty_Ls)

    return -log_posterior


def mse_2cbm(q_actual_cal, exper_info,
             DeltaG, DeltaH, P0, Ls, DeltaH_0):
    """
    :param q_actual_cal: observed heats in calorie
    :param exper_info: an object of _data_io.ITCExperiment class
    :param DeltaG: float, free energy of binding (kcal/mol)
    :param DeltaH: float, enthalpy of binding (kcal/mol)
    :param P0: float, Cell concentration (millimolar)
    :param Ls: float, Syringe concentration (millimolar)
    :param DeltaH_0: float, heat of injection (cal)
    :return: mse, float
    """

    V0 = exper_info.get_cell_volume_liter()
    DeltaVn = exper_info.get_injection_volumes_liter()
    beta = 1 / KB / exper_info.get_target_temperature_kelvin()
    n_injections = exper_info.get_number_injections()

    q_model_cal = heats_TwoComponentBindingModel(V0, DeltaVn, P0, Ls, DeltaG, DeltaH, DeltaH_0, beta, n_injections)

    mse = np.mean((q_actual_cal - q_model_cal) ** 2) * 10.**12
    return mse


def minus_log_posterior_rmbm(q_actual_cal, exper_info,
                             DeltaG1, DeltaDeltaG, DeltaH1, DeltaH2, P0, Ls, DeltaH_0, log_sigma,
                             dcell=0.1, dsyringe=0.1,
                             uniform_P0=False, uniform_Ls=False):
    """
    :param q_actual_cal: observed heats in calorie
    :param exper_info: an object of _data_io.ITCExperiment class
    :param DeltaG1: float, free energy of binding of ligand1 (kcal/mol)
    :param DeltaDeltaG: float, difference in binding free energy between ligand2 and ligand1: DeltaDeltaG = DeltaG2 - DeltaG1 > 0
                        DeltaDeltaG is always positive
    :param DeltaH1: float, enthalpy of binding of ligand1 (kcal/mol)
    :param DeltaH2: enthalpies of binding of ligand2 (kcal/mol)
    :param P0: float, Cell concentration (millimolar)
    :param Ls: float, Syringe concentration (millimolar)
    :param DeltaH_0: float, heat of injection (cal)
    :param log_sigma: float, log of sigma, sigma is in cal
    :param dcell: float, relative uncertainty in cell concentration (0 < dcell < 1)
    :param dsyringe: float, relative uncertainty in syringe concentration (0 < dcell < 1)
    :param uniform_P0: bool
    :param uniform_Ls: bool
    :return: minus log posterior
    """

    V0 = exper_info.get_cell_volume_liter()
    DeltaVn = exper_info.get_injection_volumes_liter()
    beta = 1 / KB / exper_info.get_target_temperature_kelvin()
    n_injections = exper_info.get_number_injections()
    rho = 0.5

    q_model_cal = heats_RacemicMixtureBindingModel(V0, DeltaVn, P0, Ls, rho, DeltaH1, DeltaH2, DeltaH_0,
                                                   DeltaG1, DeltaDeltaG, beta, n_injections)

    sigma_cal = np.exp(log_sigma)

    log_posterior = log_likelihood(q_actual_cal, q_model_cal, sigma_cal)

    if not uniform_P0:
        stated_P0 = exper_info.get_cell_concentration_milli_molar()
        uncertainty_P0 = dcell * stated_P0
        log_posterior += log_lognormal(P0, stated_P0, uncertainty_P0)

    if not uniform_Ls:
        stated_Ls = exper_info.get_syringe_concentration_milli_molar()
        uncertainty_Ls = dsyringe * stated_Ls
        log_posterior += log_lognormal(Ls, stated_Ls, uncertainty_Ls)

    return -log_posterior


def mse_rmbm(q_actual_cal, exper_info,
             DeltaG1, DeltaDeltaG, DeltaH1, DeltaH2, P0, Ls, DeltaH_0):
    """
    :param q_actual_cal: observed heats in calorie
    :param exper_info: an object of _data_io.ITCExperiment class
    :param DeltaG1: float, free energy of binding of ligand1 (kcal/mol)
    :param DeltaDeltaG: float, difference in binding free energy between ligand2 and ligand1: DeltaDeltaG = DeltaG2 - DeltaG1 > 0
                        DeltaDeltaG is always positive
    :param DeltaH1: float, enthalpy of binding of ligand1 (kcal/mol)
    :param DeltaH2: enthalpies of binding of ligand2 (kcal/mol)
    :param P0: float, Cell concentration (millimolar)
    :param Ls: float, Syringe concentration (millimolar)
    :param DeltaH_0: float, heat of injection (cal)
    :return: mse
    """

    V0 = exper_info.get_cell_volume_liter()
    DeltaVn = exper_info.get_injection_volumes_liter()
    beta = 1 / KB / exper_info.get_target_temperature_kelvin()
    n_injections = exper_info.get_number_injections()
    rho = 0.5

    q_model_cal = heats_RacemicMixtureBindingModel(V0, DeltaVn, P0, Ls, rho, DeltaH1, DeltaH2, DeltaH_0,
                                                   DeltaG1, DeltaDeltaG, beta, n_injections)

    mse = np.mean((q_actual_cal - q_model_cal) ** 2) * 10.**12
    return mse


def minus_log_posterior_embm(q_actual_cal, exper_info,
                             DeltaG1, DeltaDeltaG, DeltaH1, DeltaH2, P0, Ls, rho, DeltaH_0, log_sigma,
                             dcell=0.1, dsyringe=0.1,
                             uniform_P0=False, uniform_Ls=False):
    """
    :param q_actual_cal: observed heats in calorie
    :param exper_info: an object of _data_io.ITCExperiment class
    :param DeltaG1: float, free energy of binding of ligand1 (kcal/mol)
    :param DeltaDeltaG: float, difference in binding free energy between ligand2 and ligand1: DeltaDeltaG = DeltaG2 - DeltaG1 > 0
                        DeltaDeltaG is always positive
    :param DeltaH1: float, enthalpy of binding of ligand1 (kcal/mol)
    :param DeltaH2: enthalpies of binding of ligand2 (kcal/mol)
    :param P0: float, Cell concentration (millimolar)
    :param Ls: float, Syringe concentration (millimolar)
    :param rho: ratio between concentration of ligand1 and the total concentration of the syringe
    :param DeltaH_0: float, heat of injection (cal)
    :param log_sigma: float, log of sigma, sigma is in cal
    :param dcell: float, relative uncertainty in cell concentration (0 < dcell < 1)
    :param dsyringe: float, relative uncertainty in syringe concentration (0 < dcell < 1)
    :param uniform_P0: bool
    :param uniform_Ls: bool
    :return: minus log posterior
    """

    V0 = exper_info.get_cell_volume_liter()
    DeltaVn = exper_info.get_injection_volumes_liter()
    beta = 1 / KB / exper_info.get_target_temperature_kelvin()
    n_injections = exper_info.get_number_injections()

    q_model_cal = heats_RacemicMixtureBindingModel(V0, DeltaVn, P0, Ls, rho, DeltaH1, DeltaH2, DeltaH_0,
                                                   DeltaG1, DeltaDeltaG, beta, n_injections)

    sigma_cal = np.exp(log_sigma)

    log_posterior = log_likelihood(q_actual_cal, q_model_cal, sigma_cal)

    if not uniform_P0:
        stated_P0 = exper_info.get_cell_concentration_milli_molar()
        uncertainty_P0 = dcell * stated_P0
        log_posterior += log_lognormal(P0, stated_P0, uncertainty_P0)

    if not uniform_Ls:
        stated_Ls = exper_info.get_syringe_concentration_milli_molar()
        uncertainty_Ls = dsyringe * stated_Ls
        log_posterior += log_lognormal(Ls, stated_Ls, uncertainty_Ls)

    return -log_posterior


def mse_embm(q_actual_cal, exper_info,
             DeltaG1, DeltaDeltaG, DeltaH1, DeltaH2, P0, Ls, rho, DeltaH_0):
    """
    :param q_actual_cal: observed heats in calorie
    :param exper_info: an object of _data_io.ITCExperiment class
    :param DeltaG1: float, free energy of binding of ligand1 (kcal/mol)
    :param DeltaDeltaG: float, difference in binding free energy between ligand2 and ligand1: DeltaDeltaG = DeltaG2 - DeltaG1 > 0
                        DeltaDeltaG is always positive
    :param DeltaH1: float, enthalpy of binding of ligand1 (kcal/mol)
    :param DeltaH2: enthalpies of binding of ligand2 (kcal/mol)
    :param P0: float, Cell concentration (millimolar)
    :param Ls: float, Syringe concentration (millimolar)
    :param rho: ratio between concentration of ligand1 and the total concentration of the syringe
    :param DeltaH_0: float, heat of injection (cal)
    :return: mse
    """

    V0 = exper_info.get_cell_volume_liter()
    DeltaVn = exper_info.get_injection_volumes_liter()
    beta = 1 / KB / exper_info.get_target_temperature_kelvin()
    n_injections = exper_info.get_number_injections()

    q_model_cal = heats_RacemicMixtureBindingModel(V0, DeltaVn, P0, Ls, rho, DeltaH1, DeltaH2, DeltaH_0,
                                                   DeltaG1, DeltaDeltaG, beta, n_injections)

    mse = np.mean((q_actual_cal - q_model_cal) ** 2) * 10.**12
    return mse


def generate_objective(model, q_actual_cal, exper_info,
                       dcell=0.1, dsyringe=0.1,
                       uniform_P0=False, uniform_Ls=False):
    """
    :param model: str, one of the values ["2cbm", "rmbm", "embm"]
    :param q_actual_cal: observed heats in calorie
    :param exper_info: an object of _data_io.ITCExperiment class
    :param dcell: float, relative uncertainty in cell concentration (0 < dcell < 1)
    :param dsyringe: float, relative uncertainty in syringe concentration (0 < dcell < 1)
    :param uniform_P0: float, Cell concentration (millimolar)
    :param uniform_Ls: float, Syringe concentration (millimolar)
    :return: the objective function to be optimized
    """

    def objective_2cbm(x):
        DeltaG, DeltaH, P0, Ls, DeltaH_0, log_sigma = x
        m_log_posterior = minus_log_posterior_2cbm(q_actual_cal, exper_info,
                                                   DeltaG, DeltaH, P0, Ls, DeltaH_0, log_sigma,
                                                   dcell=dcell, dsyringe=dsyringe,
                                                   uniform_P0=uniform_P0, uniform_Ls=uniform_Ls)
        return m_log_posterior

    def objective_rmbm(x):
        DeltaG1, DeltaDeltaG, DeltaH1, DeltaH2, P0, Ls, DeltaH_0, log_sigma = x
        m_log_posterior = minus_log_posterior_rmbm(q_actual_cal, exper_info,
                                                   DeltaG1, DeltaDeltaG, DeltaH1, DeltaH2, P0, Ls, DeltaH_0, log_sigma,
                                                   dcell=dcell, dsyringe=dsyringe,
                                                   uniform_P0=uniform_P0, uniform_Ls=uniform_Ls)
        return m_log_posterior

    def objective_embm(x):
        DeltaG1, DeltaDeltaG, DeltaH1, DeltaH2, P0, Ls, rho, DeltaH_0, log_sigma = x
        m_log_posterior = minus_log_posterior_embm(q_actual_cal, exper_info,
                                                   DeltaG1, DeltaDeltaG, DeltaH1, DeltaH2, P0, Ls, rho, DeltaH_0, log_sigma,
                                                   dcell=dcell, dsyringe=dsyringe,
                                                   uniform_P0=uniform_P0, uniform_Ls=uniform_Ls)
        return m_log_posterior

    if model == "2cbm":
        return objective_2cbm
    elif model == "rmbm":
        return objective_rmbm
    elif model == "embm":
        return objective_embm
    else:
        raise ValueError("Unknown model: %s" % model)


def generate_bounds(model, q_actual_cal, exper_info,
                    DeltaG_bound, DeltaDeltaG_bound, DeltaH_bound, rho_bound,
                    dcell=0.1, dsyringe=0.1):
    """
    :param model: str, one of the values ["2cbm", "rmbm", "embm"]
    :param q_actual_cal: observed heats in calorie
    :param exper_info: an object of _data_io.ITCExperiment class
    :param DeltaG_bound: tuple of two floats, (lower, upper)
    :param DeltaDeltaG_bound: tuple of two floats, (lower, upper)
    :param DeltaH_bound: tuple of two floats, (lower, upper)
    :param rho_bound: tuple of two floats, (lower, upper)
    :param dcell: float, relative uncertainty in cell concentration (0 < dcell < 1)
    :param dsyringe: float, relative uncertainty in syringe concentration (0 < dcell < 1)
    :return: list of tuples
    """
    DeltaG = DeltaG_bound
    DeltaDeltaG = DeltaDeltaG_bound
    DeltaH = DeltaH_bound

    assert dcell < 1., "dcell must be less than one"
    stated_P0 = exper_info.get_cell_concentration_milli_molar()
    P0_lower = stated_P0 * (1. - dcell)
    P0_upper = stated_P0 * (1. + dcell)
    P0 = (P0_lower, P0_upper)

    assert dsyringe < 1., "dsyringe must be less than one"
    stated_Ls = exper_info.get_syringe_concentration_milli_molar()
    Ls_lower = stated_Ls * (1. - dsyringe)
    Ls_upper = stated_Ls * (1. + dsyringe)
    Ls = (Ls_lower, Ls_upper)

    rho = rho_bound

    log_sigma_min, log_sigma_max = logsigma_guesses(q_actual_cal)
    log_sigma = (log_sigma_min, log_sigma_max)

    DeltaH_0_min, DeltaH_0_max = deltaH0_guesses(q_actual_cal)
    DeltaH_0 = (DeltaH_0_min, DeltaH_0_max)

    if model == "2cbm":
        bounds = [DeltaG, DeltaH, P0, Ls, DeltaH_0, log_sigma]
    elif model == "rmbm":
        bounds = [DeltaG, DeltaDeltaG, DeltaH, DeltaH, P0, Ls, DeltaH_0, log_sigma]
    elif model == "embm":
        bounds = [DeltaG, DeltaDeltaG, DeltaH, DeltaH, P0, Ls, rho, DeltaH_0, log_sigma]
    else:
        raise ValueError("Unknown model: %s" % model)

    return bounds


def posterior_maximizer(model, q_actual_cal, exper_info,
                        DeltaG_bound, DeltaDeltaG_bound, DeltaH_bound, rho_bound,
                        dcell=0.1, dsyringe=0.1,
                        uniform_P0=False, uniform_Ls=False,
                        maxiter=1000, repeats=100):
    """
    :param model: str, one of the values ["2cbm", "rmbm", "embm"]
    :param q_actual_cal: observed heats in calorie
    :param exper_info: an object of _data_io.ITCExperiment class
    :param DeltaG_bound: tuple of two floats, (lower, upper)
    :param DeltaDeltaG_bound: tuple of two floats, (lower, upper)
    :param DeltaH_bound: tuple of two floats, (lower, upper)
    :param rho_bound: tuple of two floats, (lower, upper)
    :param dcell: float, relative uncertainty in cell concentration (0 < dcell < 1)
    :param dsyringe: float, relative uncertainty in syringe concentration (0 < dcell < 1)
    :param uniform_P0: bool
    :param uniform_Ls: bool
    :return: list of OptimizeResult objects
    """
    objective_func = generate_objective(model, q_actual_cal, exper_info,
                                        dcell=dcell, dsyringe=dsyringe,
                                        uniform_P0=uniform_P0, uniform_Ls=uniform_Ls)

    bounds = generate_bounds(model, q_actual_cal, exper_info,
                             DeltaG_bound, DeltaDeltaG_bound, DeltaH_bound, rho_bound,
                             dcell=dcell, dsyringe=dsyringe)

    results = []
    #for _ in range(repeats):
    #    result = optimize.shgo(objective_func, bounds)
    #    results.append(result)

    for _ in range(repeats):
        result = optimize.dual_annealing(objective_func, bounds, maxiter=maxiter)
        print(len(results), " (DA): ", result.message)
        print("fun: ", result.fun, "x: ", result.x)
        print("\n")
        results.append(result)

    for _ in range(repeats):
        result = optimize.differential_evolution(objective_func, bounds, maxiter=maxiter)
        print(len(results), " (DE): ", result.message)
        print("fun: ", result.fun, "x: ", result.x)
        print("\n")
        results.append(result)

    return results


def create_dict_from_optimize_results(results):
    """
    :param results: is a list of  OptimizeResult objects
    :return: dict
    """
    results.sort(key=lambda item: item.fun)
    best_result = results[0]
    results_dict = {"global": {"fun": best_result.fun, "x": best_result.x}}
    results_dict["all_locals"] = results

    return results_dict


def plot_heat_actual_vs_model(q_actual_micro_cal, model, exper_info, global_minimizer, out,
                              xlabel="injection #", ylabel="heat ($\mu$cal)",
                              font_scale=1.0):
    """
    :param q_actual_micro_cal: 1d array, actual heats in micro calorie
    :param model: str, abbreviated name of model
    :param exper_info: an object of _data_io.ITCExperiment class
    :param global_minimizer: 1d array of optimized parameters
    :param out: str
    :param xlabel: str
    :param ylabel: str
    :param font_scale: float
    :return: None
    """
    if model == "2cbm":
        DeltaG, DeltaH, P0, Ls, DeltaH_0, log_sigma = global_minimizer
        q_model_cal = heats_TwoComponentBindingModel(exper_info.get_cell_volume_liter(),
                                                     exper_info.get_injection_volumes_liter(),
                                                     P0, Ls, DeltaG, DeltaH, DeltaH_0,
                                                     beta=1 / KB / exper_info.get_target_temperature_kelvin(),
                                                     N=exper_info.get_number_injections())
    elif model == "rmbm":
        DeltaG1, DeltaDeltaG, DeltaH1, DeltaH2, P0, Ls, DeltaH_0, log_sigma = global_minimizer
        q_model_cal = heats_RacemicMixtureBindingModel(exper_info.get_cell_volume_liter(),
                                                       exper_info.get_injection_volumes_liter(),
                                                       P0, Ls, 0.5, DeltaH1, DeltaH2, DeltaH_0, DeltaG1, DeltaDeltaG,
                                                       beta=1 / KB / exper_info.get_target_temperature_kelvin(),
                                                       N=exper_info.get_number_injections())
    elif model == "embm":
        DeltaG1, DeltaDeltaG, DeltaH1, DeltaH2, P0, Ls, rho, DeltaH_0, log_sigma = global_minimizer
        q_model_cal = heats_RacemicMixtureBindingModel(exper_info.get_cell_volume_liter(),
                                                       exper_info.get_injection_volumes_liter(),
                                                       P0, Ls, rho, DeltaH1, DeltaH2, DeltaH_0, DeltaG1, DeltaDeltaG,
                                                       beta=1 / KB / exper_info.get_target_temperature_kelvin(),
                                                       N=exper_info.get_number_injections())
    else:
        raise ValueError("Unknown model:" + model)

    q_model_micro_cal = q_model_cal * 10.**6

    print("q_actual_micro_cal:", q_actual_micro_cal)
    print("q_model_micro_cal:", q_model_micro_cal)

    n_inj = len(q_actual_micro_cal)

    sns.set(font_scale=font_scale)
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(3.2, 2.4))
    ax.scatter(range(1, n_inj + 1), q_actual_micro_cal, s=20, c="k", marker="o", label="actual")
    ax.plot(range(1, n_inj + 1), q_model_micro_cal, c="r", linestyle="-", label="model: " + model)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(loc="lower right")

    fig.tight_layout()
    fig.savefig(out, dpi=300)

    return None


