"""
define different heat models
"""

import numpy as np
import pymc

KB = 0.0019872041      # in kcal/mol/K

# copied from the method expected_injection_heats of the class TwoComponentBindingModel in bayesian_itc/bitc/models.py
def heats_TwoComponentBindingModel(V0, DeltaVn, P0, Ls, DeltaG, DeltaH, DeltaH_0, beta, N):
    """
    Expected heats of injection for two-component binding model.

    ARGUMENTS
    V0 - cell volume (liter)
    DeltaVn - injection volumes (liter)
    P0 - Cell concentration (millimolar)
    Ls - Syringe concentration (millimolar)
    DeltaG - free energy of binding (kcal/mol)
    DeltaH - enthalpy of binding (kcal/mol)
    DeltaH_0 - heat of injection (cal)
    beta - inverse temperature * gas constant (mole / kcal)
    N - number of injections

    Returns
    -------
    expected injection heats (calorie)

    """

    Kd = np.exp(beta * DeltaG)   # dissociation constant (M)
    N = N

    # Compute complex concentrations.
    # Pn[n] is the protein concentration in sample cell after n injections
    # (M)
    Pn = np.zeros([N])
    # Ln[n] is the ligand concentration in sample cell after n injections
    # (M)
    Ln = np.zeros([N])
    # PLn[n] is the complex concentration in sample cell after n injections
    # (M)
    PLn = np.zeros([N])
    dcum = 1.0  # cumulative dilution factor (dimensionless)
    for n in range(N):
        # Instantaneous injection model (perfusion)
        # TODO: Allow injection volume to vary for each injection.
        # dilution factor for this injection (dimensionless)
        d = 1.0 - (DeltaVn[n] / V0)
        dcum *= d  # cumulative dilution factor
        # total quantity of protein in sample cell after n injections (mol)
        P = V0 * P0 * 1.e-3 * dcum
        # total quantity of ligand in sample cell after n injections (mol)
        L = V0 * Ls * 1.e-3 * (1. - dcum)
        # complex concentration (M)
        PLn[n] = (0.5 / V0 * ((P + L + Kd * V0) - ((P + L + Kd * V0) ** 2 - 4 * P * L) ** 0.5))
        # free protein concentration in sample cell after n injections (M)
        Pn[n] = P / V0 - PLn[n]
        # free ligand concentration in sample cell after n injections (M)
        Ln[n] = L / V0 - PLn[n]

    # Compute expected injection heats.
    # q_n_model[n] is the expected heat from injection n
    q_n = np.zeros([N])
    # Instantaneous injection model (perfusion)
    # first injection
    q_n[0] = (DeltaH * V0 * PLn[0])*1000 + DeltaH_0
    for n in range(1, N):
        d = 1.0 - (DeltaVn[n] / V0)  # dilution factor (dimensionless)
        # subsequent injections
        q_n[n] = (DeltaH * V0 * (PLn[n] - d * PLn[n - 1])) * 1000 + DeltaH_0

    return np.array(q_n)


# copied from method equilibrium_concentrations of class RacemicMixtureBindingModel in bayesian_itc/bitc/models.py
def _equilibrium_concentrations(Kd1, Kd2, C0_R, C0_L1, C0_L2, V):
    """
    REF: Zhi-Xin Wang An exact mathematical expression for describing competitive
                        binding of two different ligands to a protein molecule
                        FEBS Letters 360 (1995) 111-114

    Kd1, Kd2        :   disociation constants of ligand 1 and 2, respectively (M)
    C0_R            :   total concentration receptor   (M)
    C0_L1, C0_L2    :   total concentrations of ligand 1 and ligand 2, respectively (M)
    V               :   volume of sample cell (L)
    return
                [RL1, RL2]  complex concentrations (M)
    """
    assert (Kd1 > 0) and (Kd2 > 0), "Kd1 and Kd2 must be positive. Kd1=%0.5f, Kd2=%0.5f" %(Kd1, Kd2)
    assert (C0_R > 0) and (C0_L1 > 0) and (C0_L2 > 0), "concentrations must be positive (R, L1, L2) = (%0.5f, %0.5f, %0.5f)" %(C0_R, C0_L1, C0_L2)
    assert V > 0, "volume must be positive"

    a = Kd1 + Kd2 + C0_L1 + C0_L2 - C0_R

    b = Kd2*(C0_L1 - C0_R) + Kd1*(C0_L2 - C0_R) + Kd1*Kd2

    c = -Kd1 * Kd2 * C0_R

    d = np.sqrt(a*a - 3*b)

    theta = np.arccos((-2.*a**3 + 9.*a*b - 27.*c) / (2.*d**3))

    RL1 = C0_L1*(2.*d*np.cos(theta/3.) - a) / (3.*Kd1 + (2.*d*np.cos(theta/3.) - a))
    RL2 = C0_L2*(2.*d*np.cos(theta/3.) - a) / (3.*Kd2 + (2.*d*np.cos(theta/3.) - a))

    return RL1, RL2


def heats_RacemicMixtureBindingModel(V0, DeltaVn, P0, Ls, rho, DeltaH1, DeltaH2, DeltaH_0, DeltaG1, DeltaDeltaG, beta, N):
    """
    Expected heats of injection for racemic mixtrue binding model.
    :param V0: cell volume (liter)
    :param DeltaVn: injection volumes (liter)
    :param P0: Cell concentration (millimolar)
    :param Ls: Syringe concentration (millimolar)
    :param rho: ratio between concentration of ligand1 and the total concentration of the syringe
    :param DeltaH1: enthalpies of binding of ligand1 (kcal/mol)
    :param DeltaH2: enthalpies of binding of ligand2 (kcal/mol)
    :param DeltaH_0: heat of injection (cal)
    :param DeltaG1: free energy of binding of ligand1 (kcal/mol)
    :param DeltaDeltaG: difference in binding free energy between ligand2 and ligand1: DeltaDeltaG = DeltaG2 - DeltaG1 > 0
                        DeltaDeltaG is always positive
    :param beta: inverse temperature * gas constant (mole / kcal)
    :param N: number of injections

    :return: q_n - expected injection heat (calorie)
    """

    assert 0 < rho < 1, "rho out of range: %0.5f" %rho

    # compute desociation constant (M)
    DeltaG2 = DeltaG1 + DeltaDeltaG

    Kd1 = np.exp(beta * DeltaG1)   # dissociation constant of ligand1 (M)
    Kd2 = np.exp(beta * DeltaG2)   # dissociation constant of ligand2 (M)

    # Compute complex concentrations in the sample after injection n.
    RL1n = np.zeros([N])
    RL2n = np.zeros([N])
    dcum = 1.0  # cumulative dilution factor (dimensionless)
    for n in range(N):
        # Instantaneous injection model (perfusion)
        # TODO: Allow injection volume to vary for each injection.
        # dilution factor for this injection (dimensionless)
        d = 1.0 - (DeltaVn[n] / V0)
        dcum *= d  # cumulative dilution factor

        # total concentration of protein in sample cell after n injections (mol)
        C0_R = P0 * 1.e-3 * dcum

        # total concentration of ligands in sample cell after n injections (mol)
        C0_L = Ls * 1.e-3 * (1. - dcum)

        # total concentration of ligand 1
        C0_L1 = rho * C0_L
        # total concentration of ligand 2
        C0_L2 = (1. - rho) * C0_L

        RL1n[n], RL2n[n] = _equilibrium_concentrations(Kd1, Kd2, C0_R, C0_L1, C0_L2, V0)

    # Compute expected injection heats.
    # q_n_model[n] is the expected heat from injection n
    q_n = np.zeros([N])
    # Instantaneous injection model (perfusion)
    # first injection
    q_n[0] = (DeltaH1 * V0 * RL1n[0] + DeltaH2 * V0 * RL2n[0]) * 1000 + DeltaH_0
    # TODO do we need dcum = 1.0 here?
    for n in range(1, N):
        d = 1.0 - (DeltaVn[n] / V0)  # dilution factor (dimensionless)
        # subsequent injections
        q_n[n] = (DeltaH1 * V0 * (RL1n[n] - d*RL1n[n-1]) + DeltaH2 * V0 * (RL2n[n] - d*RL2n[n-1])) * 1000 + DeltaH_0

    return np.array(q_n)


def normal_likelihood(q_actual, q_model, sigma):
    """
    :param q_actual: 1d ndarray, actual or observed values of heats
    :param q_model: heat calculated from a model
    :param sigma: standard deviation
    :return: likelihood, float

    log_likelihood = -(N/2)\ln(2 \pi \sigma^2) - 1/(2 \sigma^2) \sum_{i=1}^N \epsilon^2
    """
    assert len(q_actual) == len(q_model), "q_actual and q_model must have the same len"
    sum_e_squared = np.sum((q_model - q_actual)**2)

    n_injections = len(q_actual)
    sigma_2 = sigma**2
    log_likelihood = - n_injections / 2 * np.log(2 * np.pi * sigma_2) - sum_e_squared / 2 / sigma_2

    return np.exp(log_likelihood)


def lognormal_pdf(x, stated_center, uncertainty):
    """
    :param x: float
    :param stated_center: float
    :param uncertainty: float
    :return: pdf, float
    """
    if x <= 0:
        return 0.

    m = stated_center
    v = uncertainty**2

    mu = np.log(m / np.sqrt(1 + (v / (m ** 2))))
    sigma_2 = np.log(1 + (v / (m**2)))

    pdf = 1 / x / np.sqrt(2 * np.pi * sigma_2) * np.exp(-0.5 / sigma_2 * (np.log(x) - mu)**2)
    return pdf


def uniform_pdf(x, lower, upper):
    """
    :param x:float
    :param lower: float
    :param upper: float
    :return: pdf, float
    """
    assert upper > lower, "upper must be greater than lower"
    if (x < lower) or (x > upper):
        return 0.

    return 1. / (upper - lower)


# copied from bayesian_itc/bitc/models
def logsigma_guesses(q_n_cal):
    """
    :param q_n_cal: heats in calorie
    :return:
    """
    log_sigma_guess = np.log(q_n_cal[-4:].std())
    log_sigma_min = log_sigma_guess - 10
    log_sigma_max = log_sigma_guess + 5
    return log_sigma_min, log_sigma_max


# copied from bayesian_itc/bitc/models
def deltaH0_guesses(q_n_cal):
    # Assume the last injection has the best guess for H0
    DeltaH_0_guess = q_n_cal[-1]
    heat_interval = (q_n_cal.max() - q_n_cal.min())
    DeltaH_0_min = q_n_cal.min() - heat_interval
    DeltaH_0_max = q_n_cal.max() + heat_interval
    return DeltaH_0_min, DeltaH_0_max



class PyMCLogNormal(object):
    def __init__(self, name, stated_value, uncertainty_percent):
        """
        :param name: str
        :param stated_value: float, mM
        :param uncertainty: float, 0 < uncertainty < 1
        """
        m = stated_value
        uncertainty = stated_value * uncertainty_percent
        v = uncertainty ** 2
        model = pymc.Lognormal(name,
                               mu=np.log(m / np.sqrt(1 + (v / (m**2)))),
                               tau=1.0 / np.log(1 + (v / (m**2))),
                               value=m)

        setattr(self, name, model)


class PyMCUniform(object):
    def __init__(self, lower, upper):
        """
        :param lower: float
        :param upper: float
        """
        assert lower < upper, "lower must be less than upper"