"""
define different heat models
"""

import numpy as np


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

    return q_n


# copied from method equilibrium_concentrations of class RacemicMixtureBindingModel in bayesian_itc/bitc/models.py
def equilibrium_concentrations(Kd1, Kd2, C0_R, C0_L1, C0_L2, V):
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
