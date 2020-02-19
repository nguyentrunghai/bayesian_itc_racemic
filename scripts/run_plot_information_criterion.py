"""
plot AIC (Akaike information criterion), BIC (Bayesian nformation criterion) and
DIC (deviance information criterion)
"""

import numpy as np

def aic(log_llhs, k):
    """
    aci = -2 log p(y | \theta_{MLE}) + 2k
    Ref: Andrew Gelman et al., "Bayesian Data Analysis", 3rd Ed., CRC Press, page 169
    :param log_llhs: array-like of float, log likelihood values
    :param k: number of parameters
    :return: float, Akaike information criterion
    """
    return -2 * np.max(log_llhs) + 2 * k

