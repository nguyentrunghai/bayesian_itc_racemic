"""
define function to calculate bayes factor
"""

from __future__ import print_function

import numpy as np

from _models import marginal_likelihood


def marginal_lhs_bootstrap(extracted_loglhs, sample_size=None, bootstrap_repeats=1):
    """
    :param extracted_loglhs: 1d array
    :param sample_size: int
    :param bootstrap_repeats: int
    :return: all_sample_estimate, bootstrap_samples
    """
    all_sample_estimate = marginal_likelihood(extracted_loglhs)

    bootstrap_samples = []
    for _ in range(bootstrap_repeats):
        drawn_loglhs = np.random.choice(extracted_loglhs, size=sample_size, replace=True)
        bootstrap_samples.append(drawn_loglhs)

    bootstrap_samples = np.array(bootstrap_samples)
    
    return all_sample_estimate, bootstrap_samples

