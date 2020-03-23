"""
Test BAR estimator for calculating Bayes factors for simple model
"""

import numpy as np


def generate_X1(size=100, random_state=123):
    rand = np.random.RandomState(random_state)
    mu = rand.normal(loc=10, scale=3, size=size)
    x1 = [rand.normal(loc=m, scale=4) for m in mu]
    return np.array(x1)


