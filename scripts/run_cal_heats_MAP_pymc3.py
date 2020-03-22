"""
Calculate heat from MAP estimate of the parameters
"""

from __future__ import print_function

import pymc3


def find_MAP(model, method="L-BFGS-B"):
    return pymc3.find_MAP(model=model, method=method)

