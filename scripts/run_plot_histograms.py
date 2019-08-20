"""
to compare histogram of DG, DH, P0 and Ls between twocomponent and racemicmixture models
"""

import argparse

parser = argparse.ArgumentParser()
parser.add_argument( "--bitc_mcmc_dir",     type=str, default="bitc_mcmc")
args = parser.parse_args()