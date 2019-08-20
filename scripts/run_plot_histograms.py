"""
to compare histogram of DG, DH, P0 and Ls between twocomponent and racemicmixture models
"""

import argparse
import glob
import os

parser = argparse.ArgumentParser()
parser.add_argument("--twocomponent_mcmc_dir", type=str, default="twocomponent")
parser.add_argument("--racemicmixture_mcmc_dir", type=str, default="racemicmixture")
args = parser.parse_args()

TRACES_FILE_NAME = "traces.pickle"

twocomponent_traces_files = glob.glob(os.path.join(args.twocomponent_mcmc_dir, "*", TRACES_FILE_NAME))
racemicmixture_traces_files = glob.glob(os.path.join(args.racemicmixture_mcmc_dir, "*", TRACES_FILE_NAME))

twocomponent_traces_files = {os.path.basename(os.path.dirname(f)): f for f in twocomponent_traces_files}
racemicmixture_traces_files = {os.path.basename(os.path.dirname(f)): f for f in racemicmixture_traces_files}
print("twocomponent_traces_files", twocomponent_traces_files)
print("racemicmixture_traces_files", racemicmixture_traces_files)

