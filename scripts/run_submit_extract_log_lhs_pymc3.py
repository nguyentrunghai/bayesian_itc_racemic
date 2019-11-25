"""
To submit and run jobs to extract log priors and likelihoods from traces
"""
from __future__ import print_function

import os
import glob
import sys
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--exper_info_dir", type=str, default="twocomponent_mcmc")
parser.add_argument("--exper_info_file", type=str, default="experimental_information.pickle")

# models "2cbm", "rmbm", "embm"
parser.add_argument("--model", type=str, default="2cbm")

parser.add_argument("--heat_dir", type=str, default="heat_in_origin_format")
parser.add_argument("--heat_file", type=str, default="heat.DAT")

parser.add_argument("--traces_file", type=str, default="traces.pickle")

parser.add_argument("--dP0", type=float, default=0.1)      # cell concentration relative uncertainty
parser.add_argument("--dLs", type=float, default=0.1)      # syringe concentration relative uncertainty

parser.add_argument("--uniform_P0", action="store_true", default=False)
parser.add_argument("--uniform_Ls", action="store_true", default=False)
parser.add_argument("--concentration_range_factor", type=float, default=10.)


parser.add_argument("--out_dir", type=str, default="out")
parser.add_argument("--write_qsub_script", action="store_true", default=False)
parser.add_argument("--submit", action="store_true", default=False)
args = parser.parse_args()

assert args.model in ["2cbm", "rmbm", "embm"], "Unknown model:" + args.model

if args.write_qsub_script:
    assert os.path.exists(args.exper_info_dir), args.exper_info_dir + " does not exist."
    assert os.path.exists(args.heat_dir), args.heat_dir + " does not exist."

    this_script = os.path.abspath(sys.argv[0])
    