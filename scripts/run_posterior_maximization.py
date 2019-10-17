"""
run this script to maximize the posterior
"""

from __future__ import print_function

import argparse

from _optimization import posterior_maximizer

parser = argparse.ArgumentParser()
parser.add_argument("--mcmc_dir", type=str, default="5.twocomponent_mcmc")
parser.add_argument("--model", type=str, default="2cbm")

parser.add_argument("--heat_dir", type=str, default="4.heat_in_origin_format")
parser.add_argument("--heat_file", type=str, default="heat.DAT")

parser.add_argument("--exper_info_file", type=str, default="experimental_information.pickle")

parser.add_argument("--dP0", type=float, default=0.1)      # cell concentration relative uncertainty
parser.add_argument("--dLs", type=float, default=0.1)      # syringe concentration relative uncertainty

parser.add_argument("--uniform_P0", action="store_true", default=False)
parser.add_argument("--uniform_Ls", action="store_true", default=False)
parser.add_argument("--concentration_range_factor", type=float, default=10.)

parser.add_argument("--maxiter", type=int, default=1000)
parser.add_argument("--repeats", type=int, default=100)

parser.add_argument("--experiments", type=str, default="Fokkens_1_c Fokkens_1_d")

parser.add_argument("--submit",   action="store_true", default=False)

args = parser.parse_args()

if args.submit:
    #TODO

else:
    best_result = posterior_maximizer(model, q_actual_cal, exper_info,
                        dcell=0.1, dsyringe=0.1,
                        uniform_P0=False, uniform_Ls=False,
                        concentration_range_factor=50.,
                        maxiter=1000, repeats=100)