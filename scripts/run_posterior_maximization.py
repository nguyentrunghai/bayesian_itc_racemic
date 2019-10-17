"""
run this script to maximize the posterior
"""

from __future__ import print_function

import argparse
import pickle

import numpy as np

from _optimization import posterior_maximizer

parser = argparse.ArgumentParser()
parser.add_argument("--mcmc_dir", type=str, default="5.twocomponent_mcmc")
parser.add_argument("--model", type=str, default="2cbm")

parser.add_argument("--heat_dir", type=str, default="4.heat_in_origin_format")
parser.add_argument("--heat_file", type=str, default="/home/tnguye46/bayesian_itc_racemic/4.heat_in_origin_format/Baum_59.DAT")

parser.add_argument("--exper_info_file", type=str, default="/home/tnguye46/bayesian_itc_racemic/5.twocomponent_mcmc/nsamples_5k/Baum_59/experimental_information.pickle")

parser.add_argument("--dP0", type=float, default=0.1)      # cell concentration relative uncertainty
parser.add_argument("--dLs", type=float, default=0.1)      # syringe concentration relative uncertainty

parser.add_argument("--uniform_P0", action="store_true", default=False)
parser.add_argument("--uniform_Ls", action="store_true", default=False)
parser.add_argument("--concentration_range_factor", type=float, default=50.)

parser.add_argument("--maxiter", type=int, default=1000)
parser.add_argument("--repeats", type=int, default=100)

parser.add_argument("--experiments", type=str, default="Fokkens_1_c Fokkens_1_d")

parser.add_argument("--submit",   action="store_true", default=False)

args = parser.parse_args()


def _load_heat_micro_cal(origin_heat_file):
    """
    :param origin_heat_file: str, name of heat file
    :return: 1d ndarray, heats in micro calorie
    """

    heats = []
    with open(origin_heat_file) as handle:
        handle.readline()
        for line in handle:
            if len(line.split()) == 6:
                heats.append(np.float(line.split()[0]))

    return np.array(heats)


class _ITCExperiment:
    """ store experimental design parameter """

    def __init__(self, experimental_info_pickle):
        """
        :param experimental_info_pickle: str, name of the pickle file
        """
        self._exper_info = pickle.load(open(experimental_info_pickle))

    def get_target_temperature_kelvin(self):
        return self._exper_info["target_temperature"].m

    def get_number_injections(self):
        return self._exper_info["number_of_injections"]

    def get_cell_volume_liter(self):
        return self._exper_info["cell_volume"].m_as("liter")

    def get_injection_volumes_liter(self):
        return [inj.volume.m_as("liter") for inj in self._exper_info["injections"]]

    def get_syringe_concentration_milli_molar(self):
        return self._exper_info["syringe_concentration"]["ligand"].m

    def get_cell_concentration_milli_molar(self):
        return self._exper_info["cell_concentration"]["macromolecule"].m


if args.submit:
    pass
    #TODO

else:
    model = args.model
    heat_file = args.heat_file
    exper_info_file = args.exper_info_file

    q_actual_micro_cal = _load_heat_micro_cal(heat_file)
    q_actual_cal = q_actual_micro_cal * 10 ** (-6)

    exper_info = _ITCExperiment(exper_info_file)

    dcell = args.dcell
    dsyringe = args.dsyringe
    uniform_P0 = args.uniform_P0
    uniform_Ls = args.uniform_Ls
    concentration_range_factor = args.concentration_range_factor

    maxiter = args.maxiter
    repeats = args.repeats

    best_result = posterior_maximizer(model, q_actual_cal, exper_info,
                                      dcell=dcell, dsyringe=dsyringe,
                                      uniform_P0=uniform_P0, uniform_Ls=uniform_P0,
                                      concentration_range_factor=concentration_range_factor,
                                      maxiter=maxiter, repeats=repeats)
