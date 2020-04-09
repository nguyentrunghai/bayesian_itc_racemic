"""
to convert pickle file from python 2 to python 3
this script should be run with python 2
"""

import argparse
import os
import pickle

from _data_io import ITCExperiment

parser = argparse.ArgumentParser()
parser.add_argument("--exper_info_dir", type=str, default="05.exper_info")

parser.add_argument("--inp", type=str, default="experimental_information.pickle")
parser.add_argument("--out", type=str, default="experimental_information_dict.pickle")

parser.add_argument("--experiments", type=str, default=" ")

args = parser.parse_args()


def _convert(exper_info_obj):
    exper_info_dict = {}

    exper_info_dict["target_temperature_kelvin"] = exper_info_obj.get_target_temperature_kelvin()

    exper_info_dict["number_injections"] = exper_info_obj.get_number_injections()

    exper_info_dict["cell_volume_liter"] = exper_info_obj.get_cell_volume_liter()

    exper_info_dict["injection_volumes_liter"] = exper_info_obj.get_injection_volumes_liter()

    exper_info_dict["syringe_concentration_milli_molar"] = exper_info_obj.get_syringe_concentration_milli_molar()

    exper_info_dict["cell_concentration_milli_molar"] = exper_info_obj.get_cell_concentration_milli_molar()

    return exper_info_dict
