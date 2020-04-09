"""
to convert pickle file from python 2 to python 3
this script should be run with python 2
"""
from __future__ import print_function

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


experiments = args.experiments.split()
print("experiments:", experiments)

exper_info_dir = os.path.abspath(args.exper_info_dir)
print("exper_info_dir:", exper_info_dir)

for exper in experiments:
    print("Processing " + exper)

    inp_file = os.path.join(exper_info_dir, exper, args.inp)
    out_file = os.path.join(exper_info_dir, exper, args.out)
    if os.path.exists(out_file):
        raise ValueError("File exists: " + out_file)

    exper_info_obj = ITCExperiment(inp_file)
    exper_info_dict = _convert(exper_info_obj)

    pickle.dump(exper_info_dict, open(out_file, "wb"))

print("DONE")
