
import os
import pickle

from bitc.instruments import Instrument                                                                                   
from bitc.experiments import ExperimentMicroCal, ExperimentMicroCalWithDummyITC

import numpy as np

def read_experimental_design_parameters(file_name):                                                                                                                       
    """
    file_name   :   str, this file has a specific format and was made by hand
    return
        parameters  :   dict    parameters[exper_name]  -> {"syringe_concentration":float, ...}
    """
    parameters = {}
    with open(file_name, "r") as handle:
        for line in handle:
            if not line.startswith("#"):
                entries = line.split()

                label = entries[1]
                parameters[label] = {}
                parameters[label]["syringe_concentration"] = float(entries[2])
                parameters[label]["cell_concentration"] = float(entries[3])
                parameters[label]["number_of_injections"] = int(entries[5])
                parameters[label]["injection_volume"] = float(entries[6])
                parameters[label]["spacing"] = int(entries[7])
                parameters[label]["stir_rate"] = int(entries[8])
    return parameters



def write_dummy_itc_file(experimental_design_parameters, out_file_name,
                         number_of_injections=None,
                         target_temperature=25,
                         equilibration_time=180,
                         reference_power=5,
                         injection_duration=3,
                         filter_period=5,
                         cell_volume=1.434,
                         bottom="# 25\n# 3.649\n# 9.80219\n?\n"):
    if number_of_injections is not None:
        number_of_injections = number_of_injections
    else:
        number_of_injections = experimental_design_parameters["number_of_injections"]

    out_string = "$ITC\n"
    out_string += "$ %d\n" % number_of_injections
    out_string += "$NOT\n"

    out_string += "$ %d\n" % target_temperature
    out_string += "$ %d\n" % equilibration_time
    out_string += "$ %d\n" % experimental_design_parameters["stir_rate"]
    out_string += "$ %d\n" % reference_power
    out_string += "$ 2\n"

    out_string += "$ADCGainCode:  3\n"
    out_string += "$False,True,True\n"

    for injection in range(number_of_injections):
        out_string += "$ %0.5f , %d , %d , %d\n" % (experimental_design_parameters["injection_volume"],
                                                    injection_duration,
                                                    experimental_design_parameters["spacing"],
                                                    filter_period)
    out_string += "# 0\n"
    out_string += "# %0.5f\n" % experimental_design_parameters["syringe_concentration"]
    out_string += "# %0.5f\n" % experimental_design_parameters["cell_concentration"]
    out_string += "# %0.5f\n" % cell_volume

    out_string += bottom
    open(out_file_name, "w").write(out_string)
    return None


def write_heat_in_origin_format(heats, injection_volume, out):
    out_string = "%12s %5s %12s %12s %12s %12s\n" % ("DH", "INJV", "Xt", "Mt", "XMt", "NDH")

    for heat in heats:
        out_string += "%12.5f %5.1f %12.5f %12.5f %12.5f %12.5f\n" % (heat, injection_volume, 0., 0., 0., 0.)

    out_string += "        --           0.00000      0.00000 --"

    open(out, "w").write(out_string)

    return None


def _make_experiment_from_itc_file(itc_file, is_dummy_itc_file=False):
    instrument = Instrument(itcfile=itc_file)
    if is_dummy_itc_file:
        experiment = ExperimentMicroCalWithDummyITC(itc_file, os.path.basename(itc_file), instrument)
    else:
        experiment = ExperimentMicroCal(itc_file, os.path.basename(itc_file), instrument)
    return experiment


def cell_concentration(itc_file, is_dummy_itc_file=False):
    experiment = _make_experiment_from_itc_file(itc_file, is_dummy_itc_file=is_dummy_itc_file)
    return experiment.cell_concentration["macromolecule"].m


class ITCExperiment:
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


def load_heat_micro_cal(origin_heat_file):
    """
    :param origin_heat_file: str, name of heat file
    :return: 1d ndarray
    """

    heats = []
    with open(origin_heat_file) as handle:
        handle.readline()
        for line in handle:
            if len(line.split()) == 6:
                heats.append(np.float(line.split()[0]))

    return np.array(heats)
