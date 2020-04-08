
import pickle
import numpy as np

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
    :return: 1d ndarray, heats in micro calorie
    """

    heats = []
    with open(origin_heat_file) as handle:
        handle.readline()
        for line in handle:
            if len(line.split()) == 6:
                heats.append(np.float(line.split()[0]))

    return np.array(heats)
