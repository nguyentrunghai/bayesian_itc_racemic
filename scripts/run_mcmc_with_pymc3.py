"""
Run sampling of the posterior
"""

import pymc3

from _data_io import ITCExperiment, load_heat_micro_cal
from _pymc3_models import make_TwoComponentBindingModel

exper_info_file = "/home/tnguye46/bayesian_itc_racemic/5.exper_info/Baum_59/experimental_information.pickle"
heat_file = "/home/tnguye46/bayesian_itc_racemic/4.heat_in_origin_format/Baum_59.DAT"

exper_info = ITCExperiment(exper_info_file)

q_actual_micro_cal = load_heat_micro_cal(heat_file)
q_actual_cal = q_actual_micro_cal * 10.**(-6)

model = make_TwoComponentBindingModel(q_actual_cal, exper_info,
                                      dcell=0.1, dsyringe=0.1,
                                      uniform_P0=False, uniform_Ls=False, concentration_range_factor=10)
