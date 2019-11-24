
import pickle

from _data_io import ITCExperiment, load_heat_micro_cal
from _pymc3_models import make_TwoComponentBindingModel, make_RacemicMixtureBindingModel

exper_info_file = "/home/tnguye46/bayesian_itc_racemic/05.exper_info/Baum_59/experimental_information.pickle"
heat_file = "/home/tnguye46/bayesian_itc_racemic/04.heat_in_origin_format/Baum_59.DAT"

# "2cbm", "rmbm", "embm"
model_name = "2cbm"

dcell = 0.1
dsyringe = 0.1
uniform_P0 = True
uniform_Ls = True
concentration_range_factor = 10.

auto_transform = False

exper_info = ITCExperiment(exper_info_file)
q_actual_micro_cal = load_heat_micro_cal(heat_file)
q_actual_cal = q_actual_micro_cal * 10. ** (-6)

if model_name == "2cbm":
    pm_model = make_TwoComponentBindingModel(q_actual_cal, exper_info,
                                             dcell=dcell, dsyringe=dsyringe,
                                             uniform_P0=uniform_P0, uniform_Ls=uniform_Ls,
                                             concentration_range_factor=concentration_range_factor,
                                             auto_transform=auto_transform)

elif model_name == "rmbm":
    pm_model = make_RacemicMixtureBindingModel(q_actual_cal, exper_info,
                                               dcell=dcell, dsyringe=dsyringe,
                                               uniform_P0=uniform_P0, uniform_Ls=uniform_Ls,
                                               concentration_range_factor=concentration_range_factor,
                                               is_rho_free_param=False,
                                               auto_transform=auto_transform)

elif model_name == "embm":
    pm_model = make_RacemicMixtureBindingModel(q_actual_cal, exper_info,
                                               dcell=dcell, dsyringe=dsyringe,
                                               uniform_P0=uniform_P0, uniform_Ls=uniform_Ls,
                                               concentration_range_factor=concentration_range_factor,
                                               is_rho_free_param=True,
                                               auto_transform=auto_transform)
else:
    raise ValueError("Unknown model: " + model_name)


traces = pickle.load(open("traces.pickle"))
inp_data = {key: traces[key][0] for key in traces}

logps = {rv.name: rv.logp(**inp_data) for rv in pm_model.free_RVs}

for rv in pm_model.observed_RVs:
    logps[rv.name] = rv.logp(**inp_data)

