"""
Run sampling of the posterior
"""

import pickle

import pymc3
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

from _data_io import ITCExperiment, load_heat_micro_cal
from _pymc3_models import make_TwoComponentBindingModel

exper_info_file = "/home/tnguye46/bayesian_itc_racemic/5.exper_info/Baum_59/experimental_information.pickle"
heat_file = "/home/tnguye46/bayesian_itc_racemic/4.heat_in_origin_format/Baum_59.DAT"

exper_info = ITCExperiment(exper_info_file)

q_actual_micro_cal = load_heat_micro_cal(heat_file)
q_actual_cal = q_actual_micro_cal * 10.**(-6)

# TODO: uniform_P0=True, uniform_Ls=True
model = make_TwoComponentBindingModel(q_actual_cal, exper_info,
                                      dcell=0.1, dsyringe=0.1,
                                      uniform_P0=False, uniform_Ls=False, concentration_range_factor=10)

# TODO: try steps: Metropolis, HamiltonianMC, NUTS, SMC
with model:
    step = pymc3.Metropolis()
    trace = pymc3.sample(draws=500, tune=2000, step=step)

pickle.dump(trace, open("trace_obj.pkl", "w"))

free_vars = [name for name in trace.varnames if not name.endswith("__")]
trace_vars = {name: trace.get_values(name) for name in free_vars}
pickle.dump(trace_vars, open("trace.pkl", "w"))

plt.figure()
pymc3.traceplot(trace)
plt.tight_layout()
plt.savefig("test.pdf")

