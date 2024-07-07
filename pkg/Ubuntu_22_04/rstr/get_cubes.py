import snapshot
from itertools import product
import joblib
import numpy as np
import pandas as pd

sliding_elements_hr = 5
instruments = { }

cubes = []
lr = joblib.load("linear_hourly.pkl")
n_simulations = 10
n_timesteps = 500

initial_data = pd.DataFrame(columns = ["timestep"] + list(sorted(instruments)), index = range(0, -sliding_elements_hr, -1))
initial_data.timestep = range(0, -sliding_elements_hr, -1)
initial_data = initial_data.set_index("timestep")
for n, instr in product(range(-1, -sliding_elements_hr-1, -1), sorted(instruments)):
    initial_data.loc[n+1, instr] = snapshot.prices_hist_hr[instr][n]

for sim in range(n_simulations):
    diffs = ( - initial_data.diff() / initial_data ).drop([0])
    value = { instr: initial_data.iloc[0][instr] for instr in sorted(instruments) }
    stdv  = { instr: np.std(diffs[instr]) for instr in sorted(instruments) }
    cubes.append( pd.DataFrame(columns = ["timestep"] + list(sorted(instruments))).set_index("timestep") )

    for t in range(n_timesteps):
        all_deltas_hr = []
        for delta, instr in product(list(range(t-1,t-sliding_elements_hr,-1)), sorted(instruments)):
            all_deltas_hr.append(diffs.loc[delta, instr])

        x = [ d for d in all_deltas_hr ]
        y = lr.predict([x])[0]
        y = [ np.random.normal(y[i], stdv[sorted(instruments)[i]]) for i in range(len(y)) ]

        timestep_add = max(diffs.index) + 1
        timestep_rm  = min(diffs.index)
        diffs = diffs.drop([timestep_rm])
        for i, instr in enumerate(sorted(instruments)):
            diffs.loc[timestep_add, instr] = y[i]

        value = { instr: value[instr]  + value[instr] * diffs.loc[timestep_add, instr] for instr in sorted(instruments) }

        for instr in sorted(instruments):
            cubes[sim].loc[timestep_add,instr] = value[instr]

