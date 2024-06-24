import datetime
from datetime import timedelta
import joblib
import numpy as np
import os
import pandas as pd
import snapshot
from skforecast.ForecasterAutoreg import ForecasterAutoreg
from skforecast.ForecasterAutoregMultiVariate import ForecasterAutoregMultiVariate
from skforecast.utils import save_forecaster
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from more_itertools import windowed

rt_elements = 100

all_data = {}
for filename in os.listdir("."):
    if filename.endswith(".ohlc"):

        # data = pd.read_csv(filename)
        # data.columns = ["date", "o", "h", "l", "c"]
        data = pd.DataFrame(columns = ["date", "o", "h", "l", "c"])

        all_data[filename[0:-5].replace(':', '/').replace('_', ' ')] = data

all_dates = set()
for key in all_data.keys():
    for date in all_data[key]["date"]:
        all_dates.add(date)

data = pd.DataFrame(columns = ["date"] + list(all_data.keys()) )
n = 0
# current_date = sorted(all_dates)[-1]
for date in sorted(all_dates, reverse = True):
    is_in_all = all( [ date in set(all_data[instr]["date"]) for instr in sorted(all_data.keys()) ] )
    if is_in_all:
        new_row = [ date ] + [ float(all_data[instr].loc[ all_data[instr]["date"] == date ]["c"].iloc[0]) for instr in all_data.keys() ]
        data.loc[n] = new_row
        n = n + 1
        # current_date = current_date - 3600

# last_ohlc = datetime.datetime.fromtimestamp(max(data["date"])) - timedelta( minutes = 20 )
first_rt  = datetime.datetime.fromtimestamp(min(snapshot.dates_hist_min[list(all_data.keys())[0]][-rt_elements:])) + timedelta( minutes = 20 )
def custom_weights(index):
    return np.where(
                  (index >= last_ohlc) & (index <= first_rt),
                   0,
                   1
              )

current_date = snapshot.dates_hist_min[list(all_data.keys())[0]][-1]
for offset in range(-1, -rt_elements-1, -1):

    date = current_date
    # date = snapshot.dates_hist_min[list(all_data.keys())[0]][offset]

    is_in_all = all( [ date in snapshot.dates_hist_min[instr] for instr in sorted(all_data.keys()) ] )
    if is_in_all:
        new_row = [ date ] + [ snapshot.prices_hist_min[instr][offset] for instr in all_data.keys() ]
        data.loc[n] = new_row
        n = n + 1
        current_date = current_date - 60

data['date'] = pd.to_datetime(data['date'], unit='s')
data = data.set_index('date')
data = data.sort_index()
data = data.asfreq('min', method="backfill")
# print(f'Number of rows with missing values: {data.isnull().any(axis=1).mean()}')

elems = set()
for instr in all_data.keys():
    forecaster = ForecasterAutoregMultiVariate(
                     regressor          = Ridge(random_state=123),
                     level              = instr,
                     lags               = 20,
                     steps              = 20,
                     transformer_series = StandardScaler(),
                     transformer_exog   = None,
                     # weight_func        = custom_weights,
                     n_jobs             = 'auto'
                 )
    forecaster.fit(data)

    predictions = forecaster.predict(steps=20)
    lpredictions = list(predictions[instr])
    min_p = min(lpredictions)
    max_p = max(lpredictions)

    signs = [ (y - x) for x, y in windowed(lpredictions, 2) ]
    signs = [ (-100 if lpredictions[p] == min_p else (100 if lpredictions[p] == max_p else (x))) for x,p in zip(signs, range(0, len(lpredictions))) ]
    ssigns = ''.join([ ("m" if s == -100 else ("M" if s == 100 else ("+" if s > 0 else ( "." if s == 0 else "-" )))) for s in signs ])

    p_signs = [ (y - x) for x, y in windowed(snapshot.prices_hist_min[instr], 2) ][-5:]
    pssigns = ''.join([ ("+" if s > 0 else ( "." if s == 0 else "-" )) for s in p_signs ])

    elems.add((instr, snapshot.prices[instr], pssigns + ":" + ssigns, min_p, max_p, (max_p-min_p)/((min_p+max_p)/2.0)))

for elem in sorted(elems, key=lambda e:e[5]):
    print(elem[0], "|", elem[1], "|", elem[2], "|", elem[3], "|", elem[4], "|", elem[5] )

