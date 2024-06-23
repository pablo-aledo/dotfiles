import joblib
import os
import pandas as pd
from skforecast.ForecasterAutoregMultiVariate import ForecasterAutoregMultiVariate
from skforecast.utils import save_forecaster
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

all_data = {}
for filename in os.listdir("."):
    if filename.endswith(".ohlc"):
        data = pd.read_csv(filename)
        data.columns = ["date", "o", "h", "l", "c"]
        all_data[filename[0:-5].replace(':', '/').replace('_', ' ')] = data

all_dates = set()
for key in all_data.keys():
    for date in all_data[key]["date"]:
        all_dates.add(date)

data = pd.DataFrame(columns = ["date"] + sorted(list(all_data.keys())) )
n = 0
# current_date = sorted(all_dates)[-1]
# for date in sorted(all_dates, reverse = True):
for date in all_dates:
    is_in_all = all( [ date in set(all_data[instr]["date"]) for instr in sorted(all_data.keys()) ] )
    if is_in_all:
        new_row = [ date ] + [ float(all_data[instr].loc[ all_data[instr]["date"] == date ]["c"].iloc[0]) for instr in all_data.keys() ]
        data.loc[n] = new_row
        n = n + 1
        # current_date = current_date - 3600

data['date'] = pd.to_datetime(data['date'], unit='s')
data = data.set_index('date')
data = data.sort_index()
data = data.asfreq('h', method="backfill")
# print(f'Number of rows with missing values: {data.isnull().any(axis=1).mean()}')

for instr in all_data.keys():
    forecaster = ForecasterAutoregMultiVariate(
                     regressor          = Ridge(random_state=123),
                     level              = instr,
                     lags               = 7,
                     steps              = 7,
                     transformer_series = StandardScaler(),
                     transformer_exog   = None,
                     weight_func        = None,
                     n_jobs             = 'auto'
                 )
    forecaster.fit(series=data)
    save_forecaster(forecaster, file_name=(instr + '.joblib').replace('/', ':').replace(' ', '_'), verbose=False)

