import pandas as pd
from skforecast.utils import load_forecaster
import snapshot

instruments = {}
sliding_elements = 7

data = pd.DataFrame(columns = ["date"] + sorted(list(instruments)))

n = 0
for offset in range(-10, 0):
    new_row = [ snapshot.dates_hist_hr[list(instruments)[0]][offset] ] + [ snapshot.prices_hist_hr[instr][offset] for instr in sorted(instruments) ]
    data.loc[n] = new_row
    n = n + 1

data['date'] = pd.to_datetime(data['date'], unit='s')
data = data.set_index('date')
data = data.sort_index()
data = data.asfreq('h', method="backfill")

for instr in instruments:
	forecaster = load_forecaster(instr.replace('/', ':').replace(' ', '_') + '.joblib', verbose=False)
	last_window = data[-sliding_elements:]
	predictions = forecaster.predict(last_window=last_window, steps=4)
	print(predictions)
