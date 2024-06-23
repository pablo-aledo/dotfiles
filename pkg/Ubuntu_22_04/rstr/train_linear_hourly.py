import pandas as pd
from itertools import product
from more_itertools import windowed
import os
import numpy as np
from sklearn.linear_model import LinearRegression
import joblib

all_data = {}
training_set = []
sliding_items = 5

for filename in os.listdir("."):
    if filename.endswith(".ohlc"):
        data = pd.read_csv(filename)
        data.columns = ["date", "o", "h", "l", "c"]
        all_data[filename[0:-5]] = data

all_dates = set()
for key in all_data.keys():
    for date in all_data[key]["date"]:
        all_dates.add(date)

for dates in windowed(sorted(all_dates), sliding_items):
    list_dates = list(dates)
    date_incrs = [ y - x for x, y in windowed(list_dates, 2) ]
    valid_date_incrs = all([x == 3600 for x in date_incrs])
    is_in_all = []
    for date in dates:
        is_in_all.append( all([ date in set(all_data[instr]["date"]) for instr in all_data.keys() ]) )
    all_in_all = all(is_in_all)
    if valid_date_incrs and all_in_all:
        incr = {}
        for instr in sorted(all_data.keys()):
            df = all_data[instr].loc[all_data[instr]["date"] == list_dates[-1]]
            incr[instr] = float(((df["c"] - df["o"])/df["o"]).iloc[0])
        y = [ v for k, v in sorted(incr.items()) ]

        incrs = []
        for delta, instr in product(list(range(-2,-sliding_items-1,-1)), sorted(all_data.keys())):
            df = all_data[instr].loc[all_data[instr]["date"] == list_dates[delta]]
            incr = float(((df["c"] - df["o"])/df["o"]).iloc[0])
            incrs.append(incr)
        x = [ v for v in incrs ]

        training_set.append((x, y))

reg = LinearRegression(fit_intercept=False).fit([t[0] for t in training_set], [t[1] for t in training_set])
print(reg.coef_)
print(reg.intercept_)
joblib.dump(reg, "linear_hourly.pkl")
