import datetime
from datetime import timedelta
import joblib
import macd_threshold
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

data['date'] = pd.to_datetime(data['date'], unit='s')
data = data.set_index('date')
data = data.sort_index()
data = data.asfreq('h', method="backfill")

def simulate_decision(data_w, trade, date, market):

    # if date.strftime("%Y-%m-%d %H:%M:%S") == "2024-06-11 14:53:00":
        # return { "action": "open", "direction": "buy", "quantity": 0.5, "takeprofit": 1, "stoploss": -1 }
    # if date.strftime("%Y-%m-%d %H:%M:%S") == "2024-06-11 15:06:00":
        # return { "action": "close" }
    # return {}

    k = ""
    snapshot.prices_hist_min[k] = data_w[k]

    data = pd.DataFrame()
    data['Close'] = snapshot.prices_hist_min[k]
    data['EMA12'] = data['Close'].ewm(span=12, adjust=False).mean()
    data['EMA26'] = data['Close'].ewm(span=26, adjust=False).mean()
    data['MACD'] = data['EMA12'] - data['EMA26']
    data['Signal_Line'] = data['MACD'].ewm(span=9, adjust=False).mean()
    last_row = data.iloc[-1]
    second_last_row = data.iloc[-2]

    signal_line = last_row['Signal_Line']

    if second_last_row['MACD'] > second_last_row['Signal_Line'] and last_row['MACD'] < last_row['Signal_Line']:
        if k in macd_threshold.threshold.keys():
            pass
        else:
            macd_threshold.threshold[k] = signal_line
        if signal_line > 0 and abs(signal_line) >= macd_threshold.threshold[k] and len(trade.keys()) > 0:
            return { "action": "close" }

    elif second_last_row['MACD'] < second_last_row['Signal_Line'] and last_row['MACD'] > last_row['Signal_Line']:
        if k in macd_threshold.threshold.keys():
            pass
        else:
            macd_threshold.threshold[k] = signal_line

        if signal_line < 0 and abs(signal_line) >= macd_threshold.threshold[k] and len(trade.keys()) == 0:
            return { "action": "open", "direction": "buy", "quantity": 10, "takeprofit": 1, "stoploss": -1 }

    return {}

trade = {}
leverage = 5
net_profit = 0

for offset in range(-len(data), -30):
    data_wind = data[offset:offset + 30]
    market = data_wind[""].iloc[-1]
    date = data_wind.index[-1]

    d = simulate_decision(data_wind, trade, date, market)

    if len(d.keys()) > 0 and d["action"] == "open":
        trade = {"direction": d["direction"], "quantity": d["quantity"], "takeprofit": d["takeprofit"], "stoploss": d["stoploss"], "initial_value": market , "profit": 0 }

    profit = 0 if len(trade.keys()) == 0 else trade["quantity"] * (market - trade["initial_value"]) * leverage

    if len(trade.keys()) > 0 and (profit > trade["takeprofit"] or profit < trade["stoploss"]):
        d["action"] = "close"

    if len(trade.keys()) > 0:
        trade["profit"] = profit

    if len(d.keys()) > 0 and len(trade.keys()) > 0 and d["action"] == "close":
        net_profit = net_profit + trade["profit"]
        trade = {}

    # print(offset, "|", date, "|", market, "|", d, "|", trade, "|", net_profit)
    print(offset, "|", date, "|", market, "|", trade, "|", '\033[31m' if net_profit < 0 else '\033[32m', net_profit, '\033[0m')

