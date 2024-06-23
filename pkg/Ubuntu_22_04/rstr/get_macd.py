from time import sleep, time
import pandas as pd
import os
import snapshot
import macd_threshold
from importlib import reload

signal_line = {}
sliding_elements = 30

while True:

    while int(time()) % 60 != 0:
        sleep(0.3)
    sleep(30)

    reload(macd_threshold)
    reload(snapshot)
    os.system('clear')

    for k,v in snapshot.prices.items():
        if len(snapshot.prices_hist_min[k]) < sliding_elements:
            signal_line[k] = 0
            continue

        data = pd.DataFrame()
        data['Close'] = snapshot.prices_hist_min[k]
        data['EMA12'] = data['Close'].ewm(span=12, adjust=False).mean()
        data['EMA26'] = data['Close'].ewm(span=26, adjust=False).mean()
        data['MACD'] = data['EMA12'] - data['EMA26']
        data['Signal_Line'] = data['MACD'].ewm(span=9, adjust=False).mean()
        last_row = data.iloc[-1]
        second_last_row = data.iloc[-2]

        signal_line[k] = last_row['Signal_Line']

        if second_last_row['MACD'] > second_last_row['Signal_Line'] and last_row['MACD'] < last_row['Signal_Line']:
            if k in macd_threshold.threshold.keys():
                pass
            else:
                macd_threshold.threshold[k] = signal_line[k]

            if abs(signal_line[k]) >= macd_threshold.threshold[k]:
                print('Cross Below Signal Line', '|', k)

        elif second_last_row['MACD'] < second_last_row['Signal_Line'] and last_row['MACD'] > last_row['Signal_Line']:
            if k in macd_threshold.threshold.keys():
                pass
            else:
                macd_threshold.threshold[k] = signal_line[k]

            if abs(signal_line[k]) >= macd_threshold.threshold[k]:
                print('Cross Above Signal Line', '|', k)

        else:
            pass


    with open('macd', 'w') as f:
        print( "===== 0 MACD", file=f)
        for k,v in snapshot.prices.items():
            print( k, "|", snapshot.prices[k], "|", signal_line[k], "|", macd_threshold.threshold[k], file=f)

    with open('macd_threshold.py', 'w') as f:
        print( 'threshold = {}', file=f)
        for k,v in macd_threshold.threshold.items():
            print( 'threshold["' +  k + '"] =', macd_threshold.threshold[k], file=f)

    # break

