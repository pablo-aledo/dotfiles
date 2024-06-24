from time import sleep, time, strftime, localtime
import pandas as pd
import os
import snapshot
import macd_threshold
from importlib import reload
from more_itertools import windowed


sign_hist = {}
sliding_elems = 40

while True:

    # while int(time()) % 60 != 0:
        # sleep(0.3)
    # sleep(30)

    reload(snapshot)
    os.system('clear')

    for k,v in snapshot.prices.items():
        sign_hist[k] = [ y - x for x, y in windowed(snapshot.prices_hist_min[k],2) ][-sliding_elems:-1]

    with open('signs', 'w') as f:
        for k,v in snapshot.prices.items():
            print( k, "|",
                    snapshot.prices[k], "|",
                    ''.join([ ("\033[32m+\033[0m" if s > 0 else ( "\033[33m.\033[0m" if s == 0 else "\033[31m-\033[0m" )) for s in sign_hist[k] ]), "|",
                    strftime('%Y-%m-%d %H:%M:%S', localtime(snapshot.dates_hist_min[k][-1])),
                    file=f)

    break
