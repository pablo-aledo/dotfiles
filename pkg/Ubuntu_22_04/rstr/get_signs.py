from time import sleep, time
import pandas as pd
import os
import snapshot
import macd_threshold
from importlib import reload

sign_hist = {}
sliding_elems = 40

while True:

    # while int(time()) % 60 != 0:
        # sleep(0.3)
    # sleep(30)

    reload(snapshot)
    os.system('clear')

    for k,v in snapshot.prices.items():
        sign_hist[k] = [ x - y for x, y in zip(snapshot.prices_hist_min[k][-sliding_elems:-2], snapshot.prices_hist_min[k][-sliding_elems+1:-1]) ]

    with open('signs', 'w') as f:
        for k,v in snapshot.prices.items():
            # print( k, "|", snapshot.prices[k], "|", ''.join([ ("+" if s > 0 else ( "." if s == 0 else "-" )) for s in sign_hist[k] ]), "|", file=f)
            print( k, "|", snapshot.prices[k], "|", ''.join([ ("\033[32m+\033[0m" if s > 0 else ( "\033[33m.\033[0m" if s == 0 else "\033[31m-\033[0m" )) for s in sign_hist[k] ]), "|", file=f)

    break
