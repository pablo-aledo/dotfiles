from time import sleep
from scipy import stats
import os
from time import time
import snapshot
from importlib import reload

sliding_elements = 5
slope = {}
average = {}

while True:

    while int(time()) % 60 != 0:
        sleep(0.3)
    sleep(30)

    reload(snapshot)
    os.system('clear')

    for k,v in snapshot.prices.items():
        if len(snapshot.prices_hist_min[k]) > sliding_elements:
            slope[k], _, _, _, _ = stats.linregress(list(range(0,len(snapshot.prices_hist_min[k]))),snapshot.prices_hist_min[k])
            average[k] = sum(snapshot.prices_hist_min[k])/len(snapshot.prices_hist_min[k])
        else:
            slope[k] = 0
            average[k] = 1

    with open('slopes', 'w') as f:
        print( "===== 0 SLOPES", file=f)
        for k,v in snapshot.prices.items():
            print( k, "|", snapshot.prices[k], "|", slope[k]/average[k], file=f)

    for k,v in snapshot.prices.items():
        if abs(slope[k]/average[k]) > 0.001:
            print("slope", k, slope[k]/average[k])

    # break
