from importlib import reload
from time import sleep
from time import time
import os
import snapshot

jumps = {}

while True:

    while int(time()) % 60 != 0:
        sleep(0.3)
    sleep(30)

    reload(snapshot)
    os.system('clear')

    for k,v in snapshot.prices.items():
        if len(snapshot.prices_hist_min[k]) > 1:
            jumps[k] = snapshot.prices_hist_min[k][-1] - snapshot.prices_hist_min[k][-2]
        else:
            jumps[k] = 0

    with open('jumps', 'w') as f:
        print( "===== 0 JUMPS", file=f)
        for k,v in snapshot.prices.items():
            if k in jumps.keys():
                print( k, "|", snapshot.prices[k], "|", jumps[k], "|", jumps[k]/snapshot.prices[k], file=f)

    for k,v in snapshot.prices.items():
        if k in jumps.keys() and abs( jumps[k]/snapshot.prices[k] ) > 0.01:
            print( "jump", "|", k, "|", v )

    # break

