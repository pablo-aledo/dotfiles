import snapshot
from itertools import product
import joblib
import numpy as np
from importlib import reload
import os
import operator

sliding_elements_hr = 5
instruments = {}
n_simulations = 1000000

while True:

    # while int(time()) % 60 != 0:
        # sleep(0.3)
    # sleep(30)

    reload(snapshot)
    os.system('clear')

    ######
    hitmiss = {}
    for instr in sorted(instruments):
        hitmiss[instr] = list()
    for offset in range(-6, 0):
        prices_hist_aux = {}
        for instr in instruments:
            prices_hist_aux[instr] = snapshot.prices_hist_hr[instr][:offset]
        all_deltas_hr = []
        for delta, instr in product(list(range(-1,-sliding_elements_hr,-1)), sorted(instruments)):
            if len(prices_hist_aux[instr]) < sliding_elements_hr:
                continue
            delta = (prices_hist_aux[instr][delta] - prices_hist_aux[instr][delta - 1])/prices_hist_aux[instr][delta - 1]
            all_deltas_hr.append(delta)
        x = [ d for d in all_deltas_hr ]
        lr = joblib.load("linear_hourly.pkl")
        y = lr.predict([x])[0]
        n = 0
        oldpred = {}
        actual_movement = {}
        for instr in sorted(instruments):
            oldpred[instr] = "+" if y[n] > 0 else "-"
            n = n+1
        for instr in sorted(instruments):
            if prices_hist_aux[instr][-1] > prices_hist_aux[instr][-2]:
                actual_movement[instr] = "+"
            else:
                actual_movement[instr] = "-"
        for instr in sorted(instruments):
            if oldpred[instr] == actual_movement[instr]:
                hitmiss[instr].append("+")
            else:
                hitmiss[instr].append("-")
    ######

    all_deltas_hr = []
    for delta, instr in product(list(range(-1,-sliding_elements_hr,-1)), sorted(instruments)):
        if len(snapshot.prices_hist_hr[instr]) < sliding_elements_hr:
            continue
        delta = (snapshot.prices_hist_hr[instr][delta] - snapshot.prices_hist_hr[instr][delta - 1])/snapshot.prices_hist_hr[instr][delta - 1]
        all_deltas_hr.append(delta)

    x = [ d for d in all_deltas_hr ]

    lr = joblib.load("linear_hourly.pkl")
    y = lr.predict([x])[0]

    stdv = [ np.std(snapshot.prices_hist_hr[instr]) for delta, instr in product(list(range(-1,-sliding_elements_hr,-1)), sorted(instruments)) ]

    xs = np.random.normal(x, stdv, (n_simulations, len(instruments) * (sliding_elements_hr-1)))
    ys = lr.predict(xs)
    stds = [ x + 0.000001 for x in np.std(ys, axis = 0) ]

    elems = set()
    n = 0
    for instr in sorted(instruments):
        elems.add( (instr, y[n], stds[n], y[n]/stds[n], ''.join(hitmiss[instr]), operator.countOf(hitmiss[instr], "+")) )
        n = n+1

    print("===== Gain")
    for elem in sorted(elems, key=lambda e:abs(e[1])):
        print(elem[0], "|", elem[1], "|", elem[2], "|", elem[3], "|", elem[4], "|", elem[5], )

    print("===== Stdv")
    for elem in sorted(elems, key=lambda e:1/abs(e[2])):
        print(elem[0], "|", elem[1], "|", elem[2], "|", elem[3], "|", elem[4], "|", elem[5], )

    print("===== Combined")
    for elem in sorted(elems, key=lambda e:abs(e[3])):
        print(elem[0], "|", elem[1], "|", elem[2], "|", elem[3], "|", elem[4], "|", elem[5], )

    print("===== Hits")
    for elem in sorted(elems, key=lambda e:e[5]):
        print(elem[0], "|", elem[1], "|", elem[2], "|", elem[3], "|", elem[4], "|", elem[5], )

    break
