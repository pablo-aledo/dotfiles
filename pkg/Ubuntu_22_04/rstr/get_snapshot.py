from selenium import webdriver
from selenium.webdriver.firefox.options import Options
from time import sleep, time
import snapshot
import datetime as dt
from importlib import reload

firefox_options = Options()
firefox_options.headless = True
firefox_options.add_argument("--headless")
driver = webdriver.Firefox(options=firefox_options)

driver.get("http://trade.rs-prime.com")
sleep(5)

element = driver.find_elements("class name", "css-11aywtz")[0]
element.send_keys("")
element = driver.find_elements("class name", "css-11aywtz")[1]
element.send_keys("")
element = [ a for a in driver.find_elements("class name", "css-175oi2r") if a.text == 'LOGIN' ][0]
element.click()
sleep(3)

sliding_elements_min = 60 * 10
sliding_elements_hr  = 24 * 10
sliding_elements_day = 30 * 10

time_sec = int(time())
time_hr = int(time_sec/60/60)
time_day = int(time_sec/60/60/24)
while True:
    prev_time_hr = time_hr
    prev_time_day = time_day
    while int(time()) % 60 != 0:
        sleep(0.3)
    time_sec = int(time())
    time_hr = int(time_sec/60/60)
    time_day = int(time_sec/60/60/24)

    reload(snapshot)

    driver.refresh()
    sleep(5)

    for a in range(len(driver.find_elements("class name", "r-1ikidpy"))):
        element1 = driver.find_elements("class name", "r-1ikidpy")[a]
        element2 = element1.find_element("xpath", "..")
        element3 = element2.find_element("xpath", "..")
        element4 = element3.find_element("xpath", "..")
        list_row = element4.text.split('\n')
        elements_row = [ list_row[i] for i in [0, 3] ]
        if len(elements_row) == 2:
            try:
                snapshot.prices[elements_row[0]] = float(elements_row[1])
            except ValueError:
                pass

    for k,v in snapshot.prices.items():
        if k in snapshot.prices_hist_min.keys():
            pass
        else:
            snapshot.prices_hist_min[k] = list()
        if k in snapshot.dates_hist_min.keys():
            pass
        else:
            snapshot.dates_hist_min[k] = list()

        if k in snapshot.prices_hist_hr.keys():
            pass
        else:
            snapshot.prices_hist_hr[k] = list()
        if k in snapshot.dates_hist_hr.keys():
            pass
        else:
            snapshot.dates_hist_hr[k] = list()

        if k in snapshot.prices_hist_day.keys():
            pass
        else:
            snapshot.prices_hist_day[k] = list()
        if k in snapshot.dates_hist_day.keys():
            pass
        else:
            snapshot.dates_hist_day[k] = list()

        snapshot.prices_hist_min[k].append(snapshot.prices[k])
        while len(snapshot.prices_hist_min[k]) > sliding_elements_min:
            snapshot.prices_hist_min[k] = snapshot.prices_hist_min[k][1:]
        snapshot.dates_hist_min[k].append(time_sec)
        while len(snapshot.dates_hist_min[k]) > sliding_elements_min:
            snapshot.dates_hist_min[k] = snapshot.dates_hist_min[k][1:]

    if time_hr != prev_time_hr:
        for k,v in snapshot.prices.items():
            snapshot.prices_hist_hr[k].append(snapshot.prices[k])
            while len(snapshot.prices_hist_hr[k]) > sliding_elements_hr:
                snapshot.prices_hist_hr[k] = snapshot.prices_hist_hr[k][1:]
            snapshot.dates_hist_hr[k].append(time_sec)
            while len(snapshot.dates_hist_hr[k]) > sliding_elements_hr:
                snapshot.dates_hist_hr[k] = snapshot.dates_hist_hr[k][1:]

    if time_day != prev_time_day:
        for k,v in snapshot.prices.items():
            if k in snapshot.prices_hist_day.keys():
                pass
            else:
                snapshot.prices_hist_day[k] = list()
            if k in snapshot.dates_hist_day.keys():
                pass
            else:
                snapshot.dates_hist_day[k] = list()

            snapshot.prices_hist_day[k].append(snapshot.prices[k])
            while len(snapshot.prices_hist_day[k]) > sliding_elements_day:
                snapshot.prices_hist_day[k] = snapshot.prices_hist_day[k][1:]
            snapshot.dates_hist_day[k].append(time_sec)
            while len(snapshot.dates_hist_day[k]) > sliding_elements_day:
                snapshot.dates_hist_day[k] = snapshot.dates_hist_day[k][1:]

    time_offset = 2
    hod =(dt.datetime.now() + dt.timedelta(hours = time_offset)).hour
    moh = dt.datetime.now().minute

    set_rmholes = set([])
    if ( hod < 15 or ( hod == 15 and moh <= 30 ) ) or ( hod > 21 ):
        for k in set_rmholes:
            if k in snapshot.prices.keys():
                snapshot.prices_hist_min[k] = snapshot.prices_hist_min[k][:-1]
                snapshot.dates_hist_min[k] = snapshot.dates_hist_min[k][:-1]
                if time_hr != prev_time_hr:
                    snapshot.prices_hist_hr[k] = snapshot.prices_hist_hr[k][:-1]
                    snapshot.dates_hist_hr[k] = snapshot.dates_hist_hr[k][:-1]

    set_rmholes = set([])
    if ( hod < 8 or ( hod == 8 and moh == 0 ) ) or ( hod > 21 ) :
        for k in set_rmholes:
            if k in snapshot.prices.keys():
                snapshot.prices_hist_min[k] = snapshot.prices_hist_min[k][:-1]
                snapshot.dates_hist_min[k] = snapshot.dates_hist_min[k][:-1]
                if time_hr != prev_time_hr:
                    snapshot.prices_hist_hr[k] = snapshot.prices_hist_hr[k][:-1]
                    snapshot.dates_hist_hr[k] = snapshot.dates_hist_hr[k][:-1]

    set_rmholes = set([])
    if ( hod < 9 or ( hod == 9 and moh == 0 ) ) or ( hod > 21 ) :
        for k in set_rmholes:
            if k in snapshot.prices.keys():
                snapshot.prices_hist_min[k] = snapshot.prices_hist_min[k][:-1]
                snapshot.dates_hist_min[k] = snapshot.dates_hist_min[k][:-1]
                if time_hr != prev_time_hr:
                    snapshot.prices_hist_hr[k] = snapshot.prices_hist_hr[k][:-1]
                    snapshot.dates_hist_hr[k] = snapshot.dates_hist_hr[k][:-1]

    if dt.datetime.today().weekday() == 4:
        if ( hod > 23 ) :
            for k in snapshot.prices.keys():
                snapshot.prices_hist_min[k] = snapshot.prices_hist_min[k][:-1]
                snapshot.dates_hist_min[k] = snapshot.dates_hist_min[k][:-1]
                if time_hr != prev_time_hr:
                    snapshot.prices_hist_hr[k] = snapshot.prices_hist_hr[k][:-1]
                    snapshot.dates_hist_hr[k] = snapshot.dates_hist_hr[k][:-1]
                if time_day != prev_time_day:
                    snapshot.prices_hist_day[k] = snapshot.prices_hist_day[k][:-1]
                    snapshot.dates_hist_day[k] = snapshot.dates_hist_day[k][:-1]

    if dt.datetime.today().weekday() == 5 or dt.datetime.today().weekday() == 6:
        for k in snapshot.prices.keys():
            snapshot.prices_hist_min[k] = snapshot.prices_hist_min[k][:-1]
            snapshot.dates_hist_min[k] = snapshot.dates_hist_min[k][:-1]
            if time_hr != prev_time_hr:
                snapshot.prices_hist_hr[k] = snapshot.prices_hist_hr[k][:-1]
                snapshot.dates_hist_hr[k] = snapshot.dates_hist_hr[k][:-1]
            if time_day != prev_time_day:
                snapshot.prices_hist_day[k] = snapshot.prices_hist_day[k][:-1]
                snapshot.dates_hist_day[k] = snapshot.dates_hist_day[k][:-1]

    with open('snapshot.py', 'w') as f:
        print( 'prices = {}', file=f)
        print( 'prices_hist_min = {}', file=f)
        print( 'prices_hist_hr = {}', file=f)
        print( 'prices_hist_day = {}', file=f)
        print( 'dates_hist_min = {}', file=f)
        print( 'dates_hist_hr = {}', file=f)
        print( 'dates_hist_day = {}', file=f)
        for k,v in snapshot.prices.items():
            print( 'prices["' +  k + '"] = ', snapshot.prices[k], file=f)
        for k,v in snapshot.prices_hist_min.items():
            print( 'prices_hist_min["' +  k + '"] = ', snapshot.prices_hist_min[k], file=f)
        for k,v in snapshot.prices_hist_hr.items():
            print( 'prices_hist_hr["' +  k + '"] = ', snapshot.prices_hist_hr[k], file=f)
        for k,v in snapshot.prices_hist_day.items():
            print( 'prices_hist_day["' +  k + '"] = ', snapshot.prices_hist_day[k], file=f)
        for k,v in snapshot.dates_hist_min.items():
            print( 'dates_hist_min["' +  k + '"] = ', snapshot.dates_hist_min[k], file=f)
        for k,v in snapshot.dates_hist_hr.items():
            print( 'dates_hist_hr["' +  k + '"] = ', snapshot.dates_hist_hr[k], file=f)
        for k,v in snapshot.dates_hist_day.items():
            print( 'dates_hist_day["' +  k + '"] = ', snapshot.dates_hist_day[k], file=f)

