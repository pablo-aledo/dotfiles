from selenium import webdriver
from selenium.webdriver.firefox.options import Options
from time import sleep
import snapshot

# firefox_options = Options()
# firefox_options.headless = True
# firefox_options.add_argument("--headless")
# driver = webdriver.Firefox(options=firefox_options)
driver = webdriver.Chrome()

driver.get("http://trade.rs-prime.com")
sleep(5)

element = driver.find_elements("class name", "css-11aywtz")[0]
element.send_keys("")
element = driver.find_elements("class name", "css-11aywtz")[1]
element.send_keys("")
element = [ a for a in driver.find_elements("class name", "css-175oi2r") if a.text == 'LOGIN' ][0]
element.click()
sleep(3)

file1 = open('close_trades', 'r')
lines = [ line.strip() for line in file1.readlines() ]

for close_id in lines:
    while True:
        try:
            ot = [ a for a in driver.find_elements("class name", "css-146c3p1") if a.text == 'OPEN TRADES' ][0]
            ot.click()
            l = driver.find_elements("class name", "sc-aXZVg")[1]
            element = l.find_elements("xpath", "div")[0]
            ots = [ e for e in element.find_elements("xpath", "div") if e.text.split('\n')[2] == close_id ][0]
            ots.click()
            ots = [ e for e in element.find_elements("xpath", "div") if e.text.split('\n')[2] == close_id ][0]
            trd = ots.find_elements("xpath", "div")[1]
            trd.click()
            ots = [ e for e in element.find_elements("xpath", "div") if e.text.split('\n')[2] == close_id ][0]
            clbtn = [ a for a in ots.find_elements("class name", "css-175oi2r") if a.text == "Close Trade" ][0]
            clbtn.click()
            driver.refresh()
        except Exception as e:
            continue
        break


