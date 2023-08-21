import requests
import time
from bs4 import BeautifulSoup
import pandas as pd
from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as ec
from selenium.webdriver.common.by import By

url = "https://www.tokopedia.com/vivo/vivo-iqoo-z7x-8-128-80-watt-6000-mah-snapdragon-695-tropical-blue-bd2b1?extParam=cmp%3D1%26ivf%3Dfalse&src=topads"
driver=webdriver.Chrome()
driver.get(url)
time.sleep(3)
soup=BeautifulSoup(driver.page_source,'html.parser')
contain_file=soup