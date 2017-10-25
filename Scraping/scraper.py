#!/usr/bin/env python
# -*- coding: utf-8 -*-

from bs4 import BeautifulSoup
import requests
import re
from collections import OrderedDict
import calendar
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from pyvirtualdisplay import Display
import os
import urllib

videodownloader = urllib.URLopener()

chromedriver = '/var/chromedriver/chromedriver'
os.environ['webdriver.chrome.driver'] = chromedriver
display = Display(visible=0 , size=(800, 600))
display.start()
chrome_options = webdriver.ChromeOptions()
chrome_options.add_argument("--mute-audio")
driver = webdriver.Chrome(chromedriver, chrome_options=chrome_options)

import re

urls_pages = ['http://www.metacritic.com/game/pc/sudden-strike-4',]

for url in urls_pages:
    url_parts = url.split('/')
    filename = url_parts[len(url_parts)-1]
    try:
        driver.get(url)
    except Exception as e:
        print 'Failed to get movie'
    try:
        trailer_url = driver.find_element_by_id('video_holder_wrapper').get_attribute('data-mcvideourl')
        metascore = driver.find_elements_by_xpath('//*[@id="main"]/div/div[3]/div/div/div[2]/div[1]/div[1]/div/div/a/div/span')[0].get_attribute('innerHTML').strip()
        userscore = driver.find_elements_by_xpath('//*[@id="main"]/div/div[3]/div/div/div[2]/div[1]/div[2]/div[1]/div/a/div')[0].get_attribute('innerHTML').strip()
        summary = driver.find_elements_by_xpath('//*[@id="main"]/div/div[3]/div/div/div[2]/div[2]/div[1]/ul/li/span[2]')[0].text.strip().replace('\n', ' ')[:-8]
        age_rating = driver.find_elements_by_xpath('//*[@id="main"]/div/div[3]/div/div/div[2]/div[2]/div[2]/ul/li[4]/span[2]')[0].get_attribute('innerHTML').strip()
        developer = driver.find_elements_by_xpath('//*[@id="main"]/div/div[3]/div/div/div[2]/div[2]/div[2]/ul/li[1]/span[2]')[0].get_attribute('innerHTML').strip()
        print(developer, age_rating, metascore, userscore, summary, trailer_url)
        videodownloader.retrieve(trailer_url, filename + '.mp4')
        textfile = open(filename + '.txt', 'w')
        textfile.write(metascore + ';;;' + userscore + ';;;' + summary + ';;;' + developer + ';;;' + age_rating + '\n')
        textfile.close()
    except Exception as e:
        print 'Failed to get game details for', url, 'due to error: ', e

display.stop()
driver.quit()
