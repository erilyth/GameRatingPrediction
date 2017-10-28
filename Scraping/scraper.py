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

chromedriver = '/var/chromedriver/chromedriver'
os.environ['webdriver.chrome.driver'] = chromedriver
display = Display(visible=0 , size=(800, 600))
display.start()
chrome_options = webdriver.ChromeOptions()
chrome_options.add_argument("--mute-audio")
driver = webdriver.Chrome(chromedriver, chrome_options=chrome_options)

import re

# For the website: http://www.metacritic.com/

genres = ['action', 'adventure', 'fighting', 'first-person', 'flight', 'party', 'platformer', 'puzzle', 'racing', 'real-time', 'role-playing', 'simulation', 'sports', 'strategy', 'third-person', 'turn-based', 'wargame', 'wrestling']
pages = [290, 61, 18, 53, 7, 4, 36, 17, 45, 26, 67, 38, 68, 66, 30, 22, 2, 4]

def create_url_list():
    for genre_idx in range(len(genres)):
        genre_file = open('urls/' + genres[genre_idx] + '.txt', 'w')
        for page_idx in range(pages[genre_idx]):
            print "Genre: " + genres[genre_idx] + ", Page: " + str(page_idx)
            current_url = 'http://www.metacritic.com/browse/games/genre/date/' + genres[genre_idx] + '/all/?page=' + str(page_idx)
            driver.get(current_url)
            url_class = driver.find_elements_by_class_name('product_title')
            cnt = 0
            for element in url_class:
                cnt += 1
                if cnt > 30:
                    break
                url = element.find_elements_by_tag_name('a')[0].get_attribute('href')
                genre_file.write(url + '\n')
        genre_file.close()


def scrape_url_list():
    for genre_idx in range(len(genres)):
        url_pages = open('urls/' + genres[genre_idx] + '.txt', 'r').readlines()
        #url_pages = ['http://www.metacritic.com/game/playstation/tekken-2',] #Example
        for url in url_pages:
            url = url.strip()
            url_parts = url.split('/')
            filename = 'dataset/' + genres[genre_idx] + '/' + url_parts[len(url_parts)-1]
            try:
                driver.get(url)
            except Exception as e:
                print 'Failed to fetch game page'
            try:
                trailer_url = driver.find_element_by_id('video_holder_wrapper').get_attribute('data-mcvideourl')
                metascore = driver.find_elements_by_xpath('//*[@id="main"]/div/div[3]/div/div/div[2]/div[1]/div[1]/div/div/a/div/span')[0].get_attribute('innerHTML').strip()
                userscore = driver.find_elements_by_xpath('//*[@id="main"]/div/div[3]/div/div/div[2]/div[1]/div[2]/div[1]/div/a/div')[0].get_attribute('innerHTML').strip()
                summary = driver.find_elements_by_xpath('//*[@id="main"]/div/div[3]/div/div/div[2]/div[2]/div[1]/ul/li/span[2]')[0].text.strip().replace('\n', ' ')[:-8]
                age_rating = driver.find_elements_by_class_name('product_rating')[0].find_elements_by_class_name('data')[0].text.strip()
                developer = driver.find_elements_by_class_name('developer')[0].find_elements_by_class_name('data')[0].text.strip()
                print(developer, age_rating, metascore, userscore, summary, trailer_url)
                os.system('wget ' + trailer_url + ' --proxy=off -O ' + filename + '-old.mp4 -o templog.txt')
                os.system("ffmpeg -i " + filename + "-old.mp4" + " -s 640x480 -vcodec h264 -acodec mp2 -t 00:03:00 " + filename + ".mp4 -y ")
                os.system("rm " + filename + "-old.mp4")
                textfile = open(filename + '.txt', 'w')
                textfile.write(metascore + ';;;' + userscore + ';;;' + summary + ';;;' + developer + ';;;' + age_rating + ';;;' + genres[genre_idx] + ';;;' + trailer_url + '\n')
                textfile.close()
            except Exception as e:
                print 'Failed to get game details for', url, 'due to error: ', e


#create_url_list()
scrape_url_list()

display.stop()
driver.quit()
