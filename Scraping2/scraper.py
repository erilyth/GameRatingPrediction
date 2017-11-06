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
    positive_critic = open('positivecritic.txt', 'a')
    neutral_critic = open('neutralcritic.txt', 'a')
    negative_critic = open('negativecritic.txt', 'a')
    positive_user = open('positiveuser.txt', 'a')
    neutral_user = open('neutraluser.txt', 'a')
    negative_user = open('negativeuser.txt', 'a')
    for genre_idx in range(len(genres)):
        url_pages = open('urls/' + genres[genre_idx] + '.txt', 'r').readlines()
        #url_pages = ['http://www.metacritic.com/game/playstation/tekken-2',] #Example
        for url in url_pages:
            url = url.strip()
            print url
            try:
                driver.get(url + '/critic-reviews')
            except Exception as e:
                print 'Failed to fetch game page'
            try:
                critic_reviews = driver.find_elements_by_class_name('critic_reviews_module')[0]
                critic_review_list = critic_reviews.find_elements_by_class_name('review_section')
                for review in critic_review_list:
                    if len(review.find_elements_by_class_name('review_body')) > 0:
                        description = review.find_elements_by_class_name('review_body')[0].get_attribute('innerHTML').encode('utf8').strip()
                        score = review.find_elements_by_class_name('metascore_w')[0].get_attribute('innerHTML').encode('utf8').strip()
                        description = description.replace('\n', ' ')
                        score = int(float(score))
                        if score > 75:
                            positive_critic.write(description + '\n')
                        elif score > 50:
                            neutral_critic.write(description + '\n')
                        else:
                            negative_critic.write(description + '\n')
                        print description, score, 'Critic'
            except Exception as e:
                print 'Failed to get game details for', url, 'due to error: ', e
            
            try:
                driver.get(url + '/user-reviews')
            except Exception as e:
                print 'Failed to fetch game page'
            try:
                user_reviews = driver.find_elements_by_class_name('user_reviews')[0]
                user_review_list = user_reviews.find_elements_by_class_name('review_section')
                for review_idx in range(len(user_review_list)):
                    if len(user_review_list[review_idx].find_elements_by_class_name('review_body')) > 0:
                        score = user_review_list[review_idx].find_elements_by_class_name('metascore_w')[0].get_attribute('innerHTML').encode('utf8').strip()
                        description = user_review_list[review_idx].find_elements_by_class_name('review_body')[0]
                        if len(description.find_elements_by_class_name('blurb_collapsed')) == 0:
                            description = description.find_elements_by_tag_name('span')[0].text.strip().encode('utf8')
                        else:
                            description = description.find_elements_by_class_name('blurb_collapsed')[0].text.strip().encode('utf8')[:-8]
                        description = description.replace('\n', ' ')
                        score = float(score)
                        if score > 7.5:
                            positive_user.write(description + '\n')
                        elif score > 5.0:
                            neutral_user.write(description + '\n')
                        else:
                            negative_user.write(description + '\n')  
                        print description, score, 'User'
            except Exception as e:
                print 'Failed to get game details for', url, 'due to error: ', e
    positive_critic.close()
    negative_critic.close()
    neutral_critic.close()
    positive_user.close()
    negative_user.close()
    neutral_user.close()


#create_url_list()
scrape_url_list()

display.stop()
driver.quit()
