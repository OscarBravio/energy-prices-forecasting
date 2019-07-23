#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 14:33:01 2019

@author: bukowskio
"""

import requests
import datetime
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup



def get_month(date):
    
    x=str(date.month)
    
    if len(x)<2:
        
        x='0'+x
    
    return(x)


def get_day(date):
    
    x=str(date.day)
    
    if len(x)<2:
        
        x='0'+x
    
    return(x)


date1=datetime.datetime.utcnow().date()

date2=date1-datetime.timedelta(25)

delta=date1-date2

rdn=[]
rds=[]
date=[]
hour=[]


while date2<=date1:
    
    
    print(date2)
    
    #date=date+list(np.repeat(date2,24))
    hour=hour+list(range(1,25))
    
    str_date=get_day(date2)+'-'+get_month(date2)+'-'+str(date2.year)
    
    URL="https://tge.pl/energia-elektryczna-rdn?dateShow="+str_date+"&dateAction=prev"

    #URL = "https://tge.pl/energia-elektryczna-rdn?dateShow=08-07-2019&dateAction=prev"
    r = requests.get(URL) 
    
    print(r.content) 
    
    soup = BeautifulSoup(r.content, 'html5lib')
    print(soup.prettify()) 
    
    
    # getting all tables from the website
    
    tables=[]
    
    for row in soup.find_all('table'):
        
        #print row
        tables.append(row)
    
    
    # getting prices from the 3rd table
    
    for row in tables[2].find('tbody').find_all('tr'):
        
        cols=row.find_all('td')
        p1=cols[1].text.strip()
        p2=cols[3].text.strip()
        p1=p1.replace(',','.')
        p2=p2.replace(',','.')
        rdn.append(float(p1))
        rds.append(float(p2))
        
    date2=date2+datetime.timedelta(1)
    
    date=date+list(np.repeat(date2,24))
        

output=pd.DataFrame({'date':date, 'hour':hour, 'rdn':rdn, 'rds':rds})

output.to_csv('rdn_prices.csv')


# scraping el spot prices


