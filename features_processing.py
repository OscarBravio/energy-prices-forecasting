#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  4 16:12:05 2019

@author: bukowskio
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
import datetime
import dash_core_components as dcc
import dash_html_components as html
import numpy as np
import pandas as pd
import plotly.graph_objs as go
from IPython.display import Markdown
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, precision_recall_curve
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Input, LSTM, SimpleRNN
from keras.utils import to_categorical
from keras.optimizers import SGD, Adam
from keras.models import Model
from keras import regularizers
from keras import backend as K



df2=pd.read_csv("~/Oskar/energy-prices-forecasting/dane.csv")

dates=df2['date'].apply(lambda x: datetime.datetime.strptime(x, '%d/%m/%Y'))

df2['weekdays']=dates.apply(lambda x: x.weekday())

df2['month']=dates.apply(lambda x: x.month)



#df2=df2.fillna(0)

#df2['year']=df2['Data'].astype('str').str.slice(start=0,stop=4)
#df2['month']=df2['Data'].astype('str').str.slice(start=4,stop=6)
#df2['day']=df2['Data'].astype('str').str.slice(start=6,stop=8)

# data pre-processing

df2['lag_rdn_24']=df2['rdn'].shift(24)
df2['lag_rdn_25']=df2['rdn'].shift(25)
df2['lag_rdn_26']=df2['rdn'].shift(26)
df2['lag_rdn_48']=df2['rdn'].shift(48)
df2['lag_rdn_72']=df2['rdn'].shift(72)
df2['lag_rdn_96']=df2['rdn'].shift(96)
df2['lag_rdn_168']=df2['rdn'].shift(168)

df2['lag_cro_72']=df2['cro'].shift(72)
df2['lag_cro_72']=df2['cro'].shift(73)
df2['lag_cro_72']=df2['cro'].shift(74)
df2['lag_cro_72']=df2['cro'].shift(96)
df2['lag_cro_72']=df2['cro'].shift(168)

df2['lag_se_24']=df2['se'].shift(24)
df2['lag_se_48']=df2['se'].shift(28)
df2['lag_se_72']=df2['se'].shift(72)

df2['ma_rdn_24']=df2['lag_rdn_24'].rolling(24).mean()
df2['ma_rdn_48']=df2['lag_rdn_24'].rolling(48).mean()
df2['ma_rdn_72']=df2['lag_rdn_24'].rolling(72).mean()
df2['ma_rdn_96']=df2['lag_rdn_24'].rolling(96).mean()
df2['ma_rdn_168']=df2['lag_rdn_24'].rolling(168).mean()

df2['med_rdn_24']=df2['lag_rdn_24'].rolling(24).median()
df2['med_rdn_48']=df2['lag_rdn_24'].rolling(48).median()
df2['med_rdn_72']=df2['lag_rdn_24'].rolling(72).median()
df2['med_rdn_96']=df2['lag_rdn_24'].rolling(96).median()
df2['med_rdn_168']=df2['lag_rdn_24'].rolling(168).median()

df2['sd_rdn_24']=df2['lag_rdn_24'].rolling(24).std()
df2['sd_rdn_48']=df2['lag_rdn_24'].rolling(48).std()
df2['sd_rdn_72']=df2['lag_rdn_24'].rolling(72).std()
df2['sd_rdn_96']=df2['lag_rdn_24'].rolling(96).std()
df2['sd_rdn_168']=df2['lag_rdn_24'].rolling(168).std()

df2['ma_cro_24']=df2['lag_cro_72'].rolling(24).mean()
df2['ma_cro_48']=df2['lag_cro_72'].rolling(48).mean()
df2['ma_cro_72']=df2['lag_cro_72'].rolling(72).mean()
df2['ma_cro_96']=df2['lag_cro_72'].rolling(96).mean()
df2['ma_cro_168']=df2['lag_cro_72'].rolling(168).mean()

df2['med_cro_24']=df2['lag_cro_72'].rolling(24).median()
df2['med_cro_48']=df2['lag_cro_72'].rolling(48).median()
df2['med_cro_72']=df2['lag_cro_72'].rolling(72).median()
df2['med_cro_96']=df2['lag_cro_72'].rolling(96).median()
df2['med_cro_168']=df2['lag_cro_72'].rolling(168).median()

df2['sd_cro_24']=df2['lag_cro_72'].rolling(24).std()
df2['sd_cro_48']=df2['lag_cro_72'].rolling(48).std()
df2['sd_cro_72']=df2['lag_cro_72'].rolling(72).std()
df2['sd_cro_96']=df2['lag_cro_72'].rolling(96).std()
df2['sd_cro_168']=df2['lag_cro_72'].rolling(168).std()

df2.isnull().sum()

df2=df2.fillna(-100)
df2=df2.reset_index()

df3=df2[360:]

unique_days=df3['date'].unique()
n_days=len(unique_days)

days_dict=dict(zip(list(unique_days), range(n_days)))

df3['day']=df3['date'].map(lambda x: days_dict[x])
#test_df=df2[480:]

#train_df.to_csv('train_df.csv')
df3.to_csv('new_df.csv')


# k-means models to find stationarity

ts_df=['lag_rdn_24','lag_rdn_25','lag_rdn_26','lag_rdn_48','lag_rdn_72','lag_rdn_96','lag_rdn_168','lag_cro_72','lag_se_24','lag_se_48','lag_se_72',
 'ma_rdn_24','ma_rdn_48','ma_rdn_72','ma_rdn_96','ma_rdn_168','med_rdn_24','med_rdn_48','med_rdn_72','med_rdn_96','med_rdn_168','sd_rdn_24','sd_rdn_48',
 'sd_rdn_72','sd_rdn_96','sd_rdn_168','ma_cro_24','ma_cro_48','ma_cro_72','ma_cro_96','ma_cro_168','med_cro_24','med_cro_48','med_cro_72','med_cro_96',
 'med_cro_168','sd_cro_24','sd_cro_48','sd_cro_72','sd_cro_96','sd_cro_168']

ts_df=[
 'ma_rdn_24','ma_rdn_48','ma_rdn_72','ma_rdn_96','ma_rdn_168','med_rdn_24','med_rdn_48','med_rdn_72','med_rdn_96','med_rdn_168','sd_rdn_24','sd_rdn_48',
 'sd_rdn_72','sd_rdn_96','sd_rdn_168','ma_cro_24','ma_cro_48','ma_cro_72','ma_cro_96','ma_cro_168','med_cro_24','med_cro_48','med_cro_72','med_cro_96',
 'med_cro_168','sd_cro_24','sd_cro_48','sd_cro_72','sd_cro_96','sd_cro_168']

week_ma=df2[['ma_rdn_168','ma_cro_168']]

