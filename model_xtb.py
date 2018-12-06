#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 31 16:35:49 2018

@author: bukowskio
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor, BernoulliRBM
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Input, LSTM, SimpleRNN
from keras.utils import to_categorical
from keras.optimizers import SGD, Adam
from keras.regularizers import l1_l2
from keras.models import Model
from keras import regularizers
from keras import backend as K
import warnings

warnings.filterwarnings('ignore')

df=pd.read_csv("~/Desktop/energy/repo/new_train.csv")


x=df.drop(['rdn','Unnamed: 0'],axis=1)

y=df['rdn']
  
df=df.reset_index() 

n=df.shape[0]
k=x.shape[1]


# tree-based models

model1=RandomForestRegressor(n_estimators=200)

model2=GradientBoostingRegressor(n_estimators=200)


# linear regressions

elastic_ratio=.7

model31=ElasticNet(alpha=.01, l1_ratio=elastic_ratio)
model32=ElasticNet(alpha=.2, l1_ratio=elastic_ratio)
model33=ElasticNet(alpha=.4, l1_ratio=elastic_ratio)
model34=ElasticNet(alpha=.6, l1_ratio=elastic_ratio)


# feed-forward neural networks

activ='tanh'
reg=0.2

model41=MLPRegressor(hidden_layer_sizes=(10,), activation=activ, alpha=reg,solver='sgd',learning_rate_init=0.01, max_iter=2000)
model42=MLPRegressor(hidden_layer_sizes=(15,), activation=activ, alpha=reg,solver='sgd',learning_rate_init=0.01, max_iter=2000)
model43=MLPRegressor(hidden_layer_sizes=(20,), activation=activ, alpha=reg,solver='sgd',learning_rate_init=0.01, max_iter=2000)
model44=MLPRegressor(hidden_layer_sizes=(10,5), activation=activ, alpha=reg,solver='sgd',learning_rate_init=0.01, max_iter=2000)
model45=MLPRegressor(hidden_layer_sizes=(15,5), activation=activ, alpha=reg,solver='sgd',learning_rate_init=0.01, max_iter=2000)
model46=MLPRegressor(hidden_layer_sizes=(20,10), activation=activ, alpha=reg,solver='sgd',learning_rate_init=0.01, max_iter=2000)


# recurrent neural networks

rnn1=Sequential()
rnn1.add(SimpleRNN(30, activation='relu', input_shape=(24,k)))
rnn1.add(Dropout(0.1))
rnn1.add(Dense(50,activation='relu', input_shape=(24,30)))
rnn1.add(Dropout(0.2))
rnn1.add(Dense(24,activation='linear'))
rnn1.compile(loss='mean_squared_error', optimizer='adam')

rnn2=Sequential()
rnn2.add(SimpleRNN(30, activation='sigmoid', input_shape=(24,k)))
rnn2.add(Dropout(0.1))
rnn2.add(Dense(50,activation='relu', input_shape=(24,30)))
rnn2.add(Dropout(0.2))
rnn2.add(Dense(24,activation='linear'))
rnn2.compile(loss='mean_squared_error', optimizer='adam')

rnn3=Sequential()
rnn3.add(SimpleRNN(30, activation='sigmoid', input_shape=(24,k)))
rnn3.add(Dropout(0.1))
rnn3.add(Dense(50,activation='tanh', input_shape=(24,30)))
rnn3.add(Dropout(0.2))
rnn3.add(Dense(24,activation='linear'))
rnn3.compile(loss='mean_squared_error', optimizer='adam')

                   
# prepare test data

df_test=pd.read_csv("~/Desktop/energy/repo/new_test.csv")

xt=df_test.drop(['rdn','Unnamed: 0'],axis=1)

yt=df_test['rdn']

df_test=df_test.reset_index()

                     
# scaling data for neural networks

skaler=StandardScaler()
skaler.fit(x)
x3=skaler.transform(x)

x4=skaler.transform(xt)


# transforming data time-series as input for recurrent neural networks

m=x3.shape[0]/24
n=x3.shape[1]

scaled_y=(y-np.mean(y))/np.std(y)

x_ts=np.array(x3).reshape(m,24,n)
y_ts=np.array(scaled_y).reshape(m,24)

x_ts2=np.array(x4).reshape(df_test.shape[0]/24,24,n)


# launching learning and forecasting process

rf_pred=[]
gbt_pred=[]

log_reg1=[]
log_reg2=[]
log_reg3=[]
log_reg4=[]

mlp41=[]
mlp42=[]
mlp43=[]
mlp44=[]
mlp45=[]
mlp46=[]

rnn1_pred=[]
rnn2_pred=[]
rnn3_pred=[]

n_iter=xt.shape[0]/24

for i in range(n_iter):

    xi=xt[24*i:24*(i+1)]
    yi=yt[24*i:24*(i+1)]
    xi2=skaler.transform(xi)
    
    xi3=xi2.reshape(1,24,n)
    
    model1.fit(x, y)
    model2.fit(x, y)
    
    model31.fit(x,np.log(y+1))
    model32.fit(x,np.log(y+1))
    model33.fit(x,np.log(y+1))
    model34.fit(x,np.log(y+1))
    
    model41.fit(x3,y)
    model42.fit(x3,y)
    model43.fit(x3,y)
    model44.fit(x3,y)
    model45.fit(x3,y)
    model46.fit(x3,y)
    
    rnn1.save_weights('rnn1_w')
    rnn2.save_weights('rnn2_w')
    rnn3.save_weights('rnn3_w')
    
    rnn1.fit(x_ts, y_ts, epochs=2000)
    rnn2.fit(x_ts, y_ts, epochs=2000)
    rnn3.fit(x_ts, y_ts, epochs=2000)
    
    rf_pred=rf_pred+list(model1.predict(xi))
    gbt_pred=gbt_pred+list(model2.predict(xi))
    
    log_reg1=log_reg1+list(np.exp(model31.predict(xi))-1)
    log_reg2=log_reg2+list(np.exp(model32.predict(xi))-1)
    log_reg3=log_reg3+list(np.exp(model33.predict(xi))-1)
    log_reg4=log_reg4+list(np.exp(model34.predict(xi))-1)
    
    mlp41=mlp41+list(model41.predict(xi2))
    mlp42=mlp42+list(model42.predict(xi2))
    mlp43=mlp43+list(model43.predict(xi2))
    mlp44=mlp44+list(model44.predict(xi2))
    mlp45=mlp45+list(model45.predict(xi2))
    mlp46=mlp46+list(model46.predict(xi2))
    
    rnn1_pred=rnn1_pred+list(rnn1.predict(xi3).reshape(24)*np.std(y)+np.mean(y))
    rnn2_pred=rnn2_pred+list(rnn2.predict(xi3).reshape(24)*np.std(y)+np.mean(y))
    rnn3_pred=rnn3_pred+list(rnn3.predict(xi3).reshape(24)*np.std(y)+np.mean(y))
    
    rnn1.load_weights('rnn1_w')
    rnn2.load_weights('rnn2_w')
    rnn3.load_weights('rnn3_w')
    
    x=pd.concat([x,pd.DataFrame(xi)],axis=0)
    x3=skaler.transform(x)
    x_ts=np.array(x3).reshape(x3.shape[0]/24,24,n)

    y=pd.Series(list(y)+list(yi))
    y2=(y-np.mean(y))/np.std(y)
    y_ts=np.array(y2).reshape(y2.shape[0]/24,24)
    

# adding previously predicted ARIMAX forecasts

arimax=pd.read_csv("~/Desktop/energy/arimax_forecasts.csv")

output=pd.DataFrame({'real_price':yt,'arimax':arimax.inv_pred,'rf':rf_pred,'gbt':gbt_pred,'log1':log_reg1,'log2':log_reg2,'log3':log_reg3,'log4':log_reg4,
                      'mlp1':mlp41,'mlp2':mlp42,'mlp3':mlp43,'mlp4':mlp44,'mlp5':mlp45,'mlp6':mlp46,
                      'rnn1':rnn1_pred,'rnn2':rnn2_pred,'rnn3':rnn3_pred})
    
output.to_csv('~/Desktop/energy/final_forecasts.csv')


