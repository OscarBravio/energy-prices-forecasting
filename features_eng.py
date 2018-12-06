#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 31 16:35:49 2018

@author: bukowskio
"""

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Input, LSTM, SimpleRNN
from keras.utils import to_categorical
from keras.optimizers import SGD, Adam
from keras.models import Model
from keras import regularizers
from keras import backend as K


df=pd.read_csv("~/Desktop/energy/train_data.csv")


# fundamental features

x=df[['rdn','deman','supply','wind_prod','reserve']]

y=df['rdn']
  
df=df.reset_index() 

n=df.shape[0]


# time-series features for dimensionality reduction

z=df[['cro2','cro3','cro4','cro5','ma_cro2','ma_cro3','sd_cro1','sd_cro2','ratio_cro1','ratio_cro2',
        'fix2','fix3','fix4','fix5','fix6','ma_fix2','ma_fix3','ma_fix4','sd_fix1','sd_fix2',
        'se2','se3','se4']]


# scaling data

model0=StandardScaler()
model0.fit(z)

new_x=model0.transform(z)


# train-test split

new_train, new_test, new_train2, new_test2 = train_test_split(new_x, df['rdn'], test_size=0.2)


# fitting PCA and few different autoencoders

PCA_model=PCA(n_components=15)
PCA_model.fit(new_train)

n_col=z.shape[1]

K.set_learning_phase(1)

ae1=Sequential()
ae1.add(Dense(15, activation='relu',input_dim=n_col))
ae1.add(Dense(n_col, activation='linear',input_dim=15))

ae1.compile(loss='mean_squared_error', optimizer='adam')
ae1.fit(np.array(new_train), np.array(new_train), epochs=500, batch_size=32)


ae2=Sequential()
ae2.add(Dense(15, activation='tanh',input_dim=n_col))
ae2.add(Dense(n_col, activation='linear',input_dim=15))

ae2.compile(loss='mean_squared_error', optimizer='adam')
ae2.fit(np.array(new_train), np.array(new_train), epochs=500, batch_size=32)


ae3=Sequential()
ae3.add(Dense(15, activation='tanh',input_dim=n_col))
ae3.add(Dropout(0.1))
ae3.add(Dense(n_col, activation='linear',input_dim=15))

ae3.compile(loss='mean_squared_error', optimizer='adam')
ae3.fit(np.array(new_train), np.array(new_train), epochs=1000, batch_size=32)


ae4=Sequential()
ae4.add(Dense(30, activation='tanh',input_dim=n_col))
ae4.add(Dropout(0.2))
ae4.add(Dense(15, activation='tanh',input_dim=30))
ae4.add(Dropout(0.1))
ae4.add(Dense(n_col, activation='linear',input_dim=15))

adam1=Adam(lr=.00, decay=.05)
ae4.compile(loss='mean_squared_error', optimizer='adam')
ae4.fit(np.array(new_train), np.array(new_train), epochs=2000, batch_size=32)


# data transformation

n_row=new_test.shape[0]

get_layer1 = K.function([ae1.layers[0].input, K.learning_phase()],[ae1.layers[0].output])
layer_output11 = get_layer1([new_train])
layer_output12 = get_layer1([new_test])

get_layer2 = K.function([ae2.layers[0].input, K.learning_phase()],[ae2.layers[0].output])
layer_output21 = get_layer2([new_train])
layer_output22 = get_layer2([new_test])

get_layer3 = K.function([ae3.layers[0].input, K.learning_phase()],[ae3.layers[0].output])
layer_output31 = get_layer3([new_train])
layer_output32 = get_layer3([new_test])

get_layer4 = K.function([ae4.layers[0].input, K.learning_phase()],[ae4.layers[2].output])
layer_output41 = get_layer4([new_train])
layer_output42 = get_layer4([new_test])

h0_train=PCA_model.transform(new_train)
h1_train=np.array(layer_output11).reshape(new_train.shape[0],15)
h2_train=np.array(layer_output21).reshape(new_train.shape[0],15)
h3_train=np.array(layer_output31).reshape(new_train.shape[0],15)
h4_train=np.array(layer_output41).reshape(new_train.shape[0],15)

h0_test=PCA_model.transform(new_test)
h1_test=np.array(layer_output12).reshape(new_test.shape[0],15)
h2_test=np.array(layer_output22).reshape(new_test.shape[0],15)
h3_test=np.array(layer_output32).reshape(new_test.shape[0],15)
h4_test=np.array(layer_output42).reshape(new_test.shape[0],15)


# building linear regression model on original and transformed data and compare accuracy of predictions on dataset using R^2 measure

lin_reg=LinearRegression()

lin_reg.fit(new_train, np.log(new_train2+1))
print(lin_reg.score(new_test, np.log(new_test2+1)))

lin_reg.fit(h0_train, np.log(new_train2+1))
print(lin_reg.score(h0_test, np.log(new_test2+1)))

lin_reg.fit(h1_train, np.log(new_train2+1))
print(lin_reg.score(h1_test, np.log(new_test2+1)))

lin_reg.fit(h2_train, np.log(new_train2+1))
print(lin_reg.score(h2_test, np.log(new_test2+1)))

lin_reg.fit(h3_train, np.log(new_train2+1))
print(lin_reg.score(h3_test, np.log(new_test2+1)))

lin_reg.fit(h4_train, np.log(new_train2+1))
print(lin_reg.score(h4_test, np.log(new_test2+1)))


# Data transformed by autoencoders with "tanh" activation function give better predictions so they are going to be overwrite and used in further mdoels

new_z=np.array(get_layer3([new_x])).reshape(new_x.shape[0],15)

x2=pd.concat([x,pd.DataFrame(new_z)],axis=1)

x2.to_csv('~/Desktop/energy/repo/new_train.csv')


# transforming and overwriting test data (we used previosuly pretrained standard scaler and autoencoeders)                    
                     
df_test=pd.read_csv("~/Desktop/energy/test_data.csv")

zt=df_test[['cro2','cro3','cro4','cro5','ma_cro2','ma_cro3','sd_cro1','sd_cro2','ratio_cro1','ratio_cro2',
        'fix2','fix3','fix4','fix5','fix6','ma_fix2','ma_fix3','ma_fix4','sd_fix1','sd_fix2',
        'se2','se3','se4']]

xt=df_test[['rdn','deman','supply','wind_prod','reserve']]

new_zt=model0.transform(zt)

new_zt2=np.array(get_layer3([new_zt])).reshape(new_zt.shape[0],15)

new_xt=pd.concat([xt,pd.DataFrame(new_zt2)],axis=1)

new_xt.to_csv('~/Desktop/energy/repo/new_test.csv')