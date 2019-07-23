#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 31 16:35:49 2018

@author: bukowskio
"""

import pandas as pd
import numpy as np
import time
import gc
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import GradientBoostingRegressor
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Input, LSTM, SimpleRNN
from keras.utils import to_categorical
from keras.optimizers import SGD, Adam
from keras.models import Model
from keras.regularizers import l1, l2, l1_l2
from keras import backend as K

#import lightgbm as lgb



def sample_neurons(n, max_neurons):
    
    z=np.random.uniform(5,max_neurons,size=n).astype('int')
    return(z)


def mutate_neurons(neurons):
    
    z=neurons+np.random.normal(0,5,size=len(neurons)).astype('int')
    y=np.where(z>0,z,neurons)
    
    return(y)


def sample_activs(n):
    
    z=np.random.choice(['tanh','relu'],n)
    return(z)


def sample_drop(n):
    
    z=np.round(np.random.exponential(size=n)/10.,1)
    return(z)


def sample_reg(n):
    
    z=np.random.exponential(size=n)/10.
    p=np.random.uniform(size=n)
    
    y=np.where(p>.5,0,z)
    
    return(y)


def mutate_reg(reg):
    
    z=reg+np.random.uniform(-.1,.1,size=len(reg))
    y=np.where(z>0,z,0)
    
    return(y)


def sample_iters(n):
    
    z=np.random.uniform(50,500,size=n).astype('int')
    return(z)


def mutate_iters(iters):
    
    z=iters+np.random.normal(0,100,size=len(iters)).astype('int')
    y=np.where(z>0,z,iters)
    
    return(y)


def mse_error(y_pred, y_real):
    
    z=(y_pred-y_real)**2
    return(z)


def ae_model(x, ae_neurons, l1_reg, l2_reg, drop_out, n_iter):
    
    ae=Sequential()
    ae.add(Dense(ae_neurons, activation='tanh', activity_regularizer=l1_l2(l1=l1_reg, l2=l2_reg), input_dim=x.shape[1]))
    ae.add(Dropout(drop_out))
    ae.add(Dense(x.shape[1], activation='linear',input_dim=ae_neurons))
    
    ae.compile(loss='mean_squared_error', optimizer='adam')
    ae.fit(np.array(x), np.array(x), epochs=n_iter, batch_size=32)
    
    return(ae)



def tuning_job(z_train, z_test, x_train, x_test, y_train, y_test,
               ae_neurons, ae_l1, ae_l2, ae_drops, nn_iter1,
               n_trees, lr, depth):
    
    ae=ae_model(z_train, ae_neurons, ae_l1, ae_l2, ae_drops, nn_iter1)
        
    time.sleep(1)
        
    get_layer = K.function([ae.layers[0].input, K.learning_phase()],[ae.layers[0].output])
    h = np.array(get_layer([z_train])).reshape(z_train.shape[0], ae_neurons)
        
    new_x=pd.concat([pd.DataFrame(x_train),pd.DataFrame(h)],axis=1)
    
    gbt=GradientBoostingRegressor(n_estimators=n_trees, learning_rate=lr, max_depth=depth)
    gbt.fit(new_x, y_train)
        
    time.sleep(1)
        
    h_test=np.array(get_layer([z_test])).reshape(z_test.shape[0], ae_neurons)
        
    new_x=pd.concat([pd.DataFrame(x_test),pd.DataFrame(h_test)],axis=1)
        
    pred=gbt.predict(new_x)
    error=mse_error(pred,y_test).mean()
    
    del ae, gbt
    gc.collect()
        
    return(error)



def random_tuning_job(z_train, z_test, x_train, x_test, y_train, y_test, n):

    ae_neurons=sample_neurons(n,30)
    ae_l1=sample_reg(n)
    ae_l2=sample_reg(n)
    ae_drops=sample_drop(n)
    nn_iter1=np.random.uniform(800,3000,size=n).astype('int')

    evaluate=[]

    for i in range(n):

        error=tuning_job(z_train, z_test, x_train, x_test, y_train, y_test,
               ae_neurons[i], ae_l1[i], ae_l2[i], ae_drops[i], nn_iter1[i],
               100, 0.1, 3)

        evaluate.append(error)

        del error
        gc.collect()

    output=pd.DataFrame({'ae_neurons':ae_neurons, 'ae_l1':ae_l1, 'ae_l2':ae_l2, 'ae_drops': ae_drops, 'nn_iter1':nn_iter1, 'mse_error':evaluate})

    output['method']=np.repeat('random',output.shape[0])

    return(output)


def evolution_tuning_job(z_train, z_test, x_train, x_test, y_train, y_test, good_params):

    ae_neurons=mutate_neurons(good_params['ae_neurons'])
    ae_l1=mutate_reg(good_params['ae_l1'])
    ae_l2=mutate_reg(good_params['ae_l2'])
    ae_drops=mutate_reg(good_params['ae_drops'])
    nn_iter1=mutate_iters(good_params['nn_iter1'])

    evaluate=[]

    n=good_params.shape[0]

    for i in range(n):

        error=tuning_job(z_train, z_test, x_train, x_test, y_train, y_test,
               ae_neurons[i], ae_l1[i], ae_l2[i], ae_drops[i], nn_iter1[i],
               100, 0.1, 3)

        evaluate.append(error)

        del error
        gc.collect()


    output=pd.DataFrame({'ae_neurons':ae_neurons, 'ae_l1':ae_l1, 'ae_l2':ae_l2, 'ae_drops': ae_drops, 'nn_iter1':nn_iter1, 'mse_error':evaluate})

    output['method']=np.repeat('optimize - evolution',output.shape[0])

    return(output)



# sampling neurons 


train_df=pd.read_csv("~/Oskar/energy-prices-forecasting/train_df.csv")
test_df=pd.read_csv("~/Oskar/energy-prices-forecasting/test_df.csv")


# choosing fundamental features

x_cols=['godz','deman','supply','wind_prod','reserve','weekend','month','arima_rdn','arima_cro']


# choosing time-series features for dimentionality reduction

ts_cols=['lag_rdn_24','lag_rdn_25','lag_rdn_26','lag_rdn_48','lag_rdn_72','lag_rdn_96','lag_rdn_168','lag_cro_72','lag_se_24','lag_se_48','lag_se_72',
 'ma_rdn_24','ma_rdn_48','ma_rdn_72','ma_rdn_96','ma_rdn_168','med_rdn_24','med_rdn_48','med_rdn_72','med_rdn_96','med_rdn_168','sd_rdn_24','sd_rdn_48',
 'sd_rdn_72','sd_rdn_96','sd_rdn_168','ma_cro_24','ma_cro_48','ma_cro_72','ma_cro_96','ma_cro_168','med_cro_24','med_cro_48','med_cro_72','med_cro_96',
 'med_cro_168','sd_cro_24','sd_cro_48','sd_cro_72','sd_cro_96','sd_cro_168']


# scaling data

model1=StandardScaler()
model1.fit(train_df[ts_cols])

z_train=model1.transform(train_df[ts_cols])
x_train=train_df[x_cols]

z_test=model1.transform(test_df[ts_cols])
x_test=test_df[x_cols]

x_train0=train_df[ts_cols+x_cols]
x_test0=test_df[ts_cols+x_cols]



# transforming time-series data as input for recurrent neural networks

y1=train_df['cro']
y2=test_df['cro']


   
# launching hyperparameters tuning

random_steps=15

optimize_steps=5

tune_job=random_tuning_job(z_train, z_test, x_train, x_test, y1, y2, random_steps)


for i in range(optimize_steps):

    good_params=tune_job.sort_values(by='mse_error')[0:3]
    good_params=good_params.reset_index()

    best_params=good_params[0:2]

    tune_job1=evolution_tuning_job(z_train, z_test, x_train, x_test, y1, y2, good_params)

    tune_job=pd.concat([tune_job,tune_job1],axis=0)


good_params=tune_job.sort_values(by='mse_error')[0:10]
print(good_params)

best_params=good_params[0:1]


# transform features

ae_neurons=int(best_params['ae_neurons'])
ae_l1=float(best_params['ae_l1'])
ae_l2=float(best_params['ae_l2'])
ae_drops=float(best_params['ae_drops'])
ae_iters=int(best_params['nn_iter1'])

ae=ae_model(z_train, ae_neurons, ae_l1, ae_l2, ae_drops, ae_iters)

get_layer = K.function([ae.layers[0].input, K.learning_phase()],[ae.layers[0].output])
h1 = np.array(get_layer([z_train])).reshape(z_train.shape[0], ae_neurons)
        
new_train=pd.concat([pd.DataFrame(x_train),pd.DataFrame(h1)],axis=1)
new_train['cro']=train_df['cro']
new_train['rdn']=train_df['rdn']

new_train.to_csv('new_train.csv')


h2 = np.array(get_layer([z_test])).reshape(z_test.shape[0], ae_neurons)
        
new_test=pd.concat([pd.DataFrame(x_test),pd.DataFrame(h2)],axis=1)
new_test['cro']=test_df['cro']
new_test['rdn']=test_df['rdn']

new_test.to_csv('new_test.csv')