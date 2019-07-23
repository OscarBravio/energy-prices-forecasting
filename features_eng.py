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


def rnn_model(model, x, y, rnn_neurons1, rnn_neurons2, activs, n_iter):

    if model=='forward_rucurrent':

        rnn=Sequential()
        rnn.add(SimpleRNN(rnn_neurons1, activation=activs, input_shape=(24, x.shape[2])))
        rnn.add(Dense(rnn_neurons2, activation=activs, input_dim=rnn_neurons1))
        rnn.add(Dense(24, activation='linear', input_dim=rnn_neurons2))

    elif model=='recurrent_forward':

        rnn=Sequential()
        rnn.add(Dense(rnn_neurons1, activation=activs, input_shape=(24, x.shape[2])))
        rnn.add(SimpleRNN(rnn_neurons2, activation=activs, input_dim=rnn_neurons1))
        rnn.add(Dense(24, activation='linear', input_dim=rnn_neurons2))

    else:

        print('WRONG MODEL!!!')

    rnn.compile(loss='mean_squared_error', optimizer='adam')
    rnn.fit(x, y, epochs=n_iter, batch_size=32)

    return(rnn)


def tuning_job(model, z_train, z_test, x_train, x_test, y_train, y_test,
               ae_neurons, ae_l1, ae_l2, ae_drops, nn_iter1,
               rnn_neurons1, rnn_neurons2, rnn_activs, nn_iter2):

    ae=ae_model(z_train, ae_neurons, ae_l1, ae_l2, ae_drops, nn_iter1)

    time.sleep(1)

    get_layer = K.function([ae.layers[0].input, K.learning_phase()],[ae.layers[0].output])
    h = np.array(get_layer([z_train])).reshape(z_train.shape[0], ae_neurons)

    new_x=pd.concat([pd.DataFrame(x_train),pd.DataFrame(h)],axis=1)
    x_ts=np.array(new_x).reshape(new_x.shape[0]/24,24,new_x.shape[1])


    rnn=rnn_model(model, x_ts, y_train, rnn_neurons1, rnn_neurons2, rnn_activs, nn_iter2)

    time.sleep(1)

    h_test=np.array(get_layer([z_test])).reshape(z_test.shape[0], ae_neurons)

    new_x=pd.concat([pd.DataFrame(x_test),pd.DataFrame(h_test)],axis=1)
    x_ts=np.array(new_x).reshape(new_x.shape[0]/24,24,new_x.shape[1])

    pred=rnn.predict(x_ts)
    error=mse_error(pred,y_test).sum()

    del ae, rnn
    gc.collect()

    return(error)


def random_tuning_job(model, z_train, z_test, x_train, x_test, y_train, y_test, n):

    ae_neurons=sample_neurons(n,50)
    ae_l1=sample_reg(n)
    ae_l2=sample_reg(n)
    ae_drops=sample_drop(n)

    rnn_neurons1=sample_neurons(n,50)
    rnn_neurons2=sample_neurons(n,50)
    rnn_activs=sample_activs(n)

    nn_iter1=np.random.uniform(500,3000,size=n).astype('int')
    nn_iter2=np.random.uniform(100,300,size=n).astype('int')

    evaluate=[]

    for i in range(n):

        error=tuning_job(model, z_train, z_test, x_train, x_test, y_train, y_test,
               ae_neurons[i], ae_l1[i], ae_l2[i], ae_drops[i], nn_iter1[i],
               rnn_neurons1[i], rnn_neurons2[i], rnn_activs[i], nn_iter2[i])

        evaluate.append(error)

        del error
        gc.collect()

    output=pd.DataFrame({'ae_neurons':ae_neurons, 'ae_l1':ae_l1, 'ae_l2':ae_l2, 'ae_drops': ae_drops, 'nn_iter1':nn_iter1,
                     'rnn_neurons1':rnn_neurons1, 'rnn_neurons2':rnn_neurons2, 'rnn_activs':rnn_activs, 'nn_iter2':nn_iter2,
                     'mse_error':evaluate})

    output['method']=np.repeat('random',output.shape[0])

    return(output)



def evolution_tuning_job(model, z_train, z_test, x_train, x_test, y_train, y_test, good_params):

    ae_neurons=mutate_neurons(good_params['ae_neurons'])
    ae_l1=mutate_reg(good_params['ae_l1'])
    ae_l2=mutate_reg(good_params['ae_l2'])
    ae_drops=mutate_reg(good_params['ae_drops'])

    rnn_neurons1=mutate_neurons(good_params['rnn_neurons1'])
    rnn_neurons2=mutate_neurons(good_params['rnn_neurons2'])

    nn_iter1=mutate_iters(good_params['nn_iter1'])
    nn_iter2=mutate_iters(good_params['nn_iter2'])

    rnn_activs=good_params['rnn_activs']

    evaluate=[]

    n=good_params.shape[0]

    for i in range(n):

        error=tuning_job(model, z_train, z_test, x_train, x_test, y_train, y_test,
               ae_neurons[i], ae_l1[i], ae_l2[i], ae_drops[i], nn_iter1[i],
               rnn_neurons1[i], rnn_neurons2[i], rnn_activs[i], nn_iter2[i])

        evaluate.append(error)

        del error
        gc.collect()


    output=pd.DataFrame({'ae_neurons':ae_neurons, 'ae_l1':ae_l1, 'ae_l2':ae_l2, 'ae_drops': ae_drops, 'nn_iter1':nn_iter1,
                     'rnn_neurons1':rnn_neurons1, 'rnn_neurons2':rnn_neurons2, 'rnn_activs':rnn_activs, 'nn_iter2':nn_iter2,
                     'mse_error':evaluate})

    output['method']=np.repeat('optimize - evolution',output.shape[0])

    return(output)


def genetic_tuning_job(model, z_train, z_test, x_train, x_test, y_train, y_test, params):

    ae_neurons=int(params['ae_neurons'].mean())
    ae_l1=float(params['ae_l1'].mean())
    ae_l2=float(params['ae_l2'].mean())
    ae_drops=float(good_params['ae_drops'].mean())

    rnn_neurons1=int(params['rnn_neurons1'].mean())
    rnn_neurons2=int(params['rnn_neurons2'].mean())

    nn_iter1=int(params['nn_iter1'].mean())
    nn_iter2=int(params['nn_iter2'].mean())

    rnn_activs=list(params['rnn_activs'])[0]


    evaluate=tuning_job(model, z_train, z_test, x_train, x_test, y_train, y_test,
             ae_neurons, ae_l1, ae_l2, ae_drops, nn_iter1,
             rnn_neurons1, rnn_neurons2, rnn_activs, nn_iter2)


    output=pd.DataFrame({'ae_neurons':ae_neurons, 'ae_l1':ae_l1, 'ae_l2':ae_l2, 'ae_drops': ae_drops, 'nn_iter1':nn_iter1,
                     'rnn_neurons1':rnn_neurons1, 'rnn_neurons2':rnn_neurons2, 'rnn_activs':rnn_activs, 'nn_iter2':nn_iter2,
                     'mse_error':evaluate},index=[0])

    output['method']=np.repeat('optimize - crossover',output.shape[0])

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

model2=StandardScaler()
model2.fit(train_df[x_cols])

z_train=model1.transform(train_df[ts_cols])
x_train=model2.transform(train_df[x_cols])

z_test=model1.transform(test_df[ts_cols])
x_test=model2.transform(test_df[x_cols])



# transforming time-series data as input for recurrent neural networks


y=train_df['cro']

y_mean=float(y.mean())
y_sd=float(y.std())

y1=(y-y_mean)/y_sd

y2=(test_df['cro']-y_mean)/y_sd

y_train=y1.values.reshape(y1.shape[0]/24,24)

y_test=y2.values.reshape(y2.shape[0]/24,24)



# launching hyperparameters tuning


random_steps=5

optimize_steps=3

model='recurrent_forward'


tune_job=random_tuning_job(model,z_train, z_test, x_train, x_test, y_train, y_test, random_steps)


for i in range(optimize_steps):

    good_params=tune_job.sort_values(by='mse_error')[0:3]
    good_params=good_params.reset_index()

    best_params=good_params[0:2]

    tune_job1=evolution_tuning_job(model, z_train, z_test, x_train, x_test, y_train, y_test, good_params)

    tune_job2=genetic_tuning_job(model, z_train, z_test, x_train, x_test, y_train, y_test, best_params)

    tune_job=pd.concat([tune_job,tune_job1,tune_job2],axis=0)


good_params=tune_job.sort_values(by='mse_error')[0:10]
print(good_params)
