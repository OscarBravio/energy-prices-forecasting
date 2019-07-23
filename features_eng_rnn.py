#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 31 16:35:49 2018

@author: bukowskio
"""

import pandas as pd
import numpy as np
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


def rnn_model(model, x, y, rnn_neurons1, rnn_neurons2, activs, n_iter, l1_reg, l2_reg, drop_out1, drop_out2):

    if model=='forward_rucurrent':

        rnn=Sequential()
        rnn.add(SimpleRNN(rnn_neurons1, activation=activs, activity_regularizer=l1_l2(l1=l1_reg, l2=l2_reg), input_shape=(24, x.shape[2])))
        rnn.add(Dropout(drop_out1))
        rnn.add(Dense(rnn_neurons2, activation=activs, input_dim=rnn_neurons1))
        rnn.add(Dropout(drop_out2))
        rnn.add(Dense(24, activation='linear', input_dim=rnn_neurons2))

    elif model=='recurrent_forward':

        rnn=Sequential()
        rnn.add(Dense(rnn_neurons1, activation=activs, activity_regularizer=l1_l2(l1=l1_reg, l2=l2_reg), input_shape=(24, x.shape[2])))
        rnn.add(Dropout(drop_out1))
        rnn.add(SimpleRNN(rnn_neurons2, activation=activs, input_dim=rnn_neurons1))
        rnn.add(Dropout(drop_out2))
        rnn.add(Dense(24, activation='linear', input_dim=rnn_neurons2))

    else:

        print('WRONG MODEL!!!')

    rnn.compile(loss='mean_squared_error', optimizer='adam')
    rnn.fit(x, y, epochs=n_iter, batch_size=32)

    return(rnn)


def tuning_job(model, x_train, x_test, y_train, y_test, l1_reg, l2_reg, drop_out1, drop_out2,
               rnn_neurons1, rnn_neurons2, rnn_activs, nn_iter2):

    x_ts=np.array(x_train).reshape(x_train.shape[0]/24,24,x_train.shape[1])

    rnn=rnn_model(model, x_ts, y_train, rnn_neurons1, rnn_neurons2, rnn_activs, nn_iter2, l1_reg, l2_reg, drop_out1, drop_out2)

    x_ts=np.array(x_test).reshape(x_test.shape[0]/24,24,x_test.shape[1])

    pred=rnn.predict(x_ts)
    error=mse_error(pred,y_test).sum()

    del rnn
    gc.collect()

    return(error)


def random_tuning_job(model, x_train, x_test, y_train, y_test, n):

    rnn_neurons1=sample_neurons(n,50)
    rnn_neurons2=sample_neurons(n,50)
    rnn_activs=sample_activs(n)
    
    l1_reg=sample_reg(n) 
    l2_reg=sample_reg(n) 
    drop_out1=sample_reg(n) 
    drop_out2=sample_reg(n)

    nn_iter2=np.random.uniform(100,2000,size=n).astype('int')

    evaluate=[]

    for i in range(n):

        error=tuning_job(model, x_train, x_test, y_train, y_test, l1_reg[i], l2_reg[i], drop_out1[i], drop_out2[i],
               rnn_neurons1[i], rnn_neurons2[i], rnn_activs[i], nn_iter2[i])

        evaluate.append(error)

        del error
        gc.collect()

    output=pd.DataFrame({
                     'rnn_neurons1':rnn_neurons1, 'rnn_neurons2':rnn_neurons2, 'rnn_activs':rnn_activs, 'nn_iter2':nn_iter2,
                     'l1_reg':l1_reg, 'l2_reg':l2_reg, 'drop1':drop_out1, 'drop2':drop_out2,
                     'mse_error':evaluate})

    output['method']=np.repeat('random',output.shape[0])

    return(output)



def evolution_tuning_job(model, x_train, x_test, y_train, y_test, good_params):


    rnn_neurons1=mutate_neurons(good_params['rnn_neurons1'])
    rnn_neurons2=mutate_neurons(good_params['rnn_neurons2'])
    
    l1_reg=mutate_reg(good_params['l1_reg'])
    l2_reg=mutate_reg(good_params['l2_reg'])
    drop_out1=mutate_reg(good_params['drop1'])
    drop_out2=mutate_reg(good_params['drop2'])

    nn_iter2=mutate_iters(good_params['nn_iter2'])

    rnn_activs=good_params['rnn_activs']

    evaluate=[]

    n=good_params.shape[0]

    for i in range(n):

        error=tuning_job(model, x_train, x_test, y_train, y_test, l1_reg[i], l2_reg[i], drop_out1[i], drop_out2[i],
               rnn_neurons1[i], rnn_neurons2[i], rnn_activs[i], nn_iter2[i])

        evaluate.append(error)

        del error
        gc.collect()


    output=pd.DataFrame({
                     'rnn_neurons1':rnn_neurons1, 'rnn_neurons2':rnn_neurons2, 'rnn_activs':rnn_activs, 'nn_iter2':nn_iter2,
                     'l1_reg':l1_reg, 'l2_reg':l2_reg, 'drop1':drop_out1, 'drop2':drop_out2,
                     'mse_error':evaluate})

    output['method']=np.repeat('optimize - evolution',output.shape[0])

    return(output)



# sampling neurons


train_df=pd.read_csv("~/Oskar/energy-prices-forecasting/new_train.csv")
test_df=pd.read_csv("~/Oskar/energy-prices-forecasting/new_test.csv")


# choosing fundamental features

y_train=train_df['cro']
x_train=train_df.drop(['Unnamed: 0','cro','rdn'],axis=1)

y_test=test_df['cro']
x_test=test_df.drop(['Unnamed: 0','cro','rdn'],axis=1)


# scaling data

model1=StandardScaler()
model1.fit(x_train)

#model2=StandardScaler()
#model2.fit(y_train)

y_mean=np.mean(y_train)
y_sd=np.std(y_train)

x_train2=model1.transform(x_train)
x_test2=model1.transform(x_test)

#y_train2=model2.transform(y_train)
#y_test2=model2.transform(y_test)

y_train2=(y_train-y_mean)/y_sd
y_test2=(y_test-y_mean)/y_sd


# transforming time-series data as input for recurrent neural networks


y_train3=y_train2.reshape(y_train2.shape[0]/24,24)

y_test3=y_test2.reshape(y_test2.shape[0]/24,24)



#model='forward_rucurrent'


# launching hyperparameters tuning

eee1=random_tuning_job('forward_rucurrent', x_train, x_test, y_train3, y_test3, 15).sort_values(by='mse_error')

eee2=random_tuning_job('recurrent_forward', x_train, x_test, y_train3, y_test3, 15).sort_values(by='mse_error')

x_ts=np.array(x_train).reshape(x_train.shape[0]/24,24,x_train.shape[1])

rnn=rnn_model('forward_rucurrent', x_ts, y_train3, 33, 39, 'relu', 200, 0.2, 0.2, 0.2, 0.05)

x_ts=np.array(x_test).reshape(x_test.shape[0]/24,24,x_test.shape[1])

pred=rnn.predict(x_ts)*y_sd+y_mean
                
pred=pred.reshape(6*24,)

plt.plot(pred[0:48],color='green')
plt.plot(y_test[0:48],color='blue')
plt.show()