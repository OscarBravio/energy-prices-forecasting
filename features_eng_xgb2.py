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

import lightgbm as lgb


# starting parameters

label='cro'

tuning=1


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
               n_trees, lr, depth, reg_alpha, reg_lambda):
    
    ae=ae_model(z_train, ae_neurons, ae_l1, ae_l2, ae_drops, nn_iter1)
        
    time.sleep(1)
        
    get_layer = K.function([ae.layers[0].input, K.learning_phase()],[ae.layers[0].output])
    h = np.array(get_layer([z_train])).reshape(z_train.shape[0], ae_neurons)
        
    new_x=pd.concat([pd.DataFrame(x_train),pd.DataFrame(h)],axis=1)
    
    gbt=lgb.LGBMRegressor(n_estimators=n_trees, learning_rate=lr, max_depth=depth, reg_alpha=reg_alpha, reg_lambda=reg_lambda)
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
    nn_iter1=np.random.uniform(800,4000,size=n).astype('int')
    
    n_trees=sample_neurons(n,200)
    depth=sample_neurons(n,15)
    lr=np.random.uniform(size=n)
    alfa1=sample_reg(n)
    alfa2=sample_reg(n)
    

    evaluate=[]

    for i in range(n):

        error=tuning_job(z_train, z_test, x_train, x_test, y_train, y_test,
               ae_neurons[i], ae_l1[i], ae_l2[i], ae_drops[i], nn_iter1[i],
               n_trees[i], lr[i], depth[i], alfa1[i], alfa2[i])

        evaluate.append(error)

        del error
        gc.collect()

    output=pd.DataFrame({'ae_neurons':ae_neurons, 'ae_l1':ae_l1, 'ae_l2':ae_l2, 'ae_drops': ae_drops, 'nn_iter1':nn_iter1, 
                         'trees':n_trees, 'lr':lr, 'depth':depth, 'xg_l1':alfa1, 'xg_l2':alfa2, 'mse_error':evaluate})

    output['method']=np.repeat('random',output.shape[0])

    return(output)


def evolution_tuning_job(z_train, z_test, x_train, x_test, y_train, y_test, good_params):

    ae_neurons=mutate_neurons(good_params['ae_neurons'])
    ae_l1=mutate_reg(good_params['ae_l1'])
    ae_l2=mutate_reg(good_params['ae_l2'])
    ae_drops=mutate_reg(good_params['ae_drops'])
    nn_iter1=mutate_iters(good_params['nn_iter1'])
    
    n_trees=mutate_neurons(good_params['trees'])
    lr=mutate_reg(good_params['lr'])
    depth=good_params['depth']
    alfa1=mutate_reg(good_params['ae_l1'])
    alfa2=mutate_reg(good_params['ae_l2'])

    evaluate=[]

    n=good_params.shape[0]

    for i in range(n):

        error=tuning_job(z_train, z_test, x_train, x_test, y_train, y_test,
               ae_neurons[i], ae_l1[i], ae_l2[i], ae_drops[i], nn_iter1[i],
               n_trees[i], lr[i], depth[i], alfa1[i], alfa2[i])

        evaluate.append(error)

        del error
        gc.collect()


    output=pd.DataFrame({'ae_neurons':ae_neurons, 'ae_l1':ae_l1, 'ae_l2':ae_l2, 'ae_drops': ae_drops, 'nn_iter1':nn_iter1, 
                         'trees':n_trees, 'lr':lr, 'depth':depth, 'xg_l1':alfa1, 'xg_l2':alfa2, 'mse_error':evaluate})

    output['method']=np.repeat('optimize - evolution',output.shape[0])

    return(output)



# sampling neurons 


train_df=pd.read_csv("~/Oskar/energy-prices-forecasting/train_df.csv")
val_test_df=pd.read_csv("~/Oskar/energy-prices-forecasting/test_df.csv")


# choosing fundamental features

x_cols=['hour','demand','supply1','supply2','supply3','supply4','wind','reserve','day','month']


# choosing time-series features for dimentionality reduction

ts_cols=['lag_rdn_24','lag_rdn_25','lag_rdn_26','lag_rdn_48','lag_rdn_72','lag_rdn_96','lag_rdn_168','lag_cro_72','lag_se_24','lag_se_48','lag_se_72',
 'ma_rdn_24','ma_rdn_48','ma_rdn_72','ma_rdn_96','ma_rdn_168','med_rdn_24','med_rdn_48','med_rdn_72','med_rdn_96','med_rdn_168','sd_rdn_24','sd_rdn_48',
 'sd_rdn_72','sd_rdn_96','sd_rdn_168','ma_cro_24','ma_cro_48','ma_cro_72','ma_cro_96','ma_cro_168','med_cro_24','med_cro_48','med_cro_72','med_cro_96',
 'med_cro_168','sd_cro_24','sd_cro_48','sd_cro_72','sd_cro_96','sd_cro_168','arima_rdn','arima_cro']


# feature to predict



if label=='rdn':
    
    test_iter=val_test_df.shape[0]-24
            
    
else:
    
    test_iter=val_test_df.shape[0]-72

val_df=val_test_df[:test_iter]
test_df=val_test_df[test_iter:]

# scaling data

model1=StandardScaler()
model1.fit(train_df[ts_cols])

z_train=model1.transform(train_df[ts_cols])
x_train=train_df[x_cols]

z_val=model1.transform(val_df[ts_cols])
x_val=val_df[x_cols]

z_test=model1.transform(test_df[ts_cols])
x_test=test_df[x_cols]


# transforming time-series data as input for recurrent neural networks

y1=train_df[label]
y2=val_df[label]


   
# launching hyperparameters tuning

if tuning>0:

    random_steps=5
    
    optimize_steps=3
    
    tune_job=random_tuning_job(z_train, z_val, x_train, x_val, y1, y2, random_steps)
    
    
    for i in range(optimize_steps):
    
        good_params=tune_job.sort_values(by='mse_error')[0:3]
        good_params=good_params.reset_index()
    
        best_params=good_params[0:2]
    
        tune_job1=evolution_tuning_job(z_train, z_val, x_train, x_val, y1, y2, good_params)
    
        tune_job=pd.concat([tune_job,tune_job1],axis=0)
    
    
    good_params=tune_job.sort_values(by='mse_error')[0:10]
    print(good_params)
    
    best_params=good_params[0:1]
    
    ae_neurons=int(best_params['ae_neurons'])
    ae_l1=float(best_params['ae_l1'])
    ae_l2=float(best_params['ae_l2'])
    ae_drops=float(best_params['ae_drops'])
    ae_iters=int(best_params['nn_iter1'])


# transform features and make prediction

x_train2=pd.concat([x_train,x_val],axis=0)
z_train2=pd.concat([pd.DataFrame(z_train),pd.DataFrame(z_val)],axis=0)
y_train2=pd.concat([y1,y2],axis=0)



#ae=ae_model(z_train, ae_neurons, ae_l1, ae_l2, ae_drops, ae_iters)
ae=ae_model(z_train2, 27, 0.05, 0, 0.3, 3500)

get_layer = K.function([ae.layers[0].input, K.learning_phase()],[ae.layers[0].output])
h1 = np.array(get_layer([z_train2])).reshape(z_train2.shape[0], 27)

x_train2=x_train2.reset_index()
x_train2=x_train2.drop(['index'],axis=1)
        
new_train=pd.concat([pd.DataFrame(x_train2),pd.DataFrame(h1)],axis=1)

gbt=lgb.LGBMRegressor(n_estimators=80, learning_rate=0.14, max_depth=14, reg_alpha=0.04, reg_lambda=0)
gbt.fit(new_train, y_train2)

col_names=list(new_train.columns)


# predict

h2 = np.array(get_layer([z_test])).reshape(z_test.shape[0], 27)

x_test=x_test.reset_index()
x_test=x_test.drop(['index'],axis=1)

new_test=pd.concat([pd.DataFrame(x_test),pd.DataFrame(h2)],axis=1)

pred=gbt.predict(new_test)

output_name=str(label)+'_prediction_xgb.csv'
pd.Series(pred).to_csv('~/Oskar/energy-prices-forecasting/'+output_name)


# recurrent neual network

h1 = np.array(get_layer([z_train])).reshape(z_train.shape[0], 27)

#x_train=x_train.reset_index()
#x_train=x_train.drop(['index'],axis=1)

new_train=pd.concat([pd.DataFrame(x_train),pd.DataFrame(h1)],axis=1)

h2 = np.array(get_layer([z_val])).reshape(z_val.shape[0], 27)

#x_val=x_val.reset_index()
#x_val=x_val.drop(['index'],axis=1)

new_val=pd.concat([pd.DataFrame(x_val),pd.DataFrame(h2)],axis=1)

skaler=StandardScaler()
skaler.fit(new_train)

new_train2=skaler.transform(new_train)
new_val2=skaler.transform(new_val)

new_train['label']=train_df[label]
new_val['label']=val_df[label]
new_test['label']=test_df[label]


new_train.to_csv('~/Oskar/energy-prices-forecasting/new_train.csv')
new_val.to_csv('~/Oskar/energy-prices-forecasting/new_val.csv')
new_test.to_csv('~/Oskar/energy-prices-forecasting/new_test.csv')


