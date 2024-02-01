#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 19 16:16:55 2024

@author: timhermans
"""

import tensorflow as tf
import matplotlib.pyplot as plt
import xarray as xr
import numpy as np
from sklearn.metrics import confusion_matrix
from keras.layers import Input
import keras_tuner
import keras
from keras.layers import Input, Concatenate, Conv2D, Flatten, Dense, MaxPooling2D, Dropout, SpatialDropout2D,BatchNormalization
from keras.models import Model
from keras import regularizers
from keras import layers

from surgeNN.models import train_mlr,predict_mlr,build_CNN_model
from surgeNN.io import load_predictand,load_predictors
from surgeNN.utils import plot_loss_evolution, get_train_test_val_idx, normalize_timeseries, rmse 
    
def weighted_mse(y_obs, y_pred):
    return tf.reduce_mean(tf.math.square(y_obs)*(tf.math.square(y_obs - y_pred)), axis=-1)  

tg = 'den_helder-denhdr-nld-rws.csv' #site to predict
predictand = load_predictand('/Users/timhermans/Documents/Github/surgeNN/input/predictands_6hourly',tg) #open predictand csv
predictors = load_predictors('/Users/timhermans/Documents/Github/surgeNN/input/predictors_6hourly',tg) #open predictor xarray dataset
predictors = predictors.sel(time=slice('1980','2016')) #period for which we have hydrodynamic output as well
predictor_timesteps = predictors.time.to_dataframe() #generate helper dictionary for predictor timesteps

# only use predictands at timesteps for which we have predictor values:
predictand = predictand[(predictand['date']>=predictors.time.isel(time=0).values) & (predictand['date']<=predictors.time.isel(time=-1).values)] 

#select predictors at predictand timesteps
predictors = predictors.sel(time=predictand['date'].values)
predictors = (predictors-predictors.mean(dim='time'))/predictors.std(dim='time',ddof=0) #normalize each variable in dataset

predictand['surge'] = normalize_timeseries(predictand['surge']) #normalize predictands
surge_obs = predictand['surge'].values #get values from dictionary

#split into training, validation and testing and get indices
idx_train,idx_test,idx_val = get_train_test_val_idx(predictors.msl.values,surge_obs,[.6,.2,.2],0) #shuffles with random seed 0
#idx_train,idx_test,idx_val = get_train_test_val_idx(predictors.msl.values[2::],surge_obs[2::],[.6,.2,.2],0) #to be able to input lagged data

def build_model(hp):
    n_neurons = hp.Int("n_neurons", min_value=32, max_value=96, step=32)
    dropout_rate = hp.Float("dropout_rate",min_value=0,max_value=.2,step=.1)
    lr = hp.Float("lr", min_value=1e-4, max_value=1e-2, sampling="log")
    n_conv = hp.Int("n_conv", min_value=1, max_value=3, step=1)
    n_dense = hp.Int("n_dense", min_value=1, max_value=2, step=1)
    n_kernels = hp.Int("n_kernels", min_value=32, max_value=96, step=32)
    # call existing model-building code with the hyperparameter values.
    model = build_CNN_model(n_conv,n_dense,n_kernels,n_neurons,predictors,['msl','w','u10','v10'],'test',dropout_rate,lr,weighted_mse,0.01)
       
    return model

build_model(keras_tuner.HyperParameters())

tuner = keras_tuner.RandomSearch(
    hypermodel=build_model,
    objective="val_loss",
    max_trials=20,
    executions_per_trial=1,
    overwrite=True,
    directory="tuning",
    project_name="test_tuning",
)
tuner.search([predictors.msl.values[idx_train],predictors.w.values[idx_train],
                       predictors.u10.values[idx_train],predictors.v10.values[idx_train]], surge_obs[idx_train], epochs=10,
             batch_size=64,
             callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5,
                                     restore_best_weights=True)],
             validation_data=([predictors.msl.values[idx_val],predictors.w.values[idx_val],
                               predictors.u10.values[idx_val],predictors.v10.values[idx_val]], surge_obs[idx_val]))