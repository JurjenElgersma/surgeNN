#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 19 16:16:55 2024

@author: timhermans
"""

import tensorflow as tf
from keras.layers import Input, Concatenate, Conv2D, Flatten, Dense, MaxPooling2D, Dropout, SpatialDropout2D,BatchNormalization
from keras.models import Model
from keras import regularizers
from sklearn.decomposition import PCA
import statsmodels.api as sm
import numpy as np
import xarray as xr
from keras.layers import Input
import keras_tuner
import keras
from keras import regularizers
from keras import layers

# Conv2d
def build_CNN_model(n_conv, n_dense, n_kernels, n_neurons, predictors, predictor_variables, model_name, dropout_rate, lr, loss_function,l2=0.01):
    
    input_shape = (len(predictors.lat_around_tg), len(predictors.lon_around_tg), 1)
    
    inputs = []
    convoluted_vars = []
    for var in predictor_variables: #for each predictor input variable, apply convolution:
        cnn_input = keras.Input(shape=input_shape)
        
        for l in np.arange(n_conv):
            x = layers.Conv2D(n_kernels, kernel_size=(3, 3), padding='same', activation='relu')(cnn_input)        
            x = layers.BatchNormalization()(x)
            x = layers.MaxPooling2D(pool_size=(2, 2), padding="same")(x)
        
        x = SpatialDropout2D(dropout_rate)(x)
        x = layers.Flatten()(x)
        
        convoluted_vars.append(x)
        inputs.append(cnn_input)
        
    concatenated = layers.concatenate(convoluted_vars)

    #dense layers:
    for l in np.arange(n_dense):
        x = layers.Dense(n_neurons,activation='relu',activity_regularizer=regularizers.l2(l2))(concatenated)
        x = layers.Dropout((dropout_rate))(x)
       
    output = layers.Dense(1,activation='linear')(x)
    
    model = keras.Model(inputs=inputs, outputs=output,name=model_name)    
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr),loss=loss_function,metrics=['accuracy']) #compile
    return model

# Multi-linear regression model (Tadesse et al., 2020; Hermans et al., 2023 in review)
def prepare_predictors_for_mlr(predictors,predictor_variables,time_idx):
    #input expects xarray dataset
    
    for var in predictor_variables: #add higher order predictors
        if '_sqd' in var:
            predictors[var] = predictors[var.split('_')[0]]**2
        elif '_cbd' in var:
            predictors[var] = predictors[var.split('_')[0]]**3
        else:
            continue
    
    for k in predictors.keys(): #normalize predictor variables
        predictors[k] = ( predictors[k] - np.mean(predictors[k].values,axis=0) ) / np.std(predictors[k].values,axis=0,ddof=0)
        
    ''' with lagging timesteps (computationally heavy, omitting for now)
    if n_lagged > 0:
        lagged_idx = np.tile(time_idx,(n_lagged+1,1)).transpose() - np.arange(n_lagged+1)
        lagged_idx_da = xr.DataArray(data=lagged_idx,dims=['matched_time','lag'],coords=dict(matched_time=np.arange(len(lagged_idx)),lag=np.arange(n_lagged+1)),) #create fancy indexer
        
        predictors = predictors.isel(time=lagged_idx_da)
    else:
        predictors = predictors.isel(time=time_idx)
        
    if 'lag' in predictors.coords:
        predictors['stacked'] = predictors[list(predictors.keys())].to_array(dim="var") #put predictor variables into one array
        predictors['stacked'] = predictors['stacked'].transpose("matched_time","lag","var","lon_around_tg",...).stack(f=['lag','var','lon_around_tg','lat_around_tg'],create_index=False)
    else:
    '''
    predictors['stacked'] = predictors[predictor_variables].to_array(dim="var") #put predictor variables into one array
    predictors['stacked'] = predictors['stacked'].transpose("time","var","lon_around_tg",...).stack(f=['var','lon_around_tg','lat_around_tg'],create_index=False)
    
    return predictors.isel(time=time_idx)

def train_mlr(predictors,predictand,idx_train,predictor_variables):
    
    X = prepare_predictors_for_mlr(predictors,predictor_variables,idx_train).stacked.values
    
    pca = PCA(.95) #first EOFs explaining at least 95% of variance
    
    pca.fit(X) #principal component analysis
    X_pca = pca.transform(X) #transform predictors
    X_pca = sm.add_constant(X_pca) #add intercept for the regression
 
    est = sm.OLS(predictand[idx_train], X_pca).fit() #estimate MLR parameters
     
    return est.params,pca.components_


def predict_mlr(predictors,mlr_coefs,training_components,idx_test,predictor_variables):
    '''prediction step, multiply estimated coefficients with predictors'''
    
    X = prepare_predictors_for_mlr(predictors,predictor_variables,idx_test).stacked.values
    
    pca_prediction = PCA(len(mlr_coefs[np.isfinite(mlr_coefs)])-1) #get same number of pcs as used for regression coefficient estimation
    pca_prediction.fit(X)
    pcs_prediction = pca_prediction.transform(X)
    
    #sign check of the components
    prediction_components = pca_prediction.components_
    p_idx = int(len(predictors.f)/4)
    
    rmses = np.sqrt(np.mean((prediction_components[:,0:p_idx]-training_components[:,0:p_idx])**2,axis=-1))
    rmses_flipped = np.sqrt(np.mean((prediction_components[:,0:p_idx]--training_components[:,0:p_idx])**2,axis=-1))
    
    s = (rmses<rmses_flipped).astype('int') #flip pcs if rmse of flipped pc is lower
    s[s==0]=-1
    pcs_prediction = pcs_prediction * s
    
    prediction = np.sum(mlr_coefs * np.column_stack((np.ones(pcs_prediction.shape[0]),pcs_prediction)),axis=1) 
  
    return prediction