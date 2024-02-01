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

# Conv2d
def build_conv_layers(x):
    
    x = Conv2D(32, (3,3), padding='same',activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(2)(x)
    x = SpatialDropout2D(.1)(x)
    x = Conv2D(64, (3,3), padding='same',activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(2)(x)
    x = SpatialDropout2D(.1)(x)
    x = Flatten()(x)
    x = BatchNormalization()(x)
    
    return x

def build_cnn(inputs):
    
    layers = [build_conv_layers(i) for i in inputs]
    concat_layers = Concatenate()(layers)
    
    #densely connected layers:
    dense = Dense(64,activation='relu',activity_regularizer=regularizers.l2(0.01))(concat_layers)
    dropped = Dropout((0.1))(dense)
    dense = Dense(64,activation='relu',activity_regularizer=regularizers.l2(0.01))(concat_layers)
    dropped = Dropout((0.1))(dense)
    output = Dense(1,activation='linear')(dropped)
    
    model = Model(inputs=inputs, outputs=output)

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