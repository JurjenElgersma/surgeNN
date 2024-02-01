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

from surgeNN.models import train_mlr,predict_mlr,build_CNN_model
from surgeNN.io import load_predictand,load_predictors
from surgeNN.utils import plot_loss_evolution, get_train_test_val_idx, normalize_timeseries, rmse 
    
def weighted_mse(y_obs, y_pred):
    return tf.reduce_mean(tf.math.square(y_obs)*(tf.math.square(y_obs - y_pred)), axis=-1)  
    #return tf.reduce_mean(tf.math.square(tf.math.square(y_obs))*(tf.math.square(y_obs - y_pred)), axis=-1)  
    #return tf.reduce_mean(tf.math.square(tf.math.maximum(0.0,y_obs))*(tf.math.square(y_obs - y_pred)), axis=-1)  

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

model = build_CNN_model(2,1,92,32,predictors,['msl','w','u10','v10'],'test',0,0.0001,weighted_mse,0.01)

'''
#build the convolutional neural network 
inputs =  [  Input((20,20,1)),  Input((20,20,1)),  Input((20,20,1)),  Input((20,20,1))]
model = build_cnn(inputs)

model.compile(optimizer='adam',loss=weighted_mse,metrics=["accuracy"]) #compile
'''
#train the model (hyperparameters are not tuned yet)
history = model.fit(x=[predictors.msl.values[idx_train],predictors.w.values[idx_train],
                       predictors.u10.values[idx_train],predictors.v10.values[idx_train]],
                    y=surge_obs[idx_train], 
                    epochs=100, 
                    validation_data=([predictors.msl.values[idx_val],predictors.w.values[idx_val],
                                      predictors.u10.values[idx_val],predictors.v10.values[idx_val]], surge_obs[idx_val]),
                    batch_size=64,
                    callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10,
                                            restore_best_weights=True)])

#generate predictions for test set
surge_cnn_test = model.predict([predictors.msl.values[idx_test],predictors.w.values[idx_test],
                            predictors.u10.values[idx_test],predictors.v10.values[idx_test]],verbose=0).flatten()

f = plot_loss_evolution(history) #plot training loss

##### do some evaluation on the test set:
threshold_pct = 95 #which extremes to look at
threshold_value = np.percentile(surge_obs[idx_test],threshold_pct) #threshold value

surge_obs_test_exceedances = (surge_obs[idx_test]>=threshold_value).flatten() #find where exceeding threshold
surge_cnn_test_exceedances = (surge_cnn_test>=threshold_value).flatten() #find where exceeding threshold

print('---CNN---')
print('bulk correlation r='+str(np.corrcoef(surge_cnn_test,surge_obs[idx_test])[0][1]))
print('bulk RMSE='+str(rmse(surge_cnn_test,surge_obs[idx_test])))
print('Confusion matrix exceedances above {0}th percentile:'.format(threshold_pct))
print(confusion_matrix(surge_obs_test_exceedances,surge_cnn_test_exceedances))

print('Correlation at timesteps where observations above {0}th percentile:'.format(threshold_pct))
print('r=' + str(np.corrcoef(surge_cnn_test[surge_obs_test_exceedances],surge_obs[idx_test][surge_obs_test_exceedances])[0][1]))
print('RMSE at timesteps where observations above {0}th percentile:'.format(threshold_pct))
print('RMSE=' + str(rmse(surge_cnn_test[surge_obs_test_exceedances],surge_obs[idx_test][surge_obs_test_exceedances])))


#use same predictors to train the MLR model of Tadesse et al. (2020)
coefs,pcas = train_mlr(predictors,surge_obs,idx_train,['msl','w','u10','u10_sqd','u10_cbd','v10','v10_sqd','v10_cbd'])
surge_mlr_test = predict_mlr(predictors,coefs,pcas,idx_test,['msl','w','u10','u10_sqd','u10_cbd','v10','v10_sqd','v10_cbd'])   
surge_mlr_test_exceedances = (surge_mlr_test>=threshold_value).flatten() #find where exceeding threshold


print('---MLR---')
print('bulk correlation r='+str(np.corrcoef(surge_mlr_test,surge_obs[idx_test])[0][1]))
print('bulk RMSE='+str(rmse(surge_mlr_test,surge_obs[idx_test])))
print('Confusion matrix exceedances above {0}th percentile:'.format(threshold_pct))
print(confusion_matrix(surge_obs_test_exceedances,surge_mlr_test_exceedances))

print('Correlation at timesteps where observations above {0}th percentile:'.format(threshold_pct))
print('r=' + str(np.corrcoef(surge_mlr_test[surge_obs_test_exceedances],surge_obs[idx_test][surge_obs_test_exceedances])[0][1]))
print('RMSE at timesteps where observations above {0}th percentile:'.format(threshold_pct))
print('RMSE=' + str(rmse(surge_mlr_test[surge_obs_test_exceedances],surge_obs[idx_test][surge_obs_test_exceedances])))


#also look at outputs of a hydrodynamic model
surge_codec = xr.open_dataset('/Users/timhermans/Documents/PostDoc/Phase3_statistical_surges/CoDEC_ERA5_at_gesla3_tgs_eu_6hourly_anoms.nc')
surge_codec = surge_codec.sel(time=predictand['date'].values)
surge_codec['surge'] = (surge_codec['surge'] - surge_codec['surge'].mean(dim='time'))/surge_codec['surge'].std(dim='time',ddof=0) #normalize
surge_codec_test = surge_codec.sel(tg=tg).sel(time=predictand['date'].values[idx_test]).surge.values #select test timesteps
surge_codec_test_exceedances = (surge_codec_test>=threshold_value).flatten() #find where exceeding threshold
 
print('---CoDEC---')
print('bulk correlation r='+str(np.corrcoef(surge_codec_test,surge_obs[idx_test])[0][1]))
print('bulk RMSE='+str(rmse(surge_codec_test,surge_obs[idx_test])))
print('Confusion matrix exceedances above {0}th percentile:'.format(threshold_pct))
print(confusion_matrix(surge_obs_test_exceedances,surge_codec_test_exceedances))

print('Correlation at timesteps where observations above {0}th percentile:'.format(threshold_pct))
print('r=' + str(np.corrcoef(surge_codec_test[surge_obs_test_exceedances],surge_obs[idx_test][surge_obs_test_exceedances])[0][1]))
print('RMSE at timesteps where observations above {0}th percentile:'.format(threshold_pct))
print('RMSE=' + str(rmse(surge_codec_test[surge_obs_test_exceedances],surge_obs[idx_test][surge_obs_test_exceedances])))


#some plots
'''
plt.figure()
plt.plot(surge_obs[idx_test][surge_obs_test_exceedances],label='obs')
plt.plot(surge_codec_test[surge_obs_test_exceedances],label='codec')
plt.legend()

plt.figure()
plt.plot(surge_obs[idx_test][surge_obs_test_exceedances],label='obs')
plt.plot(surge_cnn_test[surge_obs_test_exceedances],label='nn')
plt.legend()

plt.figure()
plt.plot(surge_obs[idx_test][surge_obs_test_exceedances],label='obs')
plt.plot(surge_mlr_test[surge_obs_test_exceedances],label='mlr')
plt.legend()

plt.figure()
plt.plot(surge_obs[idx_test],label='obs')
plt.plot(surge_codec_test,label='codec')

plt.legend()

plt.figure()
plt.plot(surge_obs[idx_test],label='obs')
plt.plot(surge_mlr_test,label='mlr')
plt.legend()

plt.figure()
plt.plot(surge_obs[idx_test],label='obs')
plt.plot(surge_cnn_test,label='nn')
plt.legend()
'''