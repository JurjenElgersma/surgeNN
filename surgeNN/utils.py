#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 31 12:00:43 2024

@author: timhermans
"""
import xarray as xr
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

def normalize_timeseries(timeseries):
    return ( timeseries - np.nanmean(timeseries) ) / np.std( timeseries, ddof=0)

def get_train_test_val_idx(x,y,fractions,random_state):
    '''fractions in order: [train,test,val]'''
    x_train, x_test, y_train, y_test, idx_train, idx_test = train_test_split(x, y,np.arange(len(x)), test_size=1 - fractions[0],shuffle=True,random_state=random_state)
    x_val, x_test, y_val, y_test, idx_val, idx_test = train_test_split(x_test, y_test, idx_test, test_size=fractions[1]/(fractions[1] + fractions[2]),shuffle=False) 

    return idx_train,idx_test,idx_val

def plot_loss_evolution(history):
    f = plt.figure()
    
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label = 'val_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    #plt.ylim([0.5, 1])
    plt.legend(loc='lower right')
    
    return f

def rmse(y_obs,y_pred):
    return np.sqrt( np.mean( (y_obs-y_pred)**2 ) )
