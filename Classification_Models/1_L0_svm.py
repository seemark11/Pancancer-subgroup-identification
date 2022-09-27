#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: seema
"""
import os
import pandas as pd  
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import copy 
import time
import gc
from functools import reduce

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split  
from sklearn.svm import SVC  
from sklearn.metrics import classification_report, confusion_matrix  
from sklearn.model_selection import GridSearchCV
import joblib


#%%
# Path for op files
path = "results"
# IP data file - should include data and labels in same file
ip_train_data_path = "pancancer_for_classification.csv"

os.chdir(path)

n_jobs = 64
#%%
svm_bst_fit_rbf_model_path = "svm_rbf.pkl"
scaler_Std = StandardScaler()
#%%
# Pre-processing train data
ip_train_data = pd.read_csv(ip_train_data_path, index_col = 0)
no_of_clusters = np.max(ip_train_data['Clusters'])
# Reorder so that labels start from zero
ip_train_data['Clusters'] = ip_train_data['Clusters'] - 1

# Display number of samples in each cluster
for i in range(no_of_clusters):
    print(f'Number of samples in cluster {i}: {(ip_train_data[ip_train_data.Clusters == i].shape)} \n')
    
print("\n Cluster samples \n" + str(ip_train_data['Clusters'].value_counts()))
print("Number of Clusters \n" + str(no_of_clusters))

# Separate features and labels
X_train = ip_train_data.drop('Clusters', axis=1)  
Y_train = ip_train_data['Clusters']

print("\n Input shape: " +str(X_train.shape))
print("\n Output shape: " +str(Y_train.shape))
del ip_train_data, ip_train_data_path
gc.collect()

#%%
Cs = [0.001, 0.01, 0.1, 1, 5, 10, 50, 100]
gammas = [0.001, 0.01, 0.1, 1, 5, 10, 50]
degrees = [1, 2, 3, 4, 5, 6, 7, 8]

#%%
# Sigma for rbf kernel
def svc_rbf_param_selection(X, y, nfolds):
    X = scaler_Std.fit_transform(X)
    param_grid = {'C': Cs, 'gamma' : gammas}
    grid_search = GridSearchCV(estimator = SVC(kernel='rbf', probability = True, random_state = 121), param_grid = param_grid, cv = nfolds, n_jobs = n_jobs,verbose = 1)
    grid_search.fit(X, y)
    print("================================\n")
    print("SVM RBF\n")
    print("Best parameters: " + str(grid_search.best_params_))
    print("================================\n")
    return (grid_search.best_estimator_)

#%%
svm_bst_fit_rbf_model = svc_rbf_param_selection(X_train, Y_train, 5)
joblib.dump(svm_bst_fit_rbf_model, svm_bst_fit_rbf_model_path)
