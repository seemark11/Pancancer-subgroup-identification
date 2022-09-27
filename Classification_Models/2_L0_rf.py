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
from sklearn.ensemble import RandomForestClassifier
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
rf_bst_fit_gini_model_path = "rf_gini.pkl"

scaler_Std = StandardScaler()

#%%
# Pre-processing train data
ip_train_data = pd.read_csv(ip_train_data_path, index_col = 0)
ip_train_data.iloc[0:5, 0:5]
ip_train_data.head()

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
# Number of features to consider at every split
max_features = ['auto', 'sqrt', 0.2, 0.3]

# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 10, stop = 100, num = 10)]

# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 100, num = 10)]
max_depth.append(None)

# Minimum number of samples required to split a node
min_samples_split = [5, 10, 15]

# Minimum number of samples required at each leaf node
min_samples_leaf = [4, 10, 20]


#%%
def rf_gini_param_selection(X, y, nfolds):
    X = scaler_Std.fit_transform(X)
    
    param_grid = {'n_estimators': n_estimators, 
                  'max_features': max_features, 
                   'max_depth': max_depth, 
                   'min_samples_split': min_samples_split,
                   'min_samples_leaf': min_samples_leaf}
    
    grid_search = GridSearchCV(estimator = RandomForestClassifier(criterion = "gini", oob_score = True, random_state = 121), param_grid = param_grid, n_jobs = n_jobs,cv = nfolds)
    grid_search.fit(X, y)
    
    print("================================\n")
    print("RF Gini\n")
    print("Best parameters: " + str(grid_search.best_params_))
    print("================================\n")
    return (grid_search.best_estimator_)
#%%
rf_bst_fit_gini_model = rf_gini_param_selection(X_train, Y_train, 5)
joblib.dump(rf_bst_fit_gini_model, rf_bst_fit_gini_model_path)

