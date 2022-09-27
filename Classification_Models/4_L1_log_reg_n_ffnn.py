#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: seema
"""

import os
import pandas as pd  
import numpy as np
import matplotlib.pyplot as plt
# import tensorflow as tf
import copy 
import time
import gc
from functools import reduce

from sklearn.model_selection import train_test_split  
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix  
from sklearn.model_selection import GridSearchCV
import joblib
#
# from skopt import BayesSearchCV
# # parameter ranges are specified by one of below
# from skopt.space import Real, Categorical, Integer
#
# from tensorflow.keras import backend as K
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, InputLayer
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import utils
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import EarlyStopping
# from tensorflow.keras.layers import PReLU
# from tensorflow.keras.layers import LeakyReLU
#
from keras.wrappers.scikit_learn import KerasClassifier
#%% input files
# methylation
omic = "methylation"
path = "results"
# IP data file - should include data and labels in same file
ip_train_data_path = "pancancer_for_classification_train.csv"
ip_test_data_path = "pancancer_for_classification_train.csv"


#%% Change working directory
os.chdir(path)
print(os.getcwd())
del path

#%% IP prediction probabilities files
ffnn_train_path = "ffnn_pred_prob_train_final.csv"
ffnn_test_path = "ffnn_pred_prob_test_final.csv"

rf_train_path = "rf_gini_pred_prob_train_final.csv"
rf_test_path = "rf_gini_pred_prob_test_final.csv"

svm_train_path = "svm_rbf_pred_prob_train_final.csv"
svm_test_path = "svm_rbf_pred_prob_test_final.csv"

#%% OP files
log_reg_bst_fit_model_path = "L1_log_reg.pkl"

ffnn_bst_fit_model_path = "L1_ffnn.pkl"

checkpoint_filepath = "Weights/L1_"+ omic + "_{epoch:02d}_{val_accuracy:.2f}.hdf5"
t = time.time()
export_path_keras = "Weights/L1_"+ omic + "_{}.h5".format(int(t))

file_name_loss_curves = "L1_"+ omic + "_loss.pdf"
file_name_acc_curves = "L1_"+ omic + "_acc.pdf"
loss_acc_obt_file = "L1_"+ omic + "_acc_loss_values_epoch.csv"

#%% Pre-processing train data
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
Y_train = ip_train_data['Clusters']
print("\n Output shape: " +str(Y_train.shape))
del ip_train_data, ip_train_data_path
gc.collect()

#%% Pre-processing test data
# Pre-processing test data
ip_test_data = pd.read_csv(ip_test_data_path, index_col = 0)
ip_test_data.iloc[0:5, 0:5]
ip_test_data.head()
# Reorder so that labels start from zero
ip_test_data['Clusters'] = ip_test_data['Clusters'] - 1

# Display number of samples in each cluster
for i in range(no_of_clusters):
    print(f'Number of samples in cluster {i}: {(ip_test_data[ip_test_data.Clusters == i].shape)} \n')
    
print("\n Cluster samples \n" +str(ip_test_data['Clusters'].value_counts()))

# Separate features and labels
Y_test = ip_test_data['Clusters']
print("\n Output shape: " +str(Y_test.shape))
del ip_test_data, ip_test_data_path
gc.collect()

#%% Read prediction probablitiies
# Read prediction probablitiies
svm_train_prob = pd.read_csv(svm_train_path, index_col = 0)
svm_train_prob.columns = svm_train_prob.columns + "_svm"

svm_test_prob = pd.read_csv(svm_test_path, index_col = 0)
svm_test_prob.columns = svm_test_prob.columns + "_svm"

#==================================================================
rf_train_prob = pd.read_csv(rf_train_path, index_col = 0)
rf_train_prob.columns = rf_train_prob.columns + "_rf"

rf_test_prob = pd.read_csv(rf_test_path, index_col = 0)
rf_test_prob.columns = rf_test_prob.columns + "_rf"

#==================================================================
ffnn_train_prob = pd.read_csv(ffnn_train_path, index_col = 0)
ffnn_train_prob.columns = ffnn_train_prob.columns + "_ffnn"

ffnn_test_prob = pd.read_csv(ffnn_test_path, index_col = 0)
ffnn_test_prob.columns = ffnn_test_prob.columns + "_ffnn"

#==================================================================

#%% Merge prediction probabilities
train_dfs = [svm_train_prob, rf_train_prob, ffnn_train_prob]
test_dfs = [svm_test_prob, rf_test_prob, ffnn_test_prob]

train_probs = reduce(lambda left,right: pd.merge(left, right, left_index = True, right_index = True), train_dfs)

test_probs = reduce(lambda left,right: pd.merge(left, right, left_index = True, right_index = True), test_dfs)

#%% train prob
train_probs = pd.merge(Y_train, train_probs, left_index = True, right_index = True)
train_probs.iloc[0:5, 0:5]
train_probs_x = train_probs.drop(['Clusters'], axis = 1)
train_probs_y = train_probs['Clusters']

test_probs = pd.merge(Y_test, test_probs, left_index = True, right_index = True)
test_probs.iloc[0:5, 0:5]
test_probs_x = test_probs.drop(['Clusters'], axis = 1)
test_probs_y = test_probs['Clusters']

#%% Hyper parameters log reg
# Build logistic regression
Cs = [0.001, 0.01, 0.1, 0, 1, 5, 10, 50, 100]
penalties = ["l2", "none"]
#%% Fit logistic regression
def log_reg_param_selection(X, y, nfolds):
    param_grid = {'C': Cs, 'penalty' : penalties}
    grid_search = GridSearchCV(estimator = LogisticRegression(random_state = 121, max_iter=10000), param_grid = param_grid, cv = nfolds, verbose = 1)
    grid_search.fit(X, y)
    print("================================\n")
    print("logistic regression\n")
    print("Best parameters: " + str(grid_search.best_params_))
    print("================================\n")
    return (grid_search.best_estimator_)

#%% Train Logistic regression
log_reg_bst_fit_model = log_reg_param_selection(train_probs_x, train_probs_y, 5)
joblib.dump(log_reg_bst_fit_model, log_reg_bst_fit_model_path)

#%% Model checkpoint
model_checkpoint_callback = ModelCheckpoint(filepath = checkpoint_filepath, save_weights_only = True, verbose = 1, monitor = 'val_accuracy', mode = 'max', save_best_only = True)
early_stopping_callback = EarlyStopping(monitor = 'val_loss', patience = 10)
# To plot loss on test data
# Create callback object
class TestCallback(Callback):
    def __init__(self, test_data_x, test_data_y):
        self.x = test_data_x
        self.y = test_data_y
        self.losses = []
        self.accuracy = []

    def on_epoch_end(self, epoch, logs={}):
        loss, acc = self.model.evaluate(self.x, self.y, verbose=0)
        self.losses.append(copy.deepcopy(loss))
        self.accuracy.append(copy.deepcopy(acc))
        print('\nTesting loss: {}, acc: {}\n'.format(loss, acc))
        
    def return_results(self):
        return self.losses, self.accuracy
 
#%% FFN train-test split
# Split valid into train and split required for FFNN
X_ffnn_train, X_ffnn_valid, Y_ffnn_train, Y_ffnn_valid = train_test_split(train_probs_x, train_probs_y, test_size = 0.1, random_state = 121, shuffle = True, stratify = train_probs_y)

# Build ffnn
Y_ffnn_train_categ = utils.to_categorical(Y_ffnn_train, num_classes = no_of_clusters)
Y_ffnn_valid_categ = utils.to_categorical(Y_ffnn_valid, num_classes = no_of_clusters)

Y_train_categ = utils.to_categorical(train_probs_y, num_classes = no_of_clusters)
Y_test_categ = utils.to_categorical(test_probs_y, num_classes = no_of_clusters)

callbacks = TestCallback(test_probs_x, Y_test_categ)
#%% FFNN parameters

# parameters
print("Declaring parameters \n")
inputs = X_ffnn_train.shape[1]
hidden1 = 24
outputs = Y_ffnn_train_categ.shape[1] 
learning_rate = 0.001
epochs = 500
batch_size = 64

param_grid = {'learning_rate': [0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001]}
print("Parameters Declared \n")
print(f'Input Nodes: {inputs} , Hidden 1 nodes : {hidden1} , Output nodes: {outputs} \n') 
print(f'Learning rate: {learning_rate} , Epochs : {epochs} , Batch size : {batch_size} \n')
print("Building and compiling model \n")

#%% FFNN model 

def ffnn_vanila(activation = 'relu', learning_rate = 0.001):    
 	# define model
    model = Sequential()
    model.add(InputLayer(input_shape = inputs))
    model.add(Dense(units = hidden1, activation = activation, kernel_initializer = 'he_normal', bias_initializer = 'zeros', name = "hidden_layer_1"))     
    model.add(Dense(units = outputs, activation = 'softmax', kernel_initializer = 'glorot_normal', bias_initializer = 'zeros', name = "output_layer"))    
    #=============================================================== 
    opt_adam = Adam(learning_rate = learning_rate)
    model.compile(optimizer = opt_adam, loss = 'categorical_crossentropy', metrics = ["accuracy"])
    #=============================================================== 
    # Prints a description of the model  
    print(model.summary())
    return model

#%% FFNN CV
model = KerasClassifier(build_fn = ffnn_vanila)
grid = GridSearchCV(estimator = model, param_grid = param_grid, cv = 5)
grid_result = grid.fit(train_probs_x, Y_train_categ)
print("Best fit parameters \n")
print(grid_result.best_params_)
print("\n ====================== \n")
joblib.dump(grid_result.best_params_, ffnn_bst_fit_model_path)
#%% fit model
ffnn_model = ffnn_vanila(learning_rate = grid_result.best_params_['learning_rate'])
ffnn_history = ffnn_model.fit(X_ffnn_train, Y_ffnn_train_categ, batch_size = batch_size, epochs = epochs, shuffle = True, validation_data = (X_ffnn_valid, Y_ffnn_valid_categ), callbacks=[early_stopping_callback, callbacks, model_checkpoint_callback])

#%% FFNNN train
ffnn_model.save(export_path_keras)
# Get test loss at every epoch
test_loss_obt, test_acc_obt = callbacks.return_results()
train_loss_obt = ffnn_history.history['loss']
valid_loss_obt = ffnn_history.history['val_loss']

train_acc_obt = ffnn_history.history['accuracy']
valid_acc_obt = ffnn_history.history['val_accuracy']

loss_acc_obt = pd.DataFrame([train_loss_obt, valid_loss_obt, test_loss_obt, train_acc_obt, valid_acc_obt, test_acc_obt]).transpose()
loss_acc_obt.columns = ["train_loss", "valid_loss", "test_loss", "train_acc", "valid_acc", "test_acc"]
loss_acc_obt.to_csv(loss_acc_obt_file, index = True, header = True)

#%% Figures
print("Plotting figures\n")
# Plot the Loss Curves
plt.figure(figsize=[8,6])
plt.plot(ffnn_history.history['loss'],'r',linewidth=3.0)
plt.plot(ffnn_history.history['val_loss'],'b',linewidth=3.0)
plt.plot(test_loss_obt,'g',linewidth=3.0)
plt.legend(['Training Loss', 'Validation Loss', 'Testing Loss'],fontsize=18)
plt.xlabel('Epochs ',fontsize=16)
plt.ylabel('Loss',fontsize=16)
plt.title('Loss Curves',fontsize=16)
plt.savefig(file_name_loss_curves, bbox_inches='tight')

# Plot the Accuracy Curves
plt.figure(figsize=[8,6])
plt.plot(ffnn_history.history['accuracy'],'r',linewidth=3.0)
plt.plot(ffnn_history.history['val_accuracy'],'b',linewidth=3.0)
plt.plot(test_acc_obt,'g',linewidth=3.0)
plt.legend(['Training Accuracy', 'Validation Accuracy', 'Testing Accuracy'],fontsize=18)
plt.xlabel('Epochs ',fontsize=16)
plt.ylabel('Accuracy',fontsize=16)
plt.title('Accuracy Curves',fontsize=16)
plt.savefig(file_name_acc_curves, bbox_inches='tight')
