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

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split  
from sklearn.metrics import classification_report, confusion_matrix  
from sklearn.model_selection import GridSearchCV
import joblib

from tensorflow.keras import backend as K
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import utils
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import EarlyStopping
# from tensorflow.keras.layers import PReLU
# from tensorflow.keras.layers import LeakyReLU
#
from keras.wrappers.scikit_learn import KerasClassifier

#%%
# Path for op files
omic = "methylation"
path = "results"
# IP data file - should include data and labels in same file
ip_train_data_path = "pancancer_for_classification_train.csv"
ip_test_data_path = "pancancer_for_classification_train.csv"

os.chdir(path)

os.mkdir("Weights")

n_jobs = 1
#%%
ffnn_bst_fit_model_path = "ffnn_learning_rate.pkl"

checkpoint_filepath = "Weights/"+ omic + "_{epoch:02d}_{val_accuracy:.2f}.hdf5"
t = time.time()
export_path_keras = "Weights/" + omic + "_{}.h5".format(int(t))

file_name_loss_curves = omic + "_loss.pdf"
file_name_acc_curves = omic + "_acc.pdf"
loss_acc_obt_file = omic + "_acc_loss_values_epoch.csv"

scaler_MinMax = MinMaxScaler()

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
X_test = ip_test_data.drop('Clusters', axis=1)  
Y_test = ip_test_data['Clusters']

print("\n Input shape: " +str(X_test.shape))
print("\n Output shape: " +str(Y_test.shape))
del ip_test_data, ip_test_data_path
gc.collect()

#%%
model_checkpoint_callback = ModelCheckpoint(filepath=checkpoint_filepath, save_weights_only = True, verbose = 1, monitor = 'val_accuracy', mode = 'max', save_best_only = True)
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
 
#%%
X_train_ffnn, X_valid_ffnn, Y_train_ffnn, Y_valid_ffnn = train_test_split(X_train, Y_train, test_size = 0.1, random_state = 121, shuffle = True, stratify = Y_train)

Y_train_ffnn_categ = utils.to_categorical(Y_train_ffnn, num_classes = no_of_clusters)
Y_valid_ffnn_categ = utils.to_categorical(Y_valid_ffnn, num_classes = no_of_clusters)
print("FFNN train")
print("\n Input shape: " +str(X_train_ffnn.shape))
print("\n Output shape: " +str(Y_train_ffnn.shape))

print("FFNN validation")
print("\n Input shape: " +str(X_valid_ffnn.shape))
print("\n Output shape: " +str(Y_valid_ffnn.shape))

Y_train_categ = utils.to_categorical(Y_train, num_classes = no_of_clusters)
Y_test_categ = utils.to_categorical(Y_test, num_classes = no_of_clusters)
print("Train")
print("\n Input shape: " +str(X_train.shape))
print("\n Output shape: " +str(Y_train.shape))

print("Test")
print("\n Input shape: " +str(X_test.shape))
print("\n Output shape: " +str(Y_test.shape))

callbacks = TestCallback(X_test, Y_test_categ)
#%%

# parameters
# 1024, 512, 256, 128, 64
print("Declaring parameters \n")
inputs = X_train.shape[1]
hidden1 = 512
hidden2 = 256
hidden3 = 128
hidden4 = 64
outputs = Y_train_categ.shape[1] 
learning_rate = 0.01
epochs = 500
batch_size = 32

param_grid = {'learning_rate': [0.05, 0.01, 0.005, 0.001]}

print("Parameters Declared \n")
print(f'Input Nodes: {inputs} , Hidden 1 nodes : {hidden1} , Hidden 2 nodes: {hidden2} \n') 
print(f'Hidden 3 nodes : {hidden3}, Hidden 4 nodes : {hidden4}, Output nodes : {outputs} \n')
print(f'Learning rate: {learning_rate} , Epochs : {epochs} , Batch size : {batch_size} \n')
print("Building and compiling model \n")

#%%

def ffnn_vanila(activation = 'relu', learning_rate = 0.01):    
 	# define model
    model = Sequential()
    model.add(Input(shape = inputs))
    model.add(Dense(units = hidden1, activation = activation, kernel_initializer = 'he_normal', bias_initializer = 'zeros', name = "hidden_layer_1"))    
    model.add(Dense(units = hidden2, activation = activation, kernel_initializer = 'he_normal', bias_initializer = 'zeros', name = "hidden_layer_2"))
    model.add(Dense(units = hidden3, activation = activation, kernel_initializer = 'he_normal', bias_initializer = 'zeros', name = "hidden_layer_3"))
    model.add(Dense(units = hidden4, activation = activation, kernel_initializer = 'he_normal', bias_initializer = 'zeros', name = "hidden_layer_4"))          
    model.add(Dense(units = outputs, activation = 'softmax', kernel_initializer = 'glorot_normal', bias_initializer = 'zeros', name = "output_layer"))    
    #=============================================================== 
    opt_adam = Adam(learning_rate = learning_rate)
    model.compile(optimizer = opt_adam, loss = 'categorical_crossentropy', metrics = ["accuracy"])
    #=============================================================== 
    # Prints a description of the model  
    print(model.summary())
    return model

#%%
model = KerasClassifier(build_fn = ffnn_vanila)
grid = GridSearchCV(estimator = model, param_grid = param_grid, n_jobs = n_jobs, cv = 5)
grid_result = grid.fit(scaler_MinMax.fit_transform(X_train), Y_train_categ)

print("Best fit parameters \n")
print(grid_result.best_params_)
print("\n ====================== \n")
joblib.dump(grid_result.best_params_, ffnn_bst_fit_model_path)
#%%
# fit model
ffnn_model = ffnn_vanila(learning_rate = grid_result.best_params_['learning_rate'])
ffnn_history = ffnn_model.fit(scaler_MinMax.fit_transform(X_train_ffnn), Y_train_ffnn_categ, batch_size = batch_size, epochs = epochs, shuffle = True, validation_data = (scaler_MinMax.fit_transform(X_valid_ffnn), Y_valid_ffnn_categ), callbacks=[early_stopping_callback, callbacks, model_checkpoint_callback])

#%%
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

print("========================================================================\n")

#%%
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
#%%
