#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 14 17:14:54 2022

@author: salehamariyam
"""

import pandas as pd
import numpy as np
from functools import reduce
import gc
import os
from sklearn.metrics import confusion_matrix, accuracy_score  

#%% 1 Working directory
print("==================================================\n")
print(os.getcwd())

# methylation
omic = "methylation"
path = "results"

os.chdir(path)
print(os.getcwd())
del path
gc.collect()

op_res_path = "accuracy_values_combined_chosen.csv"
#%% 2 Input paths

#=============================================================================================
# Train for combined data
acc_table_combined_results_train_path = "accuracy_values_combined_data_train.csv"
acc_table_train = pd.read_csv(acc_table_combined_results_train_path)
acc_table_train.head()
acc_table_train.columns = ["train_" + name for name in acc_table_train.columns]
# Retain values only for maximum accuracy
max_accurcay = max(acc_table_train['train_acc_model'])
acc_table_train = acc_table_train.loc[acc_table_train['train_acc_model'] == max_accurcay]
acc_table_train.head()

del acc_table_combined_results_train_path

#=============================================================================================
# Test for combined data
acc_table_combined_results_test_path = "accuracy_values_combined_data_test.csv"
acc_table_test = pd.read_csv(acc_table_combined_results_test_path)
acc_table_test.head()
acc_table_test.columns = ["test_" + name for name in acc_table_test.columns]
acc_table_test = acc_table_test['test_acc_model']

del acc_table_combined_results_test_path
gc.collect()

#%%
merged_accuracy_list = [acc_table_train, acc_table_test]
merged_accuracy  = reduce(lambda left,right: pd.merge(left, right, left_index = True, right_index = True), merged_accuracy_list)
merged_accuracy.iloc[0:5, 0:5]

merged_accuracy = merged_accuracy.loc[merged_accuracy["test_acc_model"] == max(merged_accuracy["test_acc_model"])]

merged_accuracy.to_csv(op_res_path, header = True, index = True)


#%%


















