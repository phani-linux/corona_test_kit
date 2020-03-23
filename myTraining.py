# -*- coding: utf-8 -*-
"""
Created on Sun Mar 22 11:29:19 2020

@author: gundawar
"""
#%% Import Library
import pandas as pd
import numpy as np
import pickle
from sklearn.linear_model import LogisticRegression


def data_split(data,ratio):
    #%% Train Test Splliting
    np.random.seed(42)
    shuffled = np.random.permutation(len(data))
    test_set_size = int(len(data) * ratio)
    test_indices = shuffled[:test_set_size]
    train_indices = shuffled[test_set_size:]
    return data.iloc[train_indices],data.iloc[test_indices] 

if __name__ == "__main__":
    
    #%% Reading Data
    df = pd.read_csv('FluDB.csv')

    #%% Split Test-Train Data
    train, test = data_split(df, 0.2)

    #%%
    X_train = train[['Age', 'Gender', 'Temperature', 'MedicalConditions', 'RunningNose','Cough','Myalgia','Headache','ThroatAche','Fever','Fatigue','Vomiting']].to_numpy()
    X_test = test[['Age', 'Gender', 'Temperature', 'MedicalConditions', 'RunningNose','Cough','Myalgia','Headache','ThroatAche','Fever','Fatigue','Vomiting']].to_numpy()

    #%%
    Y_train = train[['FluTestStatus']].to_numpy().reshape(1360,)
    Y_test = test[['FluTestStatus']].to_numpy().reshape(339,)

    #%%
    clf = LogisticRegression()
    clf.fit(X_train, Y_train)

    # open a file, where you ant to store the data
    file = open('model.pkl', 'wb')

    # dump information to that file
    pickle.dump(clf, file)

    # close the file
    file.close()    
