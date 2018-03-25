# -*- coding: utf-8 -*-
"""
Created on Sat Mar 10 11:52:33 2018

@author: User
"""


import pandas as pd
import numpy as np
from pandas import read_csv
from sklearn.preprocessing import Imputer

#data = pd.read_csv("E:\\phase2\\dataset\\clevland.xlsx", header=None)
#print(data)
#data=open(clevland)
dataset = read_csv("F:\\phase2\\dataset\\devi.csv", header=None)
dataset=dataset.replace('?', np.NaN)
print(dataset)
#print(dataset.shape)
dff = pd.DataFrame(dataset)
print(dff)
# Create an imputer object that looks for 'Nan' values, then replaces them with the mean value of the feature by columns (axis=0)
mean_imputer = Imputer(missing_values='NaN', strategy='mean', axis=0)
# Train the imputor on the df dataset
mean_imputer = mean_imputer.fit(dataset)
# Apply the imputer to the df dataset
preprocessed = mean_imputer.transform(dataset.values)
preprocessed_df=pd.DataFrame(preprocessed)
print(preprocessed_df)
preprocessed_df.to_csv("F:\\phase2\\dataset\\deviprocessed.csv", index=False, header=False)
