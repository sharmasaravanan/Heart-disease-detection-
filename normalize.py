# -*- coding: utf-8 -*-
"""
Created on Sat Mar 10 12:59:17 2018

@author: User
"""

from pandas import read_csv
import pandas as pd
from sklearn import preprocessing

dataset = read_csv("F:\\phase2\\dataset\\deviprocessed.csv", header=0)
print(dataset)
df=pd.DataFrame(dataset)
print(df)
# prepare data for normalization
values = df.values
# Create a minimum and maximum processor object
min_max_scaler = preprocessing.MinMaxScaler()

# Create an object to transform the data to fit minmax processor
x_scaled = min_max_scaler.fit_transform(values)
df_normalized = pd.DataFrame(x_scaled)
print(df_normalized)
df_normalized.to_csv("F:\\phase2\\dataset\\devinormalized.csv", index=False, header=False)  