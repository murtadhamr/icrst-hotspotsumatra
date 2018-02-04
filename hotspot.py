# -*- coding: utf-8 -*-
"""
Created on Sat Feb  3 00:41:51 2018

@author: Lenovo
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset2014 = pd.read_csv('hotspot_sumatra_2014.csv')
dataset2015 = pd.read_csv('hotspot_sumatra_2015.csv')

x_2014 = dataset2014.iloc[:, 0:3].values
y_2014 = dataset2014.iloc[:, 3].values

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X1 = LabelEncoder()
x_2014[:, 0] = labelencoder_X1.fit_transform(x_2014[:, 0])
labelencoder_X2 = LabelEncoder()
x_2014[:, 1] = labelencoder_X2.fit_transform(x_2014[:, 1])
labelencoder_X3 = LabelEncoder()
x_2014[:, 2] = labelencoder_X3.fit_transform(x_2014[:, 2].astype(str))
onehotencoder = OneHotEncoder(categorical_features = [0])
x_2014 = onehotencoder.fit_transform(x_2014).toarray()
x_2014 = x_2014[:, 1:]
onehotencoder2 = OneHotEncoder(categorical_features = [22])
x_2014 = onehotencoder2.fit_transform(x_2014).toarray()
x_2014 = x_2014[:, 1:]

y_2014[y_2014=="F"]=0
y_2014[y_2014=="T"]=1
y_2014 = y_2014.astype(int)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x_2014, y_2014, test_size = 0.2, random_state = 0)

from sklearn.ensemble import GradientBoostingClassifier
gbc = GradientBoostingClassifier()
gbc.fit(x_train, y_train)

gbc.predict(x_test)

from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, gbc.predict(x_test)))