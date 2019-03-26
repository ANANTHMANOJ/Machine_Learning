# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 10:40:09 2019

@author: AN20027664
"""

import numpy as nm
import matplotlib.pyplot as pt
import pandas as pd

dataset=pd.read_csv('Dataset.csv')
x=dataset.iloc[:,:-1].values #[rows][columns]all columns except lastone here
y=dataset.iloc[:,3].values

from sklearn.preprocessing import Imputer,LabelEncoder,OneHotEncoder #Ml models library Imputer helps in missing data labelENcoder to categorical variable
imputer=Imputer(missing_values="NaN",statergy="mean",axis=0)
imputer=imputer.fit(x[:,1:3]) #3 uper bound is not included but inner bound is included
x[:,1:3]=imputer.fit_transform(x[:,1:3])

labelencode_x=LabelEncoder()
x[:,0]=labelencode_x.fit_transform(x[:,0]) # to encode with label 
# the problem in above line is the label here can be used to compare by ML
#so we use dumy variable 
ohe=OneHotEncoder(categorical_features=[0]) # 0 represents which column
x=ohe.fit_transform[x].toarry()
labelencode_y=LabelEncoder()
y=labelencode_y.fit_transform(y)

#test set and training set
#from sklearn.cross_validation import train_test_split
from sklearn.model_selection import train_test_split
x_tr,x_ts,y_tr,y_ts=train_test_split(x,y,test_size=0.2,random_state= 0)#half data to test & half data to train

#feature scaling standardisation and normalisation is used to make sure no column value will dominate other column value in  euclidian distance.
from sklearn.preprocessing import StandardScaler
scale_x=StandardScaler()
x_tr= scale_x.fit_transform(x_tr)
x_ts=scale_x.transform(x_ts)

#missing value and categorical feature in optional







