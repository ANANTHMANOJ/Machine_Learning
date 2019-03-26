# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 13:58:36 2019

@author: AN20027664
"""

import numpy as nm
import matplotlib.pyplot as pt
import pandas as pd
dataset=pd.read_csv('50_startups.csv')
x=dataset.iloc[:,:-1].values #[rows][columns]all columns except lastone here
y=dataset.iloc[:,4].values

from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelencoder_x= LabelEncoder()
x[:,3]=labelencoder_x.fit_transform(x[:,3])
ohe=OneHotEncoder(categorial_feature=[3])
x=ohe.fit_transform(x).toarray()

#for avoiding the dummy variable trap
x=x[:,1:] 

from sklearn.model_selection import  train_test_split

x_tr,x_ts,y_tr,y_ts=train_test_split(x,y,test_size=0.2,random_state= 0)
  
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_tr,y_tr)
y_pred=regressor.predict(x_ts)
 

#backward elimination regression
import statsmodels.formula.api as sm
x=nm.append(arr=nm.ones((50,1)).astype(int),values=x,axiz=1)
x_opt=
