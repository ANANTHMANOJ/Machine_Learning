# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 12:06:07 2019

@author: AN20027664
"""
import numpy as nm
import matplotlib.pyplot as pt
import pandas as pd
dataset=pd.read_csv('Salary_Data.csv')
x=dataset.iloc[:,:-1].values #[rows][columns]all columns except lastone here
y=dataset.iloc[:,1].values


from sklearn.model_selection import train_test_split
x_tr,x_ts,y_tr,y_ts=train_test_split(x,y,test_size=0.2,random_state= 0)

#simple linear regression algorithm
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_tr,y_tr)

# prediction or testing
y_pred= regressor.predict(x_ts)    # vector of prediction of dependent variable


#plotting on training
pt.scatter(x_tr,y_tr,color='blue')
pt.plot(x_tr,regressor.predict(x_tr),color='green')
pt.title('Sal_vs_Exp')
pt.xlabel('Yrs of Exp')
pt.ylabel('Sal')
pt.show()
 
#plotting on testing
pt.scatter(x_ts,y_ts,color='yellow')
pt.plot(x_tr,regressor.predict(x_tr),color='red')
pt.title('Sal_vs_Exp')
pt.xlabel('Yrs of Exp')
pt.ylabel('Sal')
pt.show()