# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 14:31:04 2019

@author: AN20027664
"""

import numpy as nm
import pandas as pd
import matplotlib.pyplot as mt

dataset=pd.read_csv('Position_Salaries.csv')
x=dataset.iloc[:,1:2].values #1:2 is to get different element in list . See what it containd for more info
y=dataset.iloc[:,2].values

# here splitting of data into training and tesing data is not required as data set is small
# feature scaling is also not required

from sklearn.linear_model import LinearRegression # just for comparision
lin_reg1=LinearRegression()
lin_reg1.fit(x,y)

from sklearn.preprocessing import PolynomialFeatures
poly_reg=PolynomialFeatures(degree=2)
x_poly=poly_reg.fit_transform(x)
poly_reg.fit(x_poly,y)
lin_reg2=LinearRegression()
lin_reg2.fit(x_poly,y)



mt.scatter(x,y,color='red')
mt.plot(x,lin_reg1.predict(x),color='blue')
mt.title('True or false (Linear Regression)')
mt.xlabel('Position')
mt.ylabel('Salary')

x_grid=nm.arange(min(x),max(x),0.1)   #now x_grid is a vector ie a list
x_grid=nm.reshape(len(x_grid),1) #length and number of column as parameter, reshape is to make vector into matrix
mt.scatter(x,y,color='red')
mt.plot(x_grid,lin_reg2.predict(poly_reg.fit_transform(x_grid)),color='blue')
mt.title('True or false (Polynomial Linear Regression)')
mt.xlabel('Position')
mt.ylabel('Salary')

# prediction can be done by using predict 
lin_reg2.predict(poly_reg.fit_transform(x_grid))
