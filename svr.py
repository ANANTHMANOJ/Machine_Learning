# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 16:00:22 2019

@author: AN20027664
"""

import numpy as nm
import matplotlib.pyplot as pt
import pandas as pd

dataset=pd.read_csv('Position_salary.csv')
x=dataset.iloc[:,1:2].values
y=dataset.iloc[:,2].values

"""
from sklearn.cross_validation import train_test_split
x_tr,x_ts,y_tr,y_ts=train_test_split(x,y,test_size=0.2,random_state=0)
"""

#feature scaling
from sklearn.preprocessing import StandardScaler
sc_x=StandardScaler()
sc_y=StandardScaler()
x=sc_x.fit_transform(x)
y=sc_y.fit_transform(y)


#SVR function
from sklearn.svr import SVR
regressor = SVR(kernal='rbf')
regressor.fit(x,y)

#prediction
y_pred=sc_y.inverse_transform(regressor.predict(sc_x.transform(nm.array([[6.5]]))))

pt.scatter(x,y,color='blue')
pt.plot(x,regressor.predict(x),color='green')
pt.title('Truth Or Dare')
pt.xlabel('Position Level')
pt.ylabel('Salary')
pt.show()