# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 17:03:35 2019

@author: AN20027664
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 16:27:04 2019

@author: AN20027664
"""

import numpy as nm
import pandas as pd
import matplotlib.pyplot as pt

dataset=pd.read_csv('Position_salary.csv')
x=dataset.iloc[:,1:2].values
y=dataset.iloc[:,2].values

#from sklearn.preprocessing import StandardScaler
#sc_x=StandardScaler()
#sc_y=StandardScaler()
#x=sc_x.fit_transform(x)
#y=sc_y.fit_transform(y)


#random forest function
from sklearn.ensemble import RandomForestRegressor 
regressor=RandomForestRegressor(n_estimatos=10, random_state=0)
regressor.fit(x,y)
#prediction
y_pred=regressor.predict(6.5)

#visualization
x_grid= nm.arange(min(x),max(y),0.01)
x_grid = x_grid.reshape((len(x_grid),1))
pt.scatter(x,y,color='blue')
pt.plot(x_grid,regressor.predict(x_grid),color='green')
pt.title('Truth Or Dare (random forest regression')
pt.xlabel('Position Level')
pt.ylabel('Salary')
pt.show()