# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 15:39:09 2019

@author: AN20027664
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 12:09:11 2019

@author: AN20027664
"""

import numpy as nm
import pandas as pd
import matplotlib.pyplot as pt

dataset=pd.read_csv('Wine.csv')
x=dataset.iloc[:,0:13].values
y=dataset.iloc[:,13].values

from sklearn.cross_validation import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.preprocessing import StandardScaler
sc_x=StandardScaler()
x_train=sc_x.fit_transform(x_train)
x_test=sc_x.transform(x_test)


#Appling PCA
from sklearn.decomposition import PCA
pca=PCA(n_components=2) #check with n_components=None
x_train=pca.fit_transform(x_train)
x_test=pca.transform(x_test)
experienced_varience=pca.explained_variance_ratio_ # for seeing which is most important

#logistic code
from sklearn.linear_model import LogisticRegression
classifier=LogisticRegression(random_state=0)
classifier.fit(x_train,y_train)

y_pred=classifier.predict(x_test)

#including confusion matrix
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)

#graphical precentation (for trainning session)
from matplotlib.colors import ListedColormap
x_set,y_set=x_train,y_train
x1,x2=nm.meshgrid(nm.arange(start=x_set[:,0].min-1,stop=x_set[:,0].max()+1,step=0.01),nm.arange(start=x_set[:,1].min-1,stop=x_set[:,1].max()+1,step=0.01))
pt.contourf(x1,x2,classifier.predict(nm.array[x1.ravel(),x2.ravel()].T).reshape(x1.shape),alpha=0.75,cmap=ListedColormap(('red','green')))
pt.xlim(x1.min(),x1.max())
pt.ylim(x2.min(),x2.max())
for i,j in enumerate(nm.unique(y_set)):
    pt.scatter(x_set[y_set==j,0],x_set[y_set==j,1],c=ListedColormap(('red','green'))(i),label=j)
pt.title('Logestic Regression')
pt.xlabel('Age')
pt.ylabel('Estimated Salary')
pt.legend()
pt.show()

from matplotlib.colors import ListedColormap
x_set,y_set=x_test,y_test
x1,x2=nm.meshgrid(nm.arange(start=x_set[:,0].min-1,stop=x_set[:,0].max()+1,step=0.01),nm.arange(start=x_set[:,1].min-1,stop=x_set[:,1].max()+1,step=0.01))
pt.contourf(x1,x2,classifier.predict(nm.array[x1.ravel(),x2.ravel()].T).reshape(x1.shape),alpha=0.75,cmap=ListedColormap(('red','green')))
pt.xlim(x1.min(),x1.max())
pt.ylim(x2.min(),x2.max())
for i,j in enumerate(nm.unique(y_set)):
    pt.scatter(x_set[y_set==j,0],x_set[y_set==j,1],c=ListedColormap(('red','green'))(i),label=j)
pt.title('Logestic Regression')
pt.xlabel('Age')
pt.ylabel('Estimated Salary')
pt.legend() # to show the point meaning in the corner
pt.show()

