# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 14:34:32 2019

@author: AN20027664
"""

import numpy as np
import matplotlib.pyplot as pd
import pandas as pt

dataset =pt.read_csv('mall.csv')
x=dataset.iloc[:,[3,4]].values

#using dendrogram of optimal no. of clusters
import scipy.cluster.hierarchy as sch
dendrogram=sch.dendrogram(sch.linkage(x,method='ward'))#linkage is for heirarchical clustering, and ward is for minimise the variants within each clusters
pd.title('dendrogram')
pd.xlabel('customers')     
pd.ylabel('euclidean distances')
pd.show()

#fitting the dataset
from sklearn.cluster import AgglomerativeClustering
hc=AgglomerativeClustering(n_cluster=5,affinty='eucliden',linkage='ward')
y_hc=hc.fit_predict(x)

#visualisation
pd.scatter(x[y_hc==0,0],x[y_hc==0,1],s=100,c='red',label='c1')
pd.scatter(x[y_hc==1,0],x[y_hc==1,1],s=100,c='cyan',label='c2')
pd.scatter(x[y_hc==2,0],x[y_hc==2,1],s=100,c='yellow',label='c3')
pd.scatter(x[y_hc==3,0],x[y_hc==3,1],s=100,c='green',label='c4')
pd.scatter(x[y_hc==4,0],x[y_hc==4,1],s=100,c='blue',label='c5')
pd.title('Cluster of clients')
pd.xlabel('annual income')
pd.ylabel('spending score')
pd.legend()
pd.show()
