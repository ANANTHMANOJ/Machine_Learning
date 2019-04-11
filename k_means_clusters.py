# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 12:26:31 2019

@author: AN20027664
"""
import numpy as np
import matplotlib.pyplot as pd
import pandas as pt

dataset =pt.read_csv('mall.csv')
x=dataset.iloc[:,[3,4]].values

#using elbow method for finding optimal no. of clusters
from sklearn.cluster import KMeans
wcss=[]
for i in range(1,11):
    kmeans=KMeans(n_clusters=i, init='k-means++',n_init=10,max_iter=300,random_state=0)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)#wcss is also called inertia
pt.plot(range(1,11),wcss)
pt.title('the elbow method')
pt.ylabel('wcss')
pt.xlabel('Number of clusters')
pt.show()

#applying k-means on dataset
kmeans=KMeans(n_clusters=5,init='k-means++',max_iter=300,n_init=10,random_state=0)
y_km=kmeans.predict(x)

#visualization 
pd.scatter(x[y_km==0,0],x[y_km==0,1],s=100,c='red',label='c1')
pd.scatter(x[y_km==1,0],x[y_km==1,1],s=100,c='cyan',label='c2')
pd.scatter(x[y_km==2,0],x[y_km==2,1],s=100,c='yellow',label='c3')
pd.scatter(x[y_km==3,0],x[y_km==3,1],s=100,c='green',label='c4')
pd.scatter(x[y_km==4,0],x[y_km==4,1],s=100,c='blue',label='c5')
pd.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],s=300,c='magenta',label='centroids')
pd.title('Cluster of clients')
pd.xlabel('annual income')
pd.ylabel('spending score')
pd.legend()
pd.show()

