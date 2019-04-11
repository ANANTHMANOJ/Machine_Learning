# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 12:15:17 2019

@author: AN20027664
"""
#import keras before running this code

import numpy as nm
import matplotlib.pyplot as mt
import pandas as pd

dataset= pd.read_csv('Churn_Modelling.csv')
x=dataset.iloc[:,3:13].values
y=dataset.iloc[:,].values

#encoding the catogorical value sucn as country and gender
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
lb_1=LabelEncoder()
x[:,0]=lb_1.fit_transform(x[:,0])
lb_2=LabelEncoder()
x[:,1]=lb_2.fit_transform(x[:,1])
ohe=OneHotEncoder(categorical_features=[1])
x=ohe.fit_transform(x).toarray()
x=x[:,1:]


#splitting the training set an test set
from sklearn.cross_validation import train_test_split
x_tr,x_ts,y_tr,y_ts=train_test_split(x,y,test_size=.25,random_state=0)

#feature scaling
from sklearn.preprocessing import StandardScaler
stsc= StandardScaler()
x_tr = stsc.fit_transform(x_tr)
x_ts = stsc.fit_transform(y_tr)

#ANN coding
import keras
from keras.models import Sequential #to initail neural network
from keras.models import Dense #to add different layers in ANN
# 2 ways to defining in sequence of layers or defining by graph
# we are doing it in sequence
classifier=Sequential()
#input layer
# we have 11 indepent variable so 11 input nodes

classifier.add(Dense(output_dim=6,init='uniform',activation='relu',input_dim=11))  #this is an hidden layer init is for distribution of weight uniformly(here)
classifier.add(Dense(output_dim=6,init='uniform',activation='relu')) # this is 2nd hidden layer here input_dim is not required as it know the previous outputs of hidden layer
classifier.add(Dense(output_dim=6,init='uniform',activation='sigmoid'))#sigmod because its the output layer and has only 2 outputs. If it has more than 2 layer we should use softmax function
#for compiling

classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])#optimizer is for weight, adam is of stocastic gradient decent
classifier.fit(x_tr,y_tr,batch_size=10,nb_epoch=100) # epoch is number of times 

#predictor
y_pred=classifier.predict(x_ts)
y_pred=(y_pred>0.5)# this is done due to y_pred in previous step gives percentages and confusion matrix wants binary



#confusion matrix
from  sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_ts,y_pr)
