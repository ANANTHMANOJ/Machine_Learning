# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 17:08:26 2019

@author: AN20027664
"""

import numpy as np
import matplotlib.pyplot as pt
import pandas as pd

#we have restuarant.csv and restuarant.tsv file but we are using tsv file as csv is comma seperated file and it will be a problem
dataset=pd.read_csv('Restuarant_Reviews.tsv',delimiter='\t',quotes=3)

#cleaning the data for removing unwanted words
import re
import nltk # package for NLP
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus=[]
for i in range(0,1000):
    review=re.sub('[^a-bA-B]',' ',dataset['Review'][i])
    review=review.lower() #converting all letters to lower but check what will happen while using swapcase
    review=review.split()
    ps = PorterStemmer()
    review=[ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    #steming process
    #it is keeping the root of the word like love for loved loving etc
    review=' '.join(review)
    corpus.append(review)

"""
Creating bag of word model
It is taking all different and unique words in different columns
 and the rows will be each of the reviews
 cleaning the reviews helps in reducing the columns
 and then this is tokenised
 """
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(max_features=1500) # most frequently used words
x=cv.fit_transform(corpus).toarray()
y=dataset.iloc[:,1].values


from sklearn.cross_validation import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=0)

"""
feature scaling is not required here as input is 1 and 0
from sklearn.preprocessing import StandardScaler
sc_x=StandardScaler()
x_train=sc_x.fit_transform(x_train)
x_test=sc_x.transform(x_test)
"""

#logistic code
from sklearn.naive_bayes import GaussianNB
classifier=GaussianNB()
classifier.fit(x_train,y_train)

y_pred=classifier.predict(x_test)

#including confusion matrix
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)

