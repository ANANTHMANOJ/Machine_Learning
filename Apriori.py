# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 11:48:06 2019

@author: AN20027664
"""
# the apyori.py file should be downloaded and kept in this folder along with data side

import numpy as np
import matplotlib.pyplot as pd
import pandas as pt

dataset =pt.read_csv('Market_Basket_Optimisation.csv',header=None) #header=none to take first line as data
transaction=[]
for i in range(0,7501):
    transactions.append([str(dataset.values[i,j]) for j in range(0,20)])

#Training Apriori
from apyori import apriori
rules=apriori(transactions, min_support=003,min_confidence=0.2,min_lift=3,min_length=2)

#visualization
results=list(rules)
