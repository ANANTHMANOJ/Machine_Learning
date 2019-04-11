# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 12:14:22 2019

@author: AN20027664
"""

import numpy as nm
import pandas as pd
import matplotlib.pyplot as mt

dataset=pd.read_csv('Ads_CTR_Optimisation.csv') #this data is for simulations only
d=10
N=10000
#Ipper case bound method
import math as m
no_of_sel= [0]*d #vector of size d containing 0
sum_of_reward=[0]*d
sel_ad=[]
total_reward=0
for n in range(0,N):
    max_up_bound=0
    ad=0
    for i in range(0,d):
        if(no_of_sel[i]>0):
            avg_reward=sum_of_reward[i]/no_of_sel[i]
            del_i= m.sqrt(1.5*m.log(n)/no_of_sel[i])
            u_b=avg_reward+del_i
        else:
            u_b=1e400
        if u_b>max_up_bound:
            max_up_bound=u_b
            ad=i
        sel_ad.append(ad)
        no_of_sel[ad]=no_of_sel[ad]+1
        reward=dataset.values[n,ad]
        sum_of_reward[ad]=sum_of_reward[ad]+reward
        total_reward=total_reward+reward

#visualization:
mt.hist(sel_ad)
mt.title('Histogram of ada selections')
mt.xlabel('Ads')
mt.ylabel('No. of times each ad was selected')
        
        