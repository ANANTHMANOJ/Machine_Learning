# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 15:08:17 2019

@author: AN20027664
"""

import numpy as nm
import pandas as pd
import matplotlib.pyplot as mt

dataset=pd.read_csv('Ads_CTR_Optimisation.csv') #this data is for simulations only
d=10
N=10000
#Ipper case bound method
import random
no_of_rewards_1= [0]*d #vector of size d containing 0
no_of_rewards_0=[0]*d
sel_ad=[]
total_reward=0
for n in range(0,N):
    max_rand=0
    ad=0
    for i in range(0,d):
        rand_beta=random.betavariate(no_of_rewards_1[i]+1,no_of_rewards_0[i]+1)
        if rand_beta>max_rand:
            max_rand=rand_beta
            ad=i
        sel_ad.append(ad)
        reward=dataset.values[n,ad]
        if reward==1:
            no_of_rewards_1[ad]=no_of_rewards_1[ad]+1
        else:
            no_of_rewards_0[ad]=no_of_rewards_0[ad]+1
        total_reward=total_reward+reward

#visualization:
mt.hist(sel_ad)
mt.title('Histogram of ada selections')
mt.xlabel('Ads')
mt.ylabel('No. of times each ad was selected')
        