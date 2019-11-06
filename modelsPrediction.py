# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 21:09:23 2019

@author: Baudouin
"""

###################################
#     TEST DIFFERENT MODELS       #
###################################  

##### SPLIT DATA   #####µ
import pandas as pd
from sklearn.model_selection import train_test_split


X_train, X_test, y_train, y_test = train_test_split(match_cleaned,moon_labels,random_state = 5)




#####  SVM MODEL  ##### 


