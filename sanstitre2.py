# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 21:35:13 2019

@author: Baudouin
"""

from sklearn.base import TransformerMixin  # gives fit_transform method for free
from sklearn.preprocessing import LabelBinarizer, OneHotEncoder, MultiLabelBinarizer
from sklearn.preprocessing import Imputer, OrdinalEncoder
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.pipeline import Pipeline


class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[self.attribute_names].values


class MyLabelBinarizer(TransformerMixin):
    def __init__(self, *args, **kwargs):
        self.encoder = MultiLabelBinarizer(*args, **kwargs)

    def fit(self, x, y=0):
        self.encoder.fit(x)
        return self

    def transform(self, x, y=0):
        return self.encoder.transform(x)

import pandas as pd
from sklearn.pipeline import FeatureUnion

matchsTrain = pd.read_csv('./csv/matchsTrainFinal.csv')

matchsTrain.drop(columns=['Unnamed: 0','index'],inplace=True)


matchsTrainNumVariables = matchsTrain.iloc[:,7:16]

matchsTrainNumVariables.head()



import matplotlib.pyplot as plt
import seaborn as sns

## PLOTING BOXPLOT
fig3,ax3 = plt.subplots(figsize=(50,20))
ax = sns.boxplot(data=matchsTrainNumVariables,fliersize=20)
ax.tick_params(labelsize=20)
plt.show()


import seaborn as sns

for i in range(0,matchsTrainNumVariables.shape[1]):
    plt.figure(i)
    sns.distplot(matchsTrainNumVariables.iloc[:,i])
    plt.show()

