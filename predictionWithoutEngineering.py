# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 17:46:00 2019

@author: Baudouin
"""


import sqlite3
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import Imputer

dat = sqlite3.connect('database.sqlite')

table = ["Country", "League", "X_Train", "Player", "Player_Attributes", 
         "Team", "Team_Attributes"]

csv = {}

for name in table:
  query = dat.execute("SELECT * From " + name)
  cols = [column[0] for column in query.description]
  results= pd.DataFrame.from_records(data = query.fetchall(), columns = cols)
  results.to_csv(r''+name+'.csv')

import pandas as pd

matchs = pd.read_csv('X_Train.csv')
players = pd.read_csv('Player.csv')
countries = pd.read_csv('Country.csv')
leagues = pd.read_csv('League.csv')
teams = pd.read_csv('Team.csv')
team_attr = pd.read_csv('Team_Attributes.csv')
player_attr = pd.read_csv('Player_Attributes.csv')


def det_label(score1, score2):
    if(score1 == score2):
        return 0
    if(score1 < score2):
        return -1
    else:
        return 1


# Create labels
matchs['label'] = matchs.apply(lambda row: det_label(
    row.home_team_goal, row.away_team_goal), axis=1)

matchs = matchs.drop(['country_id', 'league_id', 'match_api_id','home_team_api_id', 'away_team_api_id', 'Unnamed: 0'], axis=1)

matchs.shape

numerical_data = matchs.select_dtypes("float64")

num_attribs = list(numerical_data)


from sklearn.base import BaseEstimator, TransformerMixin
class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names
    def fit(self, X, y=None):        
        return self
    def transform(self, X):
        return X[self.attribute_names].values

num_pipeline = Pipeline([
            ('selector',DataFrameSelector(num_attribs)),
            ('imputer',Imputer(strategy="median")),
            ('min_max_scaler',MinMaxScaler()),
            ])

