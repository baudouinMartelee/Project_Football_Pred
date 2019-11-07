# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 17:46:00 2019

@author: Baudouin
"""


import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import Imputer,OrdinalEncoder
from sklearn.preprocessing import LabelBinarizer,OneHotEncoder,MultiLabelBinarizer

from sklearn.base import BaseEstimator, TransformerMixin
class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names
    def fit(self, X, y=None):        
        return self
    def transform(self, X):
        return X[self.attribute_names].values
    
from sklearn.base import TransformerMixin #gives fit_transform method for free
class MyLabelBinarizer(TransformerMixin):
    def __init__(self, *args, **kwargs):
        self.encoder = LabelBinarizer(*args, **kwargs)
    def fit(self, x, y=0):
        self.encoder.fit(x)
        return self
    def transform(self, x, y=0):
        return self.encoder.transform(x)



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

#Droping irrelevent columns
matchs = matchs.drop(['country_id', 'league_id','date', 'match_api_id','home_team_api_id', 'away_team_api_id', 'Unnamed: 0'], axis=1)


###################################
#        CLEANING DATA            #
###################################        


# Récupérer toutes les features du type float64
numerical_data = matchs.select_dtypes("float64")

num_attribs = list(numerical_data)

#Changer le type des saisons d'objets à categorique
matchs[['season']] = matchs[['season']].apply(lambda x: x.astype('category'))
categorical_attrib = ['season']

#label
label = matchs[['label']]



# Utilisation de pipeline pour clean les data

num_pipeline = Pipeline([
            ('selector',DataFrameSelector(num_attribs)),
            ('imputer',Imputer(strategy="median")),
            ('min_max_scaler',MinMaxScaler()),
            ])
    


categorical_pipeline = Pipeline([
                ('selector',DataFrameSelector(categorical_attrib)),
                 ('label_binazer',MyLabelBinarizer()),
        ])

from sklearn.pipeline  import FeatureUnion
full_pipeline = FeatureUnion(transformer_list=[
            ("num_pipeline",num_pipeline),
            ("categorical_pipeline",categorical_pipeline)
        ])
    
#data are clean here
match_cleaned = full_pipeline.fit_transform(matchs)



###################################
#     TEST DIFFERENT MODELS       #
###################################  

##### SPLIT DATA   #####µ
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


X_train, X_test, y_train, y_test = train_test_split(match_cleaned,label,random_state = 5)




#####  RANDOM FOREST MODEL  ##### 

from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier()

rf.fit(X_train,y_train)

predicted_values_SVM = rf.predict(X_test)

print("score : ",rf.score(X_test,y_test))



#####  KNeighbors  ##### 

from sklearn.neighbors import KNeighborsClassifier

rf = KNeighborsClassifier()

rf.fit(X_train,y_train)

predicted_values_SVM = rf.predict(X_test)

print("score : ",rf.score(X_test,y_test))


#####  SVM MODEL  ##### 
from sklearn.svm import NuSVC

rf = NuSVC(gamma='scale')

rf.fit(X_train,y_train)

predicted_values_SVM = rf.predict(X_test)

print("score : ",rf.score(X_test,y_test))

#####  SGDSVM MODEL  ##### 
from sklearn.linear_model import SGDClassifier

rf = SGDClassifier(loss='hinge')


rf.fit(X_train,y_train)

predicted_values_SVM = rf.predict(X_test)

print("score : ",rf.score(X_test,y_test))

#####  SGDSVM MODEL AVEC GRIDSEARCH ##### 
from sklearn.model_selection import GridSearchCV

gamma = [0.1,0.5,1,3,5,8,10,25,50,100]
c = [0.1,0.5,1,3,5,8,10,25,50,100]
params_grid = [{'gamma':gamma,'C':c}]

gs = GridSearchCV(SVC(),param_grid=params_grid)

gs.fit(X_train,y_train.values.ravel())

best_param = gs.best_params_


#####  SVM MODEL  ##### 
from sklearn.svm import SVC

rf = SVC(kernel='rbf')


rf.fit(X_train,y_train)

predicted_values_SVM = rf.predict(X_test)

print("score : ",rf.score(X_test,y_test))

#####  SVM MODEL  AVEC GRIDSEARCH ##### 
from sklearn.model_selection import GridSearchCV

gamma = [0.1,0.5,1,3,5,8,10,25,50,100]
c = [0.1,0.5,1,3,5,8,10,25,50,100]
params_grid = [{'gamma':gamma,'C':c}]

gs = GridSearchCV(SVC(),param_grid=params_grid)

gs.fit(X_train,y_train.values.ravel())

best_param = gs.best_params_





#####  to Calcul scores  ##### 
from sklearn.model_selection import cross_val_score
 
rf_scores = cross_val_score(rf,X_train,y_train,scoring="neg_mean_squared_error",cv=10)
rf_rmse_scores = np.sqrt(-rf_scores)
rf_rmse_scores
def display_scores(scores):
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())
 
display_scores(rf_rmse_scores)









