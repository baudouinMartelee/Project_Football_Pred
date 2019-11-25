# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 08:39:09 2019

@author: Baudouin
"""

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV
from data_engineering import Data_Engineering
from data_cleaning import Data_Cleaning
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from estimator_selection_helper import EstimatorSelectionHelper
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.multiclass import OneVsRestClassifier
# from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import RidgeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.svm import SVC
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import warnings
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.pipeline import Pipeline
from numpy import random
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score
from randomized_search_helper import RandomizedSearchHelper
warnings.simplefilter("ignore")

##################################
#        GETTING THE DATA        #
##################################

matchsTrain = pd.read_csv('./csv/X_Train.csv')
matchsTest = pd.read_csv('./csv/X_Test.csv')
players = pd.read_csv('./csv/Player.csv')
teams = pd.read_csv('./csv/Team.csv')
team_attr = pd.read_csv('./csv/Team_Attributes.csv')
player_attr = pd.read_csv('./csv/Player_Attributes.csv')


###################################
#       DATA ENGINEERING          #
###################################


print("*******Data Engineering for the Train Set*******")
matchsTrain = Data_Engineering.add_labels(matchsTrain)
matchsTrain = Data_Engineering(
    matchsTrain, player_attr, teams, team_attr).run()


print("*******Data Engineering for the Test Set*******")
# matchsTest = Data_Engineering(
# matchsTest, player_attr, teams, team_attr, matchsTrain).run()





numTrain = matchsTrain.iloc[:,10:20]


fig3,ax3 = plt.subplots(figsize=(40,20))
ax = sns.boxplot(data=numTrain,fliersize=20)
ax.tick_params(labelsize=25)
plt.show()


def removeOutliers(dataset,feature):
    
    indexing = []
    
    Q1 = dataset[feature].quantile(0.25)
    Q3 = dataset[feature].quantile(0.75)
    IQR = Q3 - Q1 #ecart-interquartile
        
    borneSup = Q3 + 1.5 * IQR
    borneInf = Q1 - 1.5 * IQR
    
    
    for index, row in dataset.iterrows():
        if( (row[feature] > borneSup) | (row[feature] < borneInf)) :
            indexing.append(index)
    
    datasetWithoutOutliers = dataset.drop(indexing,axis=0)
    return datasetWithoutOutliers


tableOk = removeOutliers(matchsTrain,'diff_build_up')
tableOk2 = removeOutliers(tableOk,'diff_def_press')
tableOk3 = removeOutliers(tableOk2,'diff_def_overall')
tableOk4 = removeOutliers(tableOk3,'diff_mid_overall')
tableOk5 = removeOutliers(tableOk4,'diff_att_overall')
tableOk6 = removeOutliers(tableOk5,'diff_att_home_def_away')
tableOk7 = removeOutliers(tableOk6,'diff_def_home_att_away')
tableOk8 = removeOutliers(tableOk7,'diff_def_pot')
tableOk9 = removeOutliers(tableOk8,'diff_mid_pot')
tableOk10 = removeOutliers(tableOk9,'diff_att_pot')

matchsTrain2 = pd.DataFrame(tableOk10)

label = matchsTrain2[['label']]
matchsTrain2.drop(columns=['label', 'home_team_goal',
                          'away_team_goal'], inplace=True)

fig3,ax3 = plt.subplots(figsize=(40,20))
ax = sns.boxplot(data=matchsTrain2,fliersize=20)
ax.tick_params(labelsize=25)
plt.show()


###################################
#        CLEANING DATA            #
###################################
print("*******Data Cleaning for the Train Set*******")
matchsTrainCleaned = Data_Cleaning(matchsTrain2).run()
print("*******Data Cleaning for the Test Set*******")
# matchsTestCleaned = Data_Cleaning(matchsTest).run()


X_train, X_test, y_train, y_test = train_test_split(
    matchsTrainCleaned, label, random_state=5)


#### SANS PCA

models = {
    'KNN': KNeighborsClassifier()
}

params = {
       'KNN': {
        'n_neighbors': [1, 5, 10, 25, 50, 100],
        'weights': ['uniform', 'distance'],
        'algorithm': ['auto', 'ball_tree', 'kd_tree']},
        
}



helper = RandomizedSearchHelper(models, params)
helper.fit(X_train, y_train, scoring="accuracy", n_jobs=6)

scoring_table = helper.score_summary()
    
#### AVEC PCA



pca = PCA()

# define the pipe
pipe = Pipeline([
    ('pca', pca),
    ('RandomForestClassifier',RandomForestClassifier())
])

models_pca = {
    'ada_pca': pipe
}

params_pca = {
    'ada_pca': {
        'pca__whiten': [True, False],
        'pca__n_components': random.randint(30, 41, 20),
        'rdf__n_estimators': random.randint(50, 100),
        'rdf__max_depth': random.randint(40, 70),
        'rdf__min_samples_leaf': random.randint(25, 40),
        'rdf__max_features' : ['sqrt', 'log2', 'auto'],
        'rdf__min_samples_split': random.randint(2, 10),
    }
}

"""
grid = RandomizedSearchCV(
    pipe, params['rf_pca'], n_iter=20, scoring='accuracy', n_jobs=-1, cv=3, random_state=42)
grid.fit(X_train, y_train)
grid.score(X_train, y_train)
tr_pred, te_pred = grid.predict(X_train), grid.predict(X_test)
print("Train : " + str(accuracy_score(y_train, tr_pred)))
print("Test: " + str(accuracy_score(y_test, te_pred)))
grid.best_params_
"""

helper_pca = RandomizedSearchHelper(models_pca, params_pca)
helper_pca.fit(X_train, y_train, scoring="accuracy", n_jobs=6)

scoring_table_pca = helper_pca.score_summary()
