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


label = matchsTrain[['label']]
matchsTrain.drop(columns=['label', 'home_team_goal',
                          'away_team_goal'], inplace=True)

numTrain = matchsTrain.iloc[:,10:20]


fig3,ax3 = plt.subplots(figsize=(40,20))
ax = sns.boxplot(data=numTrain,fliersize=20)
ax.tick_params(labelsize=25)
plt.show()

q4 = numTrain['diff_build_up'].quantile(1)
q0 =  numTrain['diff_build_up'].quantile(0)

numTrain['diff_build_up'].describe()

q4 = numTrain['diff_def_overall'].quantile(1)
q0 =  numTrain['diff_def_overall'].quantile(0)


Q1 = numTrain['diff_build_up'].quantile(0.25)
Q3 = numTrain['diff_build_up'].quantile(0.75)
IQR = Q3 - Q1 #ecart-interquartile
    
borneSup = Q3 + 1.5 * IQR
borneInf = Q1 - 1.5 * IQR

matchsTrainWithoutOutliers = matchsTrain.copy()

for index, row in matchsTrainWithoutOutliers.iterrows():
    if( (row['diff_build_up'] > borneSup) | (row['diff_build_up'] < borneInf)) :
           matchsTrainWithoutOutliers =  matchsTrainWithoutOutliers.drop(row,axis=2)
            
    
    
tableIndexOutliers = list(filter(lambda x: x < borneInf,numTrain['diff_build_up']))
    

findOutliers(numTrain,'diff_build_up')


fig3,ax3 = plt.subplots(figsize=(40,20))
ax = sns.boxplot(data=numTrain,fliersize=20)
ax.tick_params(labelsize=25)
plt.show()


###################################
#        CLEANING DATA            #
###################################
print("*******Data Cleaning for the Train Set*******")
matchsTrainCleaned = Data_Cleaning(matchsTrain).run()
print("*******Data Cleaning for the Test Set*******")
# matchsTestCleaned = Data_Cleaning(matchsTest).run()


X_train, X_test, y_train, y_test = train_test_split(
    matchsTrainCleaned, label, random_state=5)


#### SANS PCA

models = {
    'AdaBoostClassifier': AdaBoostClassifier(base_estimator=RandomForestClassifier())
}

params = {
        'AdaBoostClassifier': {
            'n_estimators': random.randint(10,1000),
            'learning_rate': random.uniform(low=0, high=5, size = 10),
            'random_state': [1]
        }
        
}



helper = RandomizedSearchHelper(models, params)
helper.fit(X_train, y_train, scoring="accuracy", n_jobs=6)

scoring_table = helper.score_summary()
    
#### AVEC PCA



pca = PCA()

# define the pipe
pipe = Pipeline([
    ('pca', pca),
    ('AdaBoostClassifier',AdaBoostClassifier(base_estimator=RandomForestClassifier()))
])

models_pca = {
    'ada_pca': pipe
}

params_pca = {
    'ada_pca': {
        'pca__whiten': [True, False],
        'pca__n_components': random.randint(30, 41, 20),
        'ada__n_estimators': random.uniform(low=10, high=1000, size = 10),
        'ada__learning_rate': random.uniform(low=0, high=5, size = 10),
        'ada__random_state': [1]
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
