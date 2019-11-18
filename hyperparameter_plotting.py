
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

# matchsTrain.to_csv(r'./matchsTrainFinal.csv')
# correlation = matchsTrain.corrwith(matchsTrain['label'])

print("*******Data Engineering for the Test Set*******")
# matchsTest = Data_Engineering(
# matchsTest, player_attr, teams, team_attr, matchsTrain).run()

# matchsTrain.to_csv(r'./matchsTrainEngineerd.csv')
# matchsTest.to_csv(r'./matchsTestEngineerd.csv')

label = matchsTrain[['label']]
matchsTrain.drop(columns=['label', 'home_team_goal',
                          'away_team_goal'], inplace=True)

###################################
#        CLEANING DATA            #
###################################
print("*******Data Cleaning for the Train Set*******")
matchsTrain = Data_Cleaning(matchsTrain).run()
print("*******Data Cleaning for the Test Set*******")
# matchsTestCleaned = Data_Cleaning(matchsTest).run()

###################################
#           PREDICTIONS           #
###################################


# SPLIT DATA   #####µ


X_train, X_test, y_train, y_test = train_test_split(
    matchsTrain, label, random_state=5)


from sklearn.model_selection import GridSearchCV

from sklearn.ensemble import RandomForestClassifier


num_leafs = [1, 5, 10, 20, 50, 100]
n_estimators = [50, 75, 100]
max_depth = [50, 60, 70, 100]
min_samples_leaf = [30, 40, 50, 60, 70]
max_features = ['sqrt', 'log2', 'auto']
min_samples_split = [2, 5, 10, 15]



params_grid = [{'max_depth':max_depth,'min_samples_leaf':min_samples_leaf,'n_estimators':n_estimators,
                'max_features':max_features,'min_samples_split':min_samples_split}]


gs = GridSearchCV(RandomForestClassifier(),param_grid=params_grid)

gs.fit(X_train,y_train)


predicted_values_rdf = gs.predict(X_test)

print("score Random Forest Model: ", gs.score(X_test, y_test))
