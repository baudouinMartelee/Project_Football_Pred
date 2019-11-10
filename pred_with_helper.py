from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
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
from sklearn.svm import SVC
import warnings
warnings.simplefilter("ignore")


##################################
#        GETTING THE DATA        #
##################################

matchsTrain = pd.read_csv('X_Train.csv')
matchsTest = pd.read_csv('X_Test.csv')
players = pd.read_csv('Player.csv')
# countries = pd.read_csv('Country.csv')
# leagues = pd.read_csv('League.csv')
teams = pd.read_csv('Team.csv')
team_attr = pd.read_csv('Team_Attributes.csv')
player_attr = pd.read_csv('Player_Attributes.csv')


###################################
#       DATA ENGINEERING          #
###################################

print("*******Data Engineering for the Train Set*******")
matchsTrain = Data_Engineering(
    matchsTrain, player_attr, teams, team_attr).run()
matchsTrain = Data_Engineering.add_labels(matchsTrain)
print("*******Data Engineering for the Test Set*******")
# matchsTest = Data_Engineering(matchsTest, player_attr, teams,team_attr).run()


label = matchsTrain[['label']]
matchsTrain.drop(columns=['label', 'home_team_goal',
                          'away_team_goal'], inplace=True)


###################################
#        CLEANING DATA            #
###################################
print("*******Data Cleaning for the Train Set*******")
matchsTrain = Data_Cleaning(matchsTrain).run()
print("*******Data Cleaning for the Test Set*******")
#matchsTest = Data_Cleaning(matchsTest).run()


###################################
#           PREDICTIONS           #
###################################


# SPLIT DATA   #####µ


X_train, X_test, y_train, y_test = train_test_split(
    matchsTrain, label, random_state=5)


# Grid search with the Helper

models = {
    'RandomForestClassifier': RandomForestClassifier(n_jobs=-1),
    'GradientBoostingClassifier': GradientBoostingClassifier(),
    'SVC': SVC(gamma='auto', probability=False),
    'SGDClassifier': SGDClassifier(n_jobs=-1),
    'LogisticRegression': LogisticRegression(n_jobs=-1)
}

params = {
    'RandomForestClassifier': {'n_estimators': [16, 32], 'max_depth': np.arange(1, 21), 'min_samples_leaf': [1, 5, 10, 20, 50, 100]},
    'GradientBoostingClassifier': {'n_estimators': [4, 16, 32, 64, 200], 'learning_rate': [0.5, 0.25, 0.1, 0.05, 0.01]},
    'SVC': [
        {'kernel': ['linear'], 'C': [10, 100, 1000]},
        {'kernel': ['rbf'], 'C': [10, 100, 1000]}
    ],
    'SGDClassifier': {
        # learning rate
        'alpha': [0.001, 0.01, 0.03],
        'max_iter': [1000],  # number of epochs
        # 'loss': ['hinge'],  # logistic regression,
        'loss': ['hinge', 'log', 'modified_huber'],
        'penalty': ['l2', 'l1', 'none'],
    },
    'LogisticRegression': {
        'C': [0.00001, 0.0001, 0.001, 0.01, 0.1, 1],
    }
}

helper = EstimatorSelectionHelper(models, params)
helper.fit(X_train, y_train, scoring='f1_micro', n_jobs=4)

scoring_table = helper.score_summary(sort_by='max_score')


###########ENSEMBLE MODEL###################


"""log_clf = LogisticRegression(
    C=helper.get_gs_best_params('LogisticRegression')['C'])
rnd_clf = RandomForestClassifier(
    max_depth=helper.get_gs_best_params('RandomForestClassifier')['max_depth'],
    min_samples_leaf=helper.get_gs_best_params('RandomForestClassifier')['min_samples_leaf'])"""
sgd_clf = SGDClassifier(
    loss=helper.get_gs_best_params('SGDClassifier')['loss'],
    penalty=helper.get_gs_best_params('SGDClassifier')['penalty'],
    alpha=helper.get_gs_best_params('SGDClassifier')['alpha'],
    max_iter=helper.get_gs_best_params('SGDClassifier')['max_iter'])
svc_clf = SVC(
    C=helper.get_gs_best_params('SVC')['C'],
    kernel=helper.get_gs_best_params('SVC')['kernel'])
gbc_clf = GradientBoostingClassifier(
    n_estimators=helper.get_gs_best_params(
        'GradientBoostingClassifier')['n_estimators'],
    learning_rate=helper.get_gs_best_params('GradientBoostingClassifier')['learning_rate'])

voting_clf = VotingClassifier(
    estimators=[('gbc', gbc_clf), ('sgd', sgd_clf), ('svc', svc_clf)], voting='hard', n_jobs=-1)

voting_clf.fit(X_train, y_train)

predicted_values_SVM = voting_clf.predict(X_test)

print("score Ensemble Model: ", voting_clf.score(X_test, y_test))
