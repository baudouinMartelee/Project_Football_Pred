from sklearn.metrics import confusion_matrix
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
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
import warnings
warnings.simplefilter("ignore")

##################################
#        GETTING THE DATA        #
##################################

matchsTrain = pd.read_csv('X_Train.csv')
matchsTest = pd.read_csv('X_Test.csv')
players = pd.read_csv('Player.csv')
teams = pd.read_csv('Team.csv')
team_attr = pd.read_csv('Team_Attributes.csv')
player_attr = pd.read_csv('Player_Attributes.csv')


###################################
#       DATA ENGINEERING          #
###################################
matchsTrain = Data_Engineering.add_labels(matchsTrain)
label = matchsTrain['label']


X_train, X_test, y_train, y_test = train_test_split(
    matchsTrain, label, test_size=0.33, random_state=42)

print("*******Data Engineering for the Train Set*******")
#matchsTrain = Data_Engineering.add_labels(matchsTrain)
X_train = Data_Engineering(
    X_train, player_attr, teams, team_attr).run()

print("*******Data Engineering for the Test Set*******")
X_test = Data_Engineering(
    X_test, player_attr, teams, team_attr, X_train).run()

correlation_train = X_train.corrwith(X_train['label'])
correlation_test = X_test.corrwith(X_test['label'])

X_train.drop(columns=['label', 'home_team_goal',
                      'away_team_goal'], inplace=True)

X_test.drop(columns=['label', 'home_team_goal',
                     'away_team_goal'], inplace=True)

""" self.matchs.drop(
            ['home_team_api_id', 'away_team_api_id'], axis=1, inplace=True)"""


###################################
#        CLEANING DATA            #
###################################
print("*******Data Cleaning for the Train Set*******")
X_train = Data_Cleaning(X_train).run()
print("*******Data Cleaning for the Test Set*******")
X_test = Data_Cleaning(X_test).run()


###################################
#           PREDICTIONS           #
###################################


# SPLIT DATA   #####µ


# Grid search with the Helper

"""'RandomForestClassifier': RandomForestClassifier(),
    'AdaBoostClassifier': AdaBoostClassifier(base_estimator=RandomForestClassifier()),
    'GradientBoostingClassifier': GradientBoostingClassifier(),
    'NB': GaussianNB(), ko
    'DT': DecisionTreeClassifier(), ko
    'SVM': SVC(),
    'XGBClassifier': XGBClassifier(),
    'SGDClassifier': SGDClassifier(),
    'KNN': KNeighborsClassifier() ko"""


models = {
    'SVM': SVC(),
    'SGDClassifier': SGDClassifier(),
    'RandomForestClassifier': RandomForestClassifier()
}

params = {
    'AdaBoostClassifier': {
        'n_estimators': [10, 100, 1000],
        'base_estimator__max_depth': [40, 50, 60, 100],
        'base_estimator__min_samples_leaf': [50],
        'base_estimator__max_features': ['sqrt', 'log2'],
        'base_estimator__min_samples_split': [2, 5, 10],
        'base_estimator__n_estimators': [100],
        'random_state': [1]},
    'XGBClassifier': {
        'objective': ['multi:softprob'],
        'max_depth': np.arange(5, 21),
        'n_estimators': [100, 200, 300, 400, 500],
        'learning_rate': [0.0001, 0.001, 0.01, 0.05, 0.1]},
    'RandomForestClassifier': {
        'n_estimators': [100],
        'max_depth': [40, 50, 60, 100],
        'min_samples_leaf': [50],
        'max_features': ['sqrt', 'log2'],
        'min_samples_split': [2, 5, 10],
        'random_state': [1]},
    'GradientBoostingClassifier': {
        'n_estimators': [1, 10, 100, 1000],
        'learning_rate': [0.001, 0.01, 0.05, 0.1, 0.5],
        'subsample': [0.1, 0.5, 1.0],
        'max_depth': [1, 3, 5, 10, 20, 50, 100],
        'random_state': [1]},
    'SVM': {
        'C': [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10],
        'kernel': ['linear'],
        'random_state': [1]},
    'SGDClassifier': {
        # learning rate
        'alpha': [0.001, 0.01, 0.03],
        'max_iter': [1000],  # number of epochs
        # 'loss': ['hinge'],  # logistic regression,
        'loss': ['hinge', 'log', 'modified_huber'],
        'penalty': ['l2', 'l1', 'none', 'elasticnet'],
        'random_state': [1]},
    'LogisticRegression': {
        'C': [0.00001, 0.0001, 0.001, 0.01, 0.1, 1], },
    'KNN': {
        'n_neighbors': [1, 5, 10, 25, 50, 100],
        'weights': ['uniform', 'distance'],
        'algorithm': ['auto', 'ball_tree', 'kd_tree']},
    'DT': {
        'criterion': ['gini', 'entropy'],
        'max_depth': [1, 2, 15, 20, 30, 40, 50],
        'max_features': ['sqrt', 'log2'],
        'min_samples_split': [2, 5, 10],
        'random_state': [1]},
    'NB': {},
}

helper = EstimatorSelectionHelper(models, params)
helper.fit(X_train, y_train, scoring='f1_micro', n_jobs=6)

scoring_table = helper.score_summary(sort_by='max_score')


###########ENSEMBLE MODEL###################


"""log_clf = LogisticRegression(
    C=helper.get_gs_best_params('LogisticRegression')['C'])"""
rnd_clf = RandomForestClassifier(
    max_depth=helper.get_gs_best_params('RandomForestClassifier')['max_depth'],
    min_samples_leaf=helper.get_gs_best_params('RandomForestClassifier')[
        'min_samples_leaf'],
    n_estimators=helper.get_gs_best_params(
        'RandomForestClassifier')['n_estimators'],
    max_features=helper.get_gs_best_params(
        'RandomForestClassifier')['max_features'],
    min_samples_split=helper.get_gs_best_params('RandomForestClassifier')[
        'min_samples_split'],
    random_state=helper.get_gs_best_params('RandomForestClassifier')['random_state'])
sgd_clf = SGDClassifier(
    loss=helper.get_gs_best_params('SGDClassifier')['loss'],
    penalty=helper.get_gs_best_params('SGDClassifier')['penalty'],
    alpha=helper.get_gs_best_params('SGDClassifier')['alpha'],
    max_iter=helper.get_gs_best_params('SGDClassifier')['max_iter'])
svc_clf = SVC(
    C=helper.get_gs_best_params('SVM')['C'],
    kernel=helper.get_gs_best_params('SVM')['kernel'])
"""gbc_clf = GradientBoostingClassifier(
    n_estimators=helper.get_gs_best_params(
        'GradientBoostingClassifier')['n_estimators'],
    learning_rate=helper.get_gs_best_params('GradientBoostingClassifier')['learning_rate'])"""

voting_clf = VotingClassifier(
    estimators=[('rnd', rnd_clf), ('sgd', sgd_clf), ('svc', svc_clf)], voting='hard', n_jobs=-1)

voting_clf.fit(X_train, y_train)

# predicted_values_SVM = voting_clf.predict(matchsTest)
predicted_values_SVM = voting_clf.predict(X_test)
print("score Ensemble Model: ", voting_clf.score(X_test, y_test))
confusion_matrix(y_test, predicted_values_SVM, labels=[1, 0, -1])

"""
match_soumission = pd.DataFrame(predicted_values_SVM)
match_soumission.to_csv(r"./predictionProjet1.csv")"""
