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
from sklearn.naive_bayes import GaussianNB, MultinomialNB
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
from scipy.stats import uniform
from xgboost import XGBClassifier
from sklearn.feature_selection import SelectFromModel
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
matchsTest = Data_Engineering(
    matchsTest, player_attr, teams, team_attr, matchsTrain).run()


label = matchsTrain[['label']]
matchsTrain.drop(columns=['label', 'home_team_goal',
                          'away_team_goal'], inplace=True)


###################################
#        CLEANING DATA            #
###################################
print("*******Data Cleaning for the Train Set*******")
matchsTrainCleaned = Data_Cleaning(matchsTrain).run()
print("*******Data Cleaning for the Test Set*******")
matchsTestCleaned = Data_Cleaning(matchsTest).run()


"""X_train, X_test, y_train, y_test = train_test_split(
    matchsTrainCleaned, label, random_state=5)"""


models = {
    'rf': RandomForestClassifier(random_state=42)
}

params = {
    'rf': {
        'n_estimators': random.randint(140, 350, 30),
        'max_depth': random.randint(11, 400, 30),
        'max_features': random.randint(1, 23, 30),
        'min_samples_split': random.randint(110, 400, 30),
        'min_samples_leaf': random.randint(100, 300, 30),

    }
}


helper = RandomizedSearchHelper(models, params)
helper.fit(matchsTrainCleaned, label,
           scoring="accuracy", n_jobs=6, n_iter=50, cv=3)

# helper.fit(X_train, y_train,
# scoring="accuracy", n_jobs=6, n_iter=50, cv=3)

scoring_table = helper.score_summary()
params['rf'].keys()

"""g = sns.FacetGrid(scoring_table, col="mean_score", hue="mean_score")
for i in params['rf'].keys():
    g.map(sns.scatterplot, i, hist=False, rug=True)"""

"""scoring_table.subplot(x='mean_score', y=[
                   'action', 'comedy'], figsize=(10, 5), grid=True)"""


###########ENSEMBLE MODEL###################


log_clf = LogisticRegression(
    C=helper.get_gs_best_params('log_pca')['log__C'],
    penalty=helper.get_gs_best_params('log_pca')['log__penalty'],
    solver=helper.get_gs_best_params('log_pca')['log__solver'],
    multi_class=helper.get_gs_best_params('log_pca')['log__multi_class'])

pipe_log = Pipeline([
    ('log', log_clf)
])
rf_clf = RandomForestClassifier(
    max_depth=helper.get_gs_best_params(
        'rf')['max_depth'],
    min_samples_leaf=helper.get_gs_best_params(
        'rf')['min_samples_leaf'],
    n_estimators=helper.get_gs_best_params(
        'rf')['n_estimators'],
    max_features=helper.get_gs_best_params(
        'rf')['max_features'],
    min_samples_split=helper.get_gs_best_params(
        'rf')['min_samples_split'],
    criterion=helper.get_gs_best_params(
        'rf')['criterion'])


sgd_clf = SGDClassifier(
    loss=helper.get_gs_best_params('sgd_pca')['sgd__loss'],
    alpha=helper.get_gs_best_params('sgd_pca')['sgd__alpha'],
    max_iter=helper.get_gs_best_params('sgd_pca')['sgd__max_iter'],
    penalty=helper.get_gs_best_params('sgd_pca')['sgd__penalty'],
)
pipe_sgd = Pipeline([
    ('sgd', sgd_clf)
])
voting_clf = VotingClassifier(
    estimators=[('rnd', rf_clf)], voting='hard', n_jobs=-1)

"""
voting_clf.fit(X_train, y_train)

# predicted_values_SVM = voting_clf.predict(matchsTest)
predicted_values_SVM = voting_clf.predict(X_test)
print("score Ensemble Model: ", voting_clf.score(X_test, y_test))
confusion_matrix(y_test, predicted_values_SVM, labels=[1, 0, -1])
"""


voting_clf.fit(matchsTrainCleaned, label)

#predicted_values_SVM = voting_clf.predict(matchsTest)
predicted_values_SVM = voting_clf.predict(matchsTestCleaned)

print("score Ensemble Model: ", voting_clf.score(X_test, y_test))
confusion_matrix(y_test, predicted_values_SVM, labels=[1, 0, -1])

match_soumission = pd.DataFrame(predicted_values_SVM)
match_soumission.to_csv(r"./predictionProjet1.csv")

#########################################
