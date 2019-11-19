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

"""
from matplotlib import pyplot as plt
import seaborn as sns

plt.figure(figsize=(40,15))
sns.heatmap(matchsTrain.corr(),annot=True,cmap='coolwarm', linewidths=.5)
#matchsTrain.to_csv(r'./csv/matchsTrainFinal.csv')
correlation = matchsTrain.corr()
"""
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
matchsTrainCleaned = Data_Cleaning(matchsTrain).run()
print("*******Data Cleaning for the Test Set*******")
# matchsTestCleaned = Data_Cleaning(matchsTest).run()

# PCA
"""

pca = PCA(n_components=30)
pca.fit_transform(matchsTrain)

plt.figure()
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('Number of Components')
plt.ylabel('Variance (%)')  # for each component
plt.title('Pulsar Dataset Explained Variance')
plt.show()

pca = PCA(n_components=29)
matchsTrainPca = pca.fit_transform(matchsTrain)"""

###################################
#           PREDICTIONS           #
###################################


# SPLIT DATA   #####µ


X_train, X_test, y_train, y_test = train_test_split(
    matchsTrainCleaned, label, random_state=5)


# Grid search with the Helper

"""'RandomForestClassifier': RandomForestClassifier(),
    'AdaBoostClassifier': AdaBoostClassifier(base_estimator=RandomForestClassifier()),
    'GradientBoostingClassifier': GradientBoostingClassifier(),
    'NB': GaussianNB(), ko
    'DT': DecisionTreeClassifier(), ko
    'SVM': SVC(),
    'XGBClassifier': XGBClassifier(),
    'SGDClassifier': SGDClassifier(),
    'KNN': KNeighborsClassifier(),
    'RidgeClassifier' : RidgeClassifier(),
    'BaggingClassifier' : BaggingClassifier(RandomForestClassifier()),
    'LogisticRegression' : LogisticRegression(solver='lbfgs', multi_class='multinomial',random_state=1)"""


"""TEST 0.5238
'LogisticRegression': LogisticRegression(multi_class='multinomial', random_state=1),
    'RandomForestClassifier': RandomForestClassifier(class_weight = 'balanced'),
    'SGDClassifier': SGDClassifier(),
    'SVM': SVC(),
    'RandomForestClassifier': {
        'n_estimators': [50, 75, 100],
        'max_depth': [40, 50, 60, 70],
        'min_samples_leaf': [25, 30, 35, 40],
        'max_features': ['sqrt', 'log2', 'auto'],
        'min_samples_split': [2, 5, 10]
    """

"""
params = {
    'RidgeClassifier': {},
    'BaggingClassifier': {
        'n_estimators': [10, 100, 1000],
        'base_estimator__max_depth': [40, 50, 60, 100],
        'base_estimator__min_samples_leaf': [50],
        'base_estimator__max_features': ['sqrt', 'log2'],
        'base_estimator__min_samples_split': [2, 5, 10],
        'base_estimator__n_estimators': [100],
        'random_state': [1]},
    'AdaBoostClassifier': {
        'n_estimators': [10, 100, 1000],
        'learning_rate': np.linspace(0.5, 2, 5),
        'random_state': [1]},
    'XGBClassifier': {
        'objective': ['multi:softprob'],
        'max_depth': [5, 10, 15, 20],
        'n_estimators': [100, 200, 300, 400, 500],
        'learning_rate': [0.0001, 0.001, 0.01, 0.05, 0.1]},
    'RandomForestClassifier': {
        'n_estimators': [50, 75, 100],
        'max_depth': [50, 60, 70, 100],
        'min_samples_leaf': [30, 40, 50, 60, 70],
        'max_features': ['sqrt', 'log2', 'auto'],
        'min_samples_split': [2, 5, 10, 15]},
    'GradientBoostingClassifier': {
        'learning_rate': [0.15, 0.1, 0.05, 0.01, 0.005, 0.001],
        'n_estimators': [100, 250, 500, 750, 1000, 1250, 1500, 1750]},
    'SVM': {
        'C': [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10],
        'kernel': ['linear'],
        'random_state': [1]},
    'SGDClassifier': {
        # learning rate
        'alpha': [0.00001, 0.0001, 0.001, 0.01, 0.03],
        'max_iter': [1000],  # number of epochs
        # 'loss': ['hinge'],  # logistic regression,
        'loss': ['hinge', 'log', 'modified_huber'],
        'penalty': ['l2', 'l1', 'none', 'elasticnet']},
    'LogisticRegression': {
        'C': [0.00001, 0.0001, 0.001, 0.01, 0.1, 1],
        'penalty': ['l2'],
        'solver': ['lbfgs', 'saga']},
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
    'NB': {
        'var_smoothing': np.logspace(0, -9, num=100)
    },
    
}"""

models = {
    'RandomForestClassifier': RandomForestClassifier(class_weight='balanced'),
}

params = {
    'RandomForestClassifier': {
        'n_estimators': [50, 75, 100],
        'max_depth': [50, 60, 70, 100],
        'min_samples_leaf': [30, 40, 50, 60, 70],
        'max_features': ['sqrt', 'log2', 'auto'],
        'min_samples_split': [2, 5, 10, 15]},
}

print(params.get('RandomForestClassifier'))

helper = EstimatorSelectionHelper(models, params)
helper.fit(X_train, y_train, scoring="f1_micro", n_jobs=6)

scoring_table = helper.score_summary()

#t_scoring = scoring_table.T

import seaborn as sns

ax = sns.lineplot(data= params.get('RandomForestClassifier').get('max_depth')) 

rdf = RandomForestClassifier(
    max_depth=helper.get_gs_best_params(
        'RandomForestClassifier')['max_depth'],
    min_samples_leaf=helper.get_gs_best_params('RandomForestClassifier')[
        'min_samples_leaf'],
    n_estimators=helper.get_gs_best_params(
        'RandomForestClassifier')['n_estimators'],
    max_features=helper.get_gs_best_params(
        'RandomForestClassifier')['max_features'],
    min_samples_split=helper.get_gs_best_params('RandomForestClassifier')[
        'min_samples_split'])


rdf.fit(X_train, y_train)

predicted_values_SVM = rdf.predict(X_test)

print("score Random Forest Model: ", rdf.score(X_test, y_test))


###########ENSEMBLE MODEL###################


log_clf = LogisticRegression(
    C=helper.get_gs_best_params('LogisticRegression')['C'],
    penalty=helper.get_gs_best_params('LogisticRegression')['penalty'],
    solver=helper.get_gs_best_params('LogisticRegression')['solver'])
rnd_clf = RandomForestClassifier(
    max_depth=helper.get_gs_best_params(
        'RandomForestClassifier')['max_depth'],
    min_samples_leaf=helper.get_gs_best_params('RandomForestClassifier')[
        'min_samples_leaf'],
    n_estimators=helper.get_gs_best_params(
        'RandomForestClassifier')['n_estimators'],
    max_features=helper.get_gs_best_params(
        'RandomForestClassifier')['max_features'],
    min_samples_split=helper.get_gs_best_params('RandomForestClassifier')[
        'min_samples_split'])
sgd_clf = SGDClassifier(
    loss=helper.get_gs_best_params('SGDClassifier')['loss'],
    alpha=helper.get_gs_best_params('SGDClassifier')['alpha'],
    max_iter=helper.get_gs_best_params('SGDClassifier')['max_iter'],
    random_state=1
)
nb_clf = GaussianNB()

svc_clf = SVC(
    C=helper.get_gs_best_params('SVM')['C'],
    kernel=helper.get_gs_best_params('SVM')['kernel'])
"""gbc_clf = GradientBoostingClassifier(
    n_estimators=helper.get_gs_best_params(
        'GradientBoostingClassifier')['n_estimators'],
    learning_rate=helper.get_gs_best_params('GradientBoostingClassifier')['learning_rate'])

voting_clf = VotingClassifier(
    estimators=[('rnd', rnd_clf), ('sgd', sgd_clf)], voting='hard', n_jobs=-1)

voting_clf.fit(X_train, y_train)

# predicted_values_SVM = voting_clf.predict(matchsTest)
predicted_values_SVM = voting_clf.predict(X_test)
print("score Ensemble Model: ", voting_clf.score(X_test, y_test))
confusion_matrix(y_test, predicted_values_SVM, labels=[1, 0, -1])

match_soumission = pd.DataFrame(predicted_values_SVM)
match_soumission.to_csv(r"./predictionProjet1.csv")"""
