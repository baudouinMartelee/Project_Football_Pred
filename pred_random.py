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


numTrain = matchsTrain.iloc[:,13:23]


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


numTrain2 = matchsTrain2.iloc[:,13:23]

fig3,ax3 = plt.subplots(figsize=(20,10))
ax = sns.boxplot(data=numTrain2,fliersize=20)
ax.tick_params(labelsize=25)
plt.show()


###################################
#        CLEANING DATA            #
###################################
print("*******Data Cleaning for the Train Set*******")
matchsTrainCleaned = Data_Cleaning(matchsTrain2).run()
print("*******Data Cleaning for the Test Set*******")
matchsTestCleaned = Data_Cleaning(matchsTest).run()


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
pipe_rf = Pipeline([
    ('rf', RandomForestClassifier(random_state=42))
])

pipe_log = Pipeline([
    ('log', LogisticRegression(random_state=42))
])

pipe_sgd = Pipeline([
    ('sgd', SGDClassifier(random_state=42))
])

pipe_ada = Pipeline([
    ('ada', AdaBoostClassifier(random_state=42))
])


models = {
    'ada_pca': pipe_ada
}

params = {
    'ada_pca': {
        'ada__n_estimators': random.randint(10, 2000, 50),
        'ada__learning_rate': random.uniform(low=0.0001, high=1, size=50),
    }
}

models = {
    'rf_pca': pipe_rf
}

params = {
    'rf_pca': {
        'rf__n_estimators': random.randint(1, 500, 50),
        'rf__max_depth': random.randint(1, 500, 50),
        'rf__max_features': ['sqrt', 'log2', 'auto'],
        'rf__min_samples_split': random.randint(1, 500, 50),
        'rf__min_samples_leaf': random.randint(1, 500, 50),
    }
}
"""
models = {
    'log_pca': pipe_log,
    'rf_pca': pipe_rf,
    'sgd_pca': pipe_sgd
}

params = {
    'log_pca': {
        'log__C': random.uniform(low=0.001, high=0.01, size=50),
        'log__penalty': ['l2'],
        'log__solver': ['lbfgs', 'saga'],
        'log__multi_class': ['multinomial']},
    'rf_pca': {
        'rf__n_estimators': random.randint(140, 350, 50),
        'rf__max_depth': random.randint(11, 400, 50),
        'rf__max_features': ['sqrt', 'log2', 'auto'],
        'rf__min_samples_split': random.randint(110, 400, 50),
        'rf__min_samples_leaf': random.randint(100, 300, 50)},
    'sgd_pca': {
        'sgd__alpha': random.uniform(low=0.0001, high=0.5, size=50),
        'sgd__max_iter': random.randint(1400, 5000, 50),
        'sgd__loss': ['hinge', 'log', 'modified_huber'],
        'sgd__penalty': ['l2', 'l1', 'none', 'elasticnet'],
    }

}"""

helper = RandomizedSearchHelper(models, params)
helper.fit(matchsTrainCleaned, label,
           scoring="accuracy", n_jobs=6, n_iter=50, cv=3)

# helper.fit(X_train, y_train,
# scoring="accuracy", n_jobs=6, n_iter=50, cv=3)

scoring_table = helper.score_summary()


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
        'rf_pca')['rf__max_depth'],
    min_samples_leaf=helper.get_gs_best_params(
        'rf_pca')['rf__min_samples_leaf'],
    n_estimators=helper.get_gs_best_params(
        'rf_pca')['rf__n_estimators'],
    max_features=helper.get_gs_best_params(
        'rf_pca')['rf__max_features'],
    min_samples_split=helper.get_gs_best_params(
        'rf_pca')['rf__min_samples_split'])

pipe_rf = Pipeline([
    ('rf', rf_clf)
])

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
    estimators=[('rnd', pipe_rf)], voting='hard', n_jobs=-1)

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
