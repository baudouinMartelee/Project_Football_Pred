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
matchsTrain = Data_Engineering(matchsTrain, player_attr, teams,team_attr).run()
matchsTrain = Data_Engineering.add_labels(matchsTrain)
print("*******Data Engineering for the Test Set*******")
matchsTest = Data_Engineering(matchsTest, player_attr, teams,team_attr).run()


label = matchsTrain[['label']]
matchsTrain.drop(columns=['label', 'home_team_goal',
                          'away_team_goal'], inplace=True)


###################################
#        CLEANING DATA            #
###################################
print("*******Data Cleaning for the Train Set*******")
matchsTrain = Data_Cleaning(matchsTrain).run()
print("*******Data Cleaning for the Test Set*******")
matchsTest = Data_Cleaning(matchsTest).run()


###################################
#           PREDICTIONS           #
###################################


# SPLIT DATA   #####µ


X_train, X_test, y_train, y_test = train_test_split(
    matchsTrain, label, random_state=5)


#####  SGDSVM MODEL ON TRAINING SET #####

rf = SGDClassifier(loss='hinge')

rf.fit(X_train, y_train)

predicted_values_SVM = rf.predict(X_test)

print("score SGDSVM MODEL : ", rf.score(X_test, y_test))

#####  SGDSVM MODEL ON TRAINING SET WITH GRIDSEARCH #####

# Sur SGDClassifier

grid = {
    # learning rate
    'alpha': [0.00001, 0.0001, 0.001, 0.01, 0.03, 0.1, 0.3, 0.5, 1],
    'max_iter': [1000],  # number of epochs
    'loss': ['hinge']  # logistic regression,
}

gs = GridSearchCV(SGDClassifier(), param_grid=grid)

gs.fit(X_train, y_train.values.ravel())

best_paramSGD = gs.best_params_

# Sur RandomForest
depths = np.arange(1, 21)
num_leafs = [1, 5, 10, 20, 50, 100]
params_grid = [{'max_depth': depths, 'min_samples_leaf': num_leafs}]

gs2 = GridSearchCV(RandomForestClassifier(), param_grid=params_grid)

gs2.fit(X_train, y_train.values.ravel())

best_param_RF = gs2.best_params_

# Sur LogisticRegression

grid2 = {
    'C': [0.00000001, 0.0000001, 0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1],
}

gs3 = GridSearchCV(LogisticRegression(), param_grid=grid2)

gs3.fit(X_train, y_train.values.ravel())

best_param_Logistique = gs3.best_params_


#####  Ensemble model  #####

""" Before
log_clf = LogisticRegression(C=0.1)
rnd_clf = RandomForestClassifier(max_depth=12, min_samples_leaf=50)
sgd_clf = SGDClassifier(loss='hinge', alpha=0.0001, max_iter=1000)
After
"""
log_clf = LogisticRegression(C=best_param_Logistique['C'])
rnd_clf = RandomForestClassifier(max_depth=best_param_RF['max_depth'], min_samples_leaf=best_param_RF['min_samples_leaf'])
sgd_clf = SGDClassifier(loss=best_paramSGD['loss'], alpha=best_paramSGD['alpha'], max_iter=best_paramSGD['max_iter'])

voting_clf = VotingClassifier(
    estimators=[('lr', log_clf), ('rf', rnd_clf), ('sgd', sgd_clf)], voting='hard')

voting_clf.fit(X_train, y_train)

predicted_values_SVM = voting_clf.predict(X_test)

print("score Ensemble Model: ", voting_clf.score(X_test, y_test))


###################################
#     PREDICTIONS ON TEST SET     #
###################################
"""
X_train, X_test, y_train, y_test = train_test_split(
    matchsTrain, label, random_state=5)

#####  SGDSVM MODEL  #####

rf = SGDClassifier(loss='hinge')

rf.fit(X_train, y_train)

predicted_values_SVM = rf.predict(X_test)

print("score : ", rf.score(X_test, y_test))


match_soumission = pd.DataFrame(predicted_values_SVM)

#match_soumission['classes'] = match_soumission['classes'].apply(lambda x: str(x))
#match_soumission.info()

#match_soumission.to_csv(r"./predictionProjet1.csv")
"""
