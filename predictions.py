from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier
from data_engineering import Data_Engineering
from data_cleaning import Data_Cleaning


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

matchsTrain = Data_Engineering(matchsTrain, player_attr, teams).run()
matchsTrain = Data_Engineering.add_labels(matchsTrain)
matchsTest = Data_Engineering(matchsTest, player_attr, teams).run()


label = matchsTrain[['label']]
matchsTrain.drop(columns=['label', 'home_team_goal',
                          'away_team_goal'], inplace=True)


###################################
#        CLEANING DATA            #
###################################

matchsTrain = Data_Cleaning(matchsTrain).run()
matchsTest = Data_Cleaning(matchsTest).run()


###################################
#           PREDICTIONS           #
###################################


X_train, X_test, y_train, y_test = train_test_split(
    matchsTrain, label, random_state=5)

#####  SGDSVM MODEL  #####

rf = SGDClassifier(loss='hinge')

rf.fit(X_train, y_train)

predicted_values_SVM = rf.predict(X_test)

print("score : ", rf.score(X_test, y_test))

"""

match_soumission = pd.DataFrame(predicted_values_SVM)

#match_soumission['classes'] = match_soumission['classes'].apply(lambda x: str(x))
match_soumission.info()

match_soumission.to_csv(
    r"./predictionProjet1.csv")
"""
