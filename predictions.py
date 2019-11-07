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
# teams = pd.read_csv('Team.csv')
# team_attr = pd.read_csv('Team_Attributes.csv')
player_attr = pd.read_csv('Player_Attributes.csv')


###################################
#       DATA ENGINEERING          #
###################################

matchsTrain = Data_Engineering(matchsTrain, player_attr).run_engineering_data()
matchsTrain = Data_Engineering.add_labels(matchsTrain)
matchsTest = Data_Engineering(matchsTest, player_attr).run_engineering_data()


label = matchsTrain[['label']]
matchsTrain.drop(columns=['label', 'home_team_goal',
                          'away_team_goal'], inplace=True)


###################################
#        CLEANING DATA            #
###################################

matchsTrain = Data_Cleaning(matchsTrain).run_cleaning_data()
matchsTest = Data_Cleaning(matchsTest).run_cleaning_data()


###################################
#           PREDICTIONS           #
###################################

#####  SGDSVM MODEL  #####

rf = SGDClassifier(loss='hinge')

rf.fit(matchsTrain, label)

predicted_values_SVM = rf.predict(matchsTest)

match_soumission = pd.DataFrame(predicted_values_SVM)

#match_soumission['classes'] = match_soumission['classes'].apply(lambda x: str(x))
match_soumission.info()

match_soumission.to_csv(
    r"./predictionProjet1.csv")
