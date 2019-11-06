import pandas as pd
from collections import Counter

matchs = pd.read_csv('X_Train.csv')
players = pd.read_csv('Player.csv')
# countries = pd.read_csv('Country.csv')
# leagues = pd.read_csv('League.csv')
teams = pd.read_csv('Team.csv')
team_attr = pd.read_csv('Team_Attributes.csv')
player_attr = pd.read_csv('Player_Attributes.csv')


# Determine the label of the match (0: tie , 1: home team won, -1: home team lost)
def det_label(score1, score2):
    if(score1 == score2):
        return 0
    if(score1 < score2):
        return -1
    else:
        return 1


# Create labels
matchs['label'] = matchs.apply(lambda row: det_label(
    row.home_team_goal, row.away_team_goal), axis=1)

# Droping irrelevent columns
matchs.drop(['country_id', 'league_id', 'match_api_id',
             'home_team_api_id', 'away_team_api_id', 'Unnamed: 0'], axis=1, inplace=True)

######## FEATURES ENGINEERING ##############

# Creating formations of the Y coordinates

# Fill the missing coordinates with the most recurrent ones
matchs = matchs.apply(
    lambda x: x.fillna(x.value_counts().index[0]))


# Create a formation with the Y coordinates
def create_formation(row, home):
    list_test = list()  # We need a list for Counter
    # print(row)
    for i in range(2, 12):  # use don't care of the keeper
        if(home):
            list_test.append(row['home_player_Y'+str(i)])
        else:
            list_test.append(row['away_player_Y'+str(i)])
    # Will create a dict with the occurences of the players's positions
    couter = Counter(list_test)
    # print((list(couter.values())))
    form = ''.join((str(e) for e in list(couter.values())))
    # print(form)
    return form


matchs['home_form'] = matchs.apply(
    lambda x: create_formation(x, True), axis=1)
matchs['away_form'] = matchs.apply(
    lambda x: create_formation(x, False), axis=1)

matchs.drop(['home_player_Y1', 'home_player_Y2', 'home_player_Y3', 'home_player_Y4', 'home_player_Y5',
             'home_player_Y6', 'home_player_Y7', 'home_player_Y8', 'home_player_Y9', 'home_player_Y10',
             'home_player_Y11', 'away_player_Y1', 'away_player_Y2', 'away_player_Y3', 'away_player_Y4', 'away_player_Y5',
             'away_player_Y6', 'away_player_Y7', 'away_player_Y8', 'away_player_Y9', 'away_player_Y10', 'away_player_Y11'], axis=1, inplace=True)

matchs.drop(['home_player_X1', 'home_player_X2', 'home_player_X3', 'home_player_X4', 'home_player_X5',
             'home_player_X6', 'home_player_X7', 'home_player_X8', 'home_player_X9', 'home_player_X10',
             'home_player_X11', 'away_player_X1', 'away_player_X2', 'away_player_X3', 'away_player_X4', 'away_player_X5',
             'away_player_X6', 'away_player_X7', 'away_player_X8', 'away_player_X9', 'away_player_X10', 'away_player_X11'], axis=1, inplace=True)
