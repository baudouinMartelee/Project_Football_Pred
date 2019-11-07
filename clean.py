import pandas as pd
from collections import Counter

matchs = pd.read_csv('X_Train.csv')
players = pd.read_csv('Player.csv')
XTest = pd.read_csv('X_Test.csv')
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

# print(matchs['home_form'].value_counts())
# print(matchs['away_form'].value_counts())

# Cleaning the date (take only dd-mm-yyy)
matchs['date'] = matchs['date'].apply(lambda x: x.split(' ')[0])

ply_attr = player_attr[['player_api_id', 'overall_rating', 'date']]
ply_attr['date'] = ply_attr['date'].apply(lambda x: x.split('-')[0])
#ply_attr['date_year'] = ply_attr['date'].apply(lambda x: x.split('-')[0])

ply_attr = ply_attr.groupby(
    [ply_attr['player_api_id'], ply_attr['date']]).mean()

# Replace id of players with their overall rating at the date of the match

ply_attr_dict = ply_attr.to_dict()['overall_rating']


def test_key(api_id, date):
    api_id = int(api_id)
    while True:
        if((api_id, date) in ply_attr_dict):
            return ply_attr_dict[(api_id, date)]
        else:
            date = int(date)
            date -= 1
            date = str(date)


for i in range(1, 12):
    matchs['home_player_overall_'+str(i)] = matchs.apply(
        lambda x: test_key(int(x['home_player_'+str(i)]), x['date'].split('-')[0]), axis=1)
    matchs['away_player_overall_'+str(i)] = matchs.apply(
        lambda x: test_key(int(x['away_player_'+str(i)]), x['date'].split('-')[0]), axis=1)


matchs['home_team_overall'] = matchs.select(
    lambda col: col.startswith('home_player_overall_'), axis=1).mean(axis=1)
matchs['away_team_overall'] = matchs.select(
    lambda col: col.startswith('away_player_overall_'), axis=1).mean(axis=1)

matchs.drop(matchs.select(
    lambda col: col.startswith('home_player'), axis=1), axis=1, inplace=True)

matchs.drop(matchs.select(
    lambda col: col.startswith('away_player'), axis=1), axis=1, inplace=True)


def get_best_team(home_team, away_team):
    if(home_team > away_team):
        return 1
    if(home_team == away_team):
        return 0
    else:
        return -1


matchs['best_team'] = matchs.apply(lambda x: get_best_team(
    x['home_team_overall'], x['away_team_overall']), axis=1)

