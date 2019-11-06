import pandas as pd

matchs = pd.read_csv('X_Train.csv')
players = pd.read_csv('Player.csv')
#countries = pd.read_csv('Country.csv')
#leagues = pd.read_csv('League.csv')
teams = pd.read_csv('Team.csv')
team_attr = pd.read_csv('Team_Attributes.csv')
player_attr = pd.read_csv('Player_Attributes.csv')


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

matchs.drop(['country_id', 'league_id', 'match_api_id',
             'home_team_api_id', 'away_team_api_id', 'Unnamed: 0'], axis=1)
