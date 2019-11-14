from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
from collections import Counter


class Data_Engineering:

    def __init__(self, matchs, player_attr, teams, teams_attr, matchsTrain=None):
        self.matchs = matchs
        self.player_attr = player_attr
        self.ply_attr_overall_dict = create_player_overall_dict(player_attr)
        self.ply_attr_pot_dict = create_player_pot_dict(player_attr)
        self.teams_name_dict = create_team_name_dict(teams)
        self.teams_shooting_dict = create_team_attr_chance_dict(
            teams_attr, 'buildUpPlayPassing')
        self.teams_def_dict = create_team_attr_chance_dict(
            teams_attr, 'defencePressure')
        self.matchsTrain = matchsTrain

    @staticmethod
    def add_labels(matchs):
        # Create labels
        matchs['label'] = matchs.apply(lambda row: det_label(
            row['home_team_goal'], row['away_team_goal']), axis=1)
        return matchs

    def run(self):
        # Droping irrelevent columns
        self.matchs.drop(['country_id', 'league_id',
                          'match_api_id', 'Unnamed: 0'], axis=1, inplace=True)

        print("Putting corresponding teams names...")
        self.matchs['home_team_name'] = self.matchs.apply(
            lambda x: self.teams_name_dict[x['home_team_api_id']], axis=1)
        self.matchs['away_team_name'] = self.matchs.apply(
            lambda x: self.teams_name_dict[x['away_team_api_id']], axis=1)

        ######## FEATURES ENGINEERING ##############

        # Creating formations of the Y coordinates

        # Fill the missing coordinates with the most recurrent ones
        self.matchs = self.matchs.apply(
            lambda x: x.fillna(x.value_counts().index[0]))

        # Create a formation with the Y coordinates
        print("Creating formations...")
        """self.matchs['home_form'] = self.matchs.apply(
            lambda x: create_formation(x, True), axis=1)
        self.matchs['away_form'] = self.matchs.apply(
            lambda x: create_formation(x, False), axis=1)"""

        # print(matchs['home_form'].value_counts())
        # print(matchs['away_form'].value_counts())

        # Cleaning the date (take only dd-mm-yyy)
        self.matchs['date'] = self.matchs['date'].apply(
            lambda x: x.split(' ')[0])

        print('Putting overall teams ratings...')
        for i in range(1, 12):
            self.matchs['home_player_overall_'+str(i)] = self.matchs.apply(
                lambda x: test_key(self.ply_attr_overall_dict, int(x['home_player_'+str(i)]), x['date'].split('-')[0])/99, axis=1)
            self.matchs['away_player_overall_'+str(i)] = self.matchs.apply(
                lambda x: test_key(self.ply_attr_overall_dict, int(x['away_player_'+str(i)]), x['date'].split('-')[0])/99, axis=1)

        # Overall average rating for the team
        self.matchs['home_team_overall'] = self.matchs.select(
            lambda col: col.startswith('home_player_overall_'), axis=1).mean(axis=1)
        self.matchs['away_team_overall'] = self.matchs.select(
            lambda col: col.startswith('away_player_overall_'), axis=1).mean(axis=1)

        print('Putting overall teams potential...')
        for i in range(1, 12):
            self.matchs['home_player_potential_'+str(i)] = self.matchs.apply(
                lambda x: test_key(self.ply_attr_pot_dict, int(x['home_player_'+str(i)]), x['date'].split('-')[0])/99, axis=1)
            self.matchs['away_player_potential_'+str(i)] = self.matchs.apply(
                lambda x: test_key(self.ply_attr_pot_dict, int(x['away_player_'+str(i)]), x['date'].split('-')[0])/99, axis=1)

        self.matchs['home_team_potential'] = self.matchs.select(
            lambda col: col.startswith('home_player_potential_'), axis=1).mean(axis=1)
        self.matchs['away_team_potential'] = self.matchs.select(
            lambda col: col.startswith('away_player_potential_'), axis=1).mean(axis=1)

        """self.matchs['home_gk_overall'] = self.matchs.apply(
            lambda x: test_key(self.ply_attr_overall_dict, int(x['home_player_1']), x['date'].split('-')[0])/99, axis=1)
        self.matchs['away_gk_overall'] = self.matchs.apply(
            lambda x: test_key(self.ply_attr_overall_dict, int(x['away_player_1']), x['date'].split('-')[0])/99, axis=1)"""

        self.matchs.drop(self.matchs.select(
            lambda col: col.startswith('home_player'), axis=1), axis=1, inplace=True)

        self.matchs.drop(self.matchs.select(
            lambda col: col.startswith('away_player'), axis=1), axis=1, inplace=True)

        print("Putting buildUp and defence press...")
        self.matchs['home_build_up'] = self.matchs.apply(lambda x: test_key(
            self.teams_def_dict, x['home_team_api_id'], x['date'].split('-')[0])/99, axis=1)
        self.matchs['away_build_up'] = self.matchs.apply(lambda x: test_key(
            self.teams_def_dict, x['away_team_api_id'], x['date'].split('-')[0])/99, axis=1)

        self.matchs['home_def_press'] = self.matchs.apply(lambda x: test_key(
            self.teams_def_dict, x['home_team_api_id'], x['date'].split('-')[0])/99, axis=1)
        self.matchs['away_def_press'] = self.matchs.apply(lambda x: test_key(
            self.teams_def_dict, x['away_team_api_id'], x['date'].split('-')[0])/99, axis=1)

        """self.matchs['home_win_rate'] = self.matchs.apply(
            lambda x: self.teams_home_win_dict[x['home_team_api_id']], axis=1)
        self.matchs['away_win_rate'] = self.matchs.apply(
            lambda x: self.teams_away_win_dict[x['away_team_api_id']], axis=1)

        self.matchs['home_scoring_ratio'] = self.matchs.apply(
            lambda x: self.teams_home_scoring_ratio_dict[x['home_team_api_id']], axis=1)
        self.matchs['away_scoring_ratio'] = self.matchs.apply(
            lambda x: self.teams_away_scoring_ratio_dict[x['away_team_api_id']], axis=1)"""
        if(self.matchsTrain is None):
            matchs = self.matchs
        else:
            matchs = self.matchsTrain

        self.matchs['home_goal_diff'] = self.matchs.apply(
            lambda x: get_goal_diff(matchs, x['home_team_api_id']), axis=1)
        self.matchs['away_goal_diff'] = self.matchs.apply(
            lambda x: get_goal_diff(matchs, x['away_team_api_id']), axis=1)

        self.matchs['home_matchs_won'] = self.matchs.apply(
            lambda x: get_nbr_matchs_won(matchs, x['home_team_api_id']), axis=1)
        self.matchs['away_matchs_won'] = self.matchs.apply(
            lambda x: get_nbr_matchs_won(matchs, x['away_team_api_id']), axis=1)

        self.matchs['matchs_won_against'] = self.matchs.apply(
            lambda x: get_nbr_matchs_won_against(matchs, x['home_team_api_id'], x['away_team_api_id']), axis=1)
        self.matchs['matchs_lost_against'] = self.matchs.apply(
            lambda x: get_nbr_matchs_lost_against(matchs, x['home_team_api_id'], x['away_team_api_id']), axis=1)

        """ self.matchs.drop(
            ['home_team_api_id', 'away_team_api_id'], axis=1, inplace=True)"""

        return self.matchs


# UTILS FUNCTIONS

def get_best_team(home_team, away_team):
    if(home_team > away_team):
        return 1
    if(home_team == away_team):
        return 0
    else:
        return -1


# Determine the label of the match (0: tie , 1: home team won, -1: home team lost)
def det_label(score1, score2):
    if(score1 == score2):
        return 0
    if(score1 < score2):
        return -1
    else:
        return 1


def get_player_overall(player_api_id, player_attr, date):

    ply_attr = player_attr[player_attr['player_api_id'] == player_api_id]
    current_attr = ply_attr[ply_attr['date'] <
                            date].sort_values(by='date', ascending=False)[:1]
    # print(current_attr['overall_rating'].iloc[0])
    return current_attr['overall_rating'].iloc[0]


def create_player_overall_dict(player_attr):
    ply_attr = player_attr[[
        'player_api_id', 'overall_rating', 'date']]
    ply_attr['date'] = ply_attr['date'].apply(lambda x: x.split('-')[0])

    ply_attr = ply_attr.groupby(
        [ply_attr['player_api_id'], ply_attr['date']]).mean()

    # Replace id of players with their overall rating at the date of the match

    return ply_attr.to_dict()['overall_rating']


def create_player_pot_dict(player_attr):
    ply_attr = player_attr[[
        'player_api_id', 'potential', 'date']]
    ply_attr['date'] = ply_attr['date'].apply(lambda x: x.split('-')[0])

    ply_attr = ply_attr.groupby(
        [ply_attr['player_api_id'], ply_attr['date']]).mean()

    # Replace id of players with their overall rating at the date of the match

    return ply_attr.to_dict()['potential']


def create_team_attr_chance_dict(teams_attr, key):
    tms_attr = teams_attr[['team_api_id', 'date',
                           'defencePressure', 'buildUpPlayPassing']]
    tms_attr['date'] = tms_attr['date'].apply(lambda x: x.split('-')[0])
    tms_attr = tms_attr.groupby(
        [tms_attr['team_api_id'], tms_attr['date']]).mean()
    return tms_attr.to_dict()[key]


def create_team_name_dict(teams):
    tms = teams[['team_api_id', 'team_short_name']]
    return tms.set_index('team_api_id').to_dict()['team_short_name']


def test_key(attr_dict, api_id, date):
    api_id = int(api_id)
    date = int(date)
    while date > 2000:
        if((api_id, str(date)) in attr_dict):
            return attr_dict[(api_id, str(date))]
        else:
            date -= 1
    return np.nan


# Hypothese regarder les dernieres confrontations entre les equipes

def get_matches_against_eachother(matchs, home_team_api_id, away_team_api_id):
    mae = ((matchs['home_team_api_id'] == home_team_api_id) & (matchs['away_team_api_id'] == away_team_api_id)) | (
        (matchs['home_team_api_id'] == home_team_api_id) & (matchs['away_team_api_id'] == home_team_api_id))
    return matchs[mae]


def get_nbr_matchs_won_against(matchs, home_team_api_id, away_team_api_id):
    matchs = get_matches_against_eachother(
        matchs, home_team_api_id, away_team_api_id)
    return get_nbr_matchs_won(matchs, home_team_api_id)


def get_nbr_matchs_lost_against(matchs, home_team_api_id, away_team_api_id):
    matchs = get_matches_against_eachother(
        matchs, home_team_api_id, away_team_api_id)
    return get_nbr_matchs_won(matchs, away_team_api_id)


def get_goal_diff(matchs, team_api_id):
    # Goals scored
    home_goals_scored = matchs['home_team_goal'][matchs['home_team_api_id']
                                                 == team_api_id].sum()
    away_goals_scored = matchs['away_team_goal'][matchs['away_team_api_id']
                                                 == team_api_id].sum()
    goal_diff = home_goals_scored + away_goals_scored
    # Goals conceided
    home_goals_conceided = matchs['away_team_goal'][matchs['home_team_api_id'] == team_api_id].sum(
    )
    away_goals_conceided = matchs['home_team_goal'][matchs['away_team_api_id'] == team_api_id].sum(
    )
    goal_diff = goal_diff - home_goals_conceided - away_goals_conceided
    return goal_diff


def get_nbr_matchs_won(matchs, team_api_id):
    home_matchs_won = matchs[(matchs['home_team_api_id'] == team_api_id) & (
        matchs['label'] == 1)].shape[0]
    away_matchs_won = matchs[(matchs['away_team_api_id'] == team_api_id) & (
        matchs['label'] == -1)].shape[0]
    return home_matchs_won + away_matchs_won


# matchsTrain = pd.read_csv('X_Train.csv')
"""matchsTrain = pd.read_csv('X_Train.csv')
matchsTest = pd.read_csv('X_Test.csv')
players = pd.read_csv('Player.csv')
teams = pd.read_csv('Team.csv')
team_attr = pd.read_csv('Team_Attributes.csv')
player_attr = pd.read_csv('Player_Attributes.csv')

matchsTrain['label'] = matchsTrain.apply(lambda row: det_label(
    row['home_team_goal'], row['away_team_goal']), axis=1)

matchsTrain = matchsTrain.head(10)
# matchsTrain = matchsTrain.head(10)
df = Data_Engineering(matchsTrain, player_attr, teams, team_attr).run()

correlation = mergedDf.corrwith(mergedDf['label'])
"""
# df.drop(['Unnamed: 0', 'Unnamed: 0.1'], axis=1, inplace=True)
# df.to_csv(r'./matchsTrainFinal.csv')
