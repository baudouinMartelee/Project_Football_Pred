
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
from collections import Counter


class Data_Engineering:

    def __init__(self, matchs, player_attr, teams, teams_attr):
        self.matchs = matchs
        self.ply_attr_overall_dict = create_player_overall_dict(player_attr)
        self.ply_attr_pot_dict = create_player_pot_dict(player_attr)
        self.teams_name_dict = create_team_name_dict(teams)
        self.teams_shooting_dict = create_team_attr_chance_dict(
            teams_attr, 'buildUpPlayPassing')
        self.teams_def_dict = create_team_attr_chance_dict(
            teams_attr, 'defencePressure')

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
        """print("Creating formations...")
        self.matchs['home_form'] = self.matchs.apply(
            lambda x: create_formation(x, True), axis=1)
        self.matchs['away_form'] = self.matchs.apply(
            lambda x: create_formation(x, False), axis=1)
        """
        # print(matchs['home_form'].value_counts())
        # print(matchs['away_form'].value_counts())

        # Cleaning the date (take only dd-mm-yyy)
        self.matchs['date'] = self.matchs['date'].apply(
            lambda x: x.split(' ')[0])

        print('Putting overall teams ratings...')
        for i in range(1, 12):
            self.matchs['home_player_overall_'+str(i)] = self.matchs.apply(
                lambda x: test_key(self.ply_attr_overall_dict, int(x['home_player_'+str(i)]), x['date'].split('-')[0]), axis=1)
            self.matchs['away_player_overall_'+str(i)] = self.matchs.apply(
                lambda x: test_key(self.ply_attr_overall_dict, int(x['away_player_'+str(i)]), x['date'].split('-')[0]), axis=1)

        self.matchs['home_team_overall'] = self.matchs.select(
            lambda col: col.startswith('home_player_overall_'), axis=1).mean(axis=1)
        self.matchs['away_team_overall'] = self.matchs.select(
            lambda col: col.startswith('away_player_overall_'), axis=1).mean(axis=1)

        self.matchs['home_team_overall'] = self.matchs['home_team_overall']/99
        self.matchs['away_team_overall'] = self.matchs['away_team_overall']/99

        print('Putting overall teams potential...')
        for i in range(1, 12):
            self.matchs['home_player_potential_'+str(i)] = self.matchs.apply(
                lambda x: test_key(self.ply_attr_pot_dict, int(x['home_player_'+str(i)]), x['date'].split('-')[0]), axis=1)
            self.matchs['away_player_potential_'+str(i)] = self.matchs.apply(
                lambda x: test_key(self.ply_attr_pot_dict, int(x['away_player_'+str(i)]), x['date'].split('-')[0]), axis=1)

        self.matchs['home_team_potential'] = self.matchs.select(
            lambda col: col.startswith('home_player_potential_'), axis=1).mean(axis=1)
        self.matchs['away_team_potential'] = self.matchs.select(
            lambda col: col.startswith('away_player_potential_'), axis=1).mean(axis=1)

        self.matchs['home_team_potential'] = self.matchs['home_team_potential']/99
        self.matchs['away_team_potential'] = self.matchs['away_team_potential']/99

        self.matchs['home_gk_overall'] = self.matchs.apply(
            lambda x: test_key(self.ply_attr_overall_dict, int(x['home_player_1']), x['date'].split('-')[0]), axis=1)
        self.matchs['away_gk_overall'] = self.matchs.apply(
            lambda x: test_key(self.ply_attr_overall_dict, int(x['away_player_1']), x['date'].split('-')[0]), axis=1)

        self.matchs.drop(self.matchs.select(
            lambda col: col.startswith('home_player'), axis=1), axis=1, inplace=True)

        self.matchs.drop(self.matchs.select(
            lambda col: col.startswith('away_player'), axis=1), axis=1, inplace=True)

        self.matchs['best_team'] = self.matchs.apply(lambda x: get_best_team(
            x['home_team_overall'], x['away_team_overall']), axis=1)

        self.matchs['best_team_pot'] = self.matchs.apply(lambda x: get_best_team(
            x['home_team_potential'], x['away_team_potential']), axis=1)

        self.matchs['best_team_gk'] = self.matchs.apply(lambda x: get_best_team(
            x['home_gk_overall'], x['away_gk_overall']), axis=1)

        print("Putting buildUp and defence press...")
        self.matchs['home_build_up'] = self.matchs.apply(lambda x: test_key(
            self.teams_def_dict, x['home_team_api_id'], x['date'].split('-')[0])/99, axis=1)
        self.matchs['away_build_up'] = self.matchs.apply(lambda x: test_key(
            self.teams_def_dict, x['away_team_api_id'], x['date'].split('-')[0])/99, axis=1)

        self.matchs['home_def_press'] = self.matchs.apply(lambda x: test_key(
            self.teams_def_dict, x['home_team_api_id'], x['date'].split('-')[0])/99, axis=1)
        self.matchs['away_def_press'] = self.matchs.apply(lambda x: test_key(
            self.teams_def_dict, x['away_team_api_id'], x['date'].split('-')[0])/99, axis=1)

        self.matchs.drop(
            ['home_team_api_id', 'away_team_api_id'], axis=1, inplace=True)
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


"""
def create_formation(row, home):
    list_form = list()  # We need a list for Counter
    if(home):
        list_form = row.loc[row.index.str.startswith(
            'home_player_Y')].tolist()[1:]  # We don't take the goalkeeper
    else:
        list_form = row.loc[row.index.str.startswith(
            'away_player_Y')].tolist()[1:]
    # Will create a dict with the occurences of the players's positions
    couter = Counter(list_form)
    # concatenates the values in a string like : 442
    form = ''.join((str(e) for e in list(couter.values())))
    return form
"""


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


"""

matchsTrain = pd.read_csv('X_Train.csv')
matchsTest = pd.read_csv('X_Test.csv')
players = pd.read_csv('Player.csv')
teams = pd.read_csv('Team.csv')
team_attr = pd.read_csv('Team_Attributes.csv')
player_attr = pd.read_csv('Player_Attributes.csv')

matchsTrain['label'] = matchsTrain.apply(lambda row: det_label(
    row['home_team_goal'], row['away_team_goal']), axis=1)

df = Data_Engineering(matchsTrain, player_attr, teams, team_attr).run()
correlation = df.corrwith(df['label'])

# Encoding categorical
df = Data_Engineering(matchsTrain, player_attr, teams, team_attr).run()
le = LabelEncoder()
# Categorical boolean mask
categorical_feature_mask = df.dtypes == object
# filter categorical columns using mask and turn it into a list
categorical_cols = df.columns[categorical_feature_mask].tolist()
df[categorical_cols] = df[categorical_cols].apply(
    lambda col: le.fit_transform(col))

correlation = df.corrwith(df['label'])


# Corr with team attrib
mergedDf = matchsTrain.merge(
    team_attr, left_on='home_team_api_id', right_on='team_api_id')
correlation = mergedDf.corrwith(mergedDf['label'])


# Corr with player attri
player_attr_home = player_attr.select_dtypes(include=['float64', 'int64'])
player_attr_away = player_attr.select_dtypes(include=['float64', 'int64'])
player_attr_home = player_attr.add_suffix('_home')
player_attr_away = player_attr.add_suffix('_away')

player_attr_home = player_attr_home.groupby(
    player_attr_home['player_api_id_home']).mean()
player_attr_away = player_attr_away.groupby(
    player_attr_away['player_api_id_away']).mean()

player_attr_home['player_api_id_home'] = player_attr_home.index
player_attr_away['player_api_id_away'] = player_attr_away.index


matchsTrain = matchsTrain.select_dtypes(include=['float64', 'int64'])
#matchsTrain = matchsTrain.head(1000)

mergedDf = matchsTrain.merge(
    player_attr_home, left_on='home_player_1', right_index=True)
mergedDf = matchsTrain.merge(
    player_attr_away, left_on='away_player_1', right_index=True)

for i in range(2, 12):
    player_attr_home = player_attr.add_suffix('_'+str(i))
    player_attr_away = player_attr.add_suffix('_'+str(i))
    mergedDf = mergedDf.merge(
        player_attr_home, left_on='home_player_'+str(i), right_index=True)
    mergedDf = mergedDf.merge(
        player_attr_away, left_on='away_player_'+str(i), right_index=True)

correlation = mergedDf.corrwith(mergedDf['label'])
"""