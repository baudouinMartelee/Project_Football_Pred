
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
from collections import Counter
import warnings
warnings.simplefilter("ignore")


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
        """
        if(matchsTrain is None):
            self.teams_home_win_dict = create_home_team_win(matchs)
            self.teams_away_win_dict = create_away_team_win(matchs)

            self.teams_home_scoring_ratio_dict = create_home_scoring_ratio(
                matchs)
            self.teams_away_scoring_ratio_dict = create_away_scoring_ratio(
                matchs)
        else:
            self.teams_home_win_dict = create_home_team_win(matchsTrain)
            self.teams_away_win_dict = create_away_team_win(matchsTrain)

            self.teams_home_scoring_ratio_dict = create_home_scoring_ratio(
                matchsTrain)
            self.teams_away_scoring_ratio_dict = create_away_scoring_ratio(
                matchsTrain)"""

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
        """self.matchs = self.matchs.apply(
            lambda x: x.fillna(x.value_counts().index[0]))"""

        self.matchs[['home_player_1', 'home_player_2', 'home_player_3', 'home_player_4',
                     'home_player_5', 'home_player_6', 'home_player_7', 'home_player_8', 'home_player_9',
                     'home_player_10', 'home_player_11', 'away_player_1', 'away_player_2', 'away_player_3',
                     'away_player_4', 'away_player_5', 'away_player_6', 'away_player_7', 'away_player_8',
                     'away_player_9', 'away_player_10', 'away_player_11']].fillna(0)

        self.matchs = self.matchs.apply(
            lambda x: x.fillna(x.value_counts().index[0]))

        # Create a formation with the Y coordinates
        print("Creating formations...")
        self.matchs['home_form'] = self.matchs.apply(
            lambda x: create_formation(x, True), axis=1)
        self.matchs['away_form'] = self.matchs.apply(
            lambda x: create_formation(x, False), axis=1)

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

        """self.matchs['home_team_overall'] = self.matchs.select(
            lambda col: col.startswith('home_player_overall_'), axis=1).mean(axis=1)
        self.matchs['away_team_overall'] = self.matchs.select(
            lambda col: col.startswith('away_player_overall_'), axis=1).mean(axis=1)"""

        print('Putting overall teams potential...')
        for i in range(1, 12):
            self.matchs['home_player_potential_'+str(i)] = self.matchs.apply(
                lambda x: test_key(self.ply_attr_pot_dict, int(x['home_player_'+str(i)]), x['date'].split('-')[0])/99, axis=1)
            self.matchs['away_player_potential_'+str(i)] = self.matchs.apply(
                lambda x: test_key(self.ply_attr_pot_dict, int(x['away_player_'+str(i)]), x['date'].split('-')[0])/99, axis=1)

        """self.matchs['home_team_potential'] = self.matchs.select(
            lambda col: col.startswith('home_player_potential_'), axis=1).mean(axis=1)
        self.matchs['away_team_potential'] = self.matchs.select(
            lambda col: col.startswith('away_player_potential_'), axis=1).mean(axis=1)"""

        print("Putting buildUp and defence press...")
        self.matchs['home_build_up'] = self.matchs.apply(lambda x: test_key(
            self.teams_shooting_dict, x['home_team_api_id'], x['date'].split('-')[0])/99, axis=1)
        self.matchs['away_build_up'] = self.matchs.apply(lambda x: test_key(
            self.teams_shooting_dict, x['away_team_api_id'], x['date'].split('-')[0])/99, axis=1)

        self.matchs['diff_build_up'] = self.matchs['home_build_up'] - \
            self.matchs['away_build_up']

        self.matchs['home_def_press'] = self.matchs.apply(lambda x: test_key(
            self.teams_def_dict, x['home_team_api_id'], x['date'].split('-')[0])/99, axis=1)
        self.matchs['away_def_press'] = self.matchs.apply(lambda x: test_key(
            self.teams_def_dict, x['away_team_api_id'], x['date'].split('-')[0])/99, axis=1)

        self.matchs['diff_def_press'] = self.matchs['home_def_press'] - \
            self.matchs['away_def_press']

        """self.matchs['home_form2'] = self.matchs.apply(lambda x: get_nbr_players_by_lines(
            x['home_form']), axis=1)
        self.matchs['away_form2'] = self.matchs.apply(lambda x: get_nbr_players_by_lines(
            x['away_form']), axis=1)"""

        self.matchs.drop(
            ['home_build_up', 'away_build_up', 'home_def_press', 'away_def_press'], axis=1, inplace=True)

        for index, row in self.matchs.iterrows():
            nbr_def_home, nbr_mid_home, nbr_att_home = get_nbr_players_by_lines(
                row['home_form'])
            nbr_def_away, nbr_mid_away, nbr_att_away = get_nbr_players_by_lines(
                row['away_form'])

            # Overall
            self.matchs.loc[index, 'home_def_overall'] = row.loc[[
                'home_player_overall_' + str(i) for i in range(1, nbr_def_home+1)]].mean()
            self.matchs.loc[index, 'home_mid_overall'] = row.loc[[
                'home_player_overall_' + str(i) for i in range(nbr_def_home+1, nbr_def_home + nbr_mid_home+1)]].mean()
            self.matchs.loc[index, 'home_att_overall'] = row.loc[[
                'home_player_overall_' + str(i) for i in range(nbr_def_home + nbr_mid_home+1, 12)]].mean()

            self.matchs.loc[index, 'away_def_overall'] = row.loc[[
                'away_player_overall_' + str(i) for i in range(1, nbr_def_away+1)]].mean()
            self.matchs.loc[index, 'away_mid_overall'] = row.loc[[
                'away_player_overall_' + str(i) for i in range(nbr_def_away+1, nbr_def_away + nbr_mid_away+1)]].mean()
            self.matchs.loc[index, 'away_att_overall'] = row.loc[[
                'away_player_overall_' + str(i) for i in range(nbr_def_away + nbr_mid_away+1, 12)]].mean()

            # Potential
            self.matchs.loc[index, 'home_def_pot'] = row.loc[[
                'home_player_potential_' + str(i) for i in range(1, nbr_def_home+1)]].mean()
            self.matchs.loc[index, 'home_mid_pot'] = row.loc[[
                'home_player_potential_' + str(i) for i in range(nbr_def_home+1, nbr_def_home + nbr_mid_home+1)]].mean()
            self.matchs.loc[index, 'home_att_pot'] = row.loc[[
                'home_player_potential_' + str(i) for i in range(nbr_def_home + nbr_mid_home+1, 12)]].mean()

            self.matchs.loc[index, 'away_def_pot'] = row.loc[[
                'away_player_potential_' + str(i) for i in range(1, nbr_def_away+1)]].mean()
            self.matchs.loc[index, 'away_mid_pot'] = row.loc[[
                'away_player_potential_' + str(i) for i in range(nbr_def_away+1, nbr_def_away + nbr_mid_away+1)]].mean()
            self.matchs.loc[index, 'away_att_pot'] = row.loc[[
                'away_player_potential_' + str(i) for i in range(nbr_def_away + nbr_mid_away+1, 12)]].mean()

        self.matchs['diff_def_overall'] = self.matchs['home_def_overall'] - \
            self.matchs['away_def_overall']
        self.matchs['diff_mid_overall'] = self.matchs['home_mid_overall'] - \
            self.matchs['away_mid_overall']
        self.matchs['diff_att_overall'] = self.matchs['home_att_overall'] - \
            self.matchs['away_att_overall']

        self.matchs['diff_att_home_def_away'] = self.matchs['home_att_overall'] - \
            self.matchs['away_def_overall']
        self.matchs['diff_def_home_att_away'] = self.matchs['home_def_overall'] - \
            self.matchs['away_att_overall']

        self.matchs['diff_def_pot'] = self.matchs['home_def_pot'] - \
            self.matchs['away_def_pot']
        self.matchs['diff_mid_pot'] = self.matchs['home_mid_pot'] - \
            self.matchs['away_mid_pot']
        self.matchs['diff_att_pot'] = self.matchs['home_att_pot'] - \
            self.matchs['away_att_pot']

        """self.matchs['home_win_rate'] = self.matchs.apply(
            lambda x: self.teams_home_win_dict[x['home_team_api_id']], axis=1)
        self.matchs['away_win_rate'] = self.matchs.apply(
            lambda x: self.teams_away_win_dict[x['away_team_api_id']], axis=1)

        self.matchs['home_scoring_ratio'] = self.matchs.apply(
            lambda x: self.teams_home_scoring_ratio_dict[x['home_team_api_id']], axis=1)
        self.matchs['away_scoring_ratio'] = self.matchs.apply(
            lambda x: self.teams_away_scoring_ratio_dict[x['away_team_api_id']], axis=1)"""

        """self.matchs['home_goal_diff'] = self.matchs.apply(
            lambda x: get_goal_diff(self.matchs, x['home_team_api_id']), axis=1)
        self.matchs['away_goal_diff'] = self.matchs.apply(
            lambda x: get_goal_diff(self.matchs, x['away_team_api_id']), axis=1)"""

        """self.matchs['home_matchs_won'] = self.matchs.apply(
            lambda x: get_nbr_matchs_won(self.matchs, x['home_team_api_id']), axis=1)
        self.matchs['away_matchs_won'] = self.matchs.apply(
            lambda x: get_nbr_matchs_won(self.matchs, x['away_team_api_id']), axis=1)
        """
        """self.matchs['matchs_won_against'] = self.matchs.apply(
            lambda x: get_nbr_matchs_won_against(self.matchs, x['home_team_api_id'], x['away_team_api_id']), axis=1)
        self.matchs['matchs_lost_against'] = self.matchs.apply(
            lambda x: get_nbr_matchs_lost_against(self.matchs, x['home_team_api_id'], x['away_team_api_id']), axis=1)
        """
        self.matchs.drop(
            ['home_team_api_id', 'away_team_api_id', 'stage'], axis=1, inplace=True)

        self.matchs.drop(self.matchs.select(
            lambda col: col.startswith('home_player'), axis=1), axis=1, inplace=True)

        self.matchs.drop(self.matchs.select(
            lambda col: col.startswith('away_player'), axis=1), axis=1, inplace=True)

        self.matchs.drop(self.matchs[['home_def_overall','home_mid_overall','home_att_overall','away_def_pot','away_mid_pot','away_att_pot',
        'home_def_pot','home_mid_pot','home_att_pot','away_def_overall','away_mid_overall','away_att_overall']], axis=1, inplace=True)
        
        return self.matchs


# UTILS FUNCTIONS
"""self.matchs['best_team'] = self.matchs.apply(lambda x: get_best_team(
            x['home_team_overall'], x['away_team_overall']), axis=1)

        self.matchs['best_team_pot'] = self.matchs.apply(lambda x: get_best_team(
            x['home_team_potential'], x['away_team_potential']), axis=1)

        self.matchs['best_team_gk'] = self.matchs.apply(lambda x: get_best_team(
            x['home_gk_overall'], x['away_gk_overall']), axis=1)"""


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
    couter_val = Counter(sorted(couter.elements())).values()
    # concatenates the values in a string like : 442
    form = ''.join((str(e) for e in list(couter_val)))
    return form


def get_player_overall(player_api_id, player_attr, date):

    ply_attr = player_attr[player_attr['player_api_id'] == player_api_id]
    current_attr = ply_attr[ply_attr['date'] <
                            date].sort_values(by='date', ascending=False)[: 1]
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


def create_home_team_win(matchs):
    tms = matchs[['home_team_api_id', 'label']]
    tms['label'] = tms.apply(lambda row: 0 if row['label']
                             != 1 else row['label'], axis=1)
    tms = tms.groupby([tms['home_team_api_id']]).agg(
        {'label': 'mean'}).to_dict()['label']
    return tms


def create_away_team_win(matchs):
    tms = matchs[['away_team_api_id', 'label']]
    tms['label'] = tms.apply(lambda row: 0 if row['label']
                             != -1 else 1, axis=1)
    tms = tms.groupby([tms['away_team_api_id']]).agg(
        {'label': 'mean'}).to_dict()['label']
    return tms


def create_home_scoring_ratio(matchs):
    tms = matchs[['home_team_api_id', 'home_team_goal']]
    tms = tms.groupby([tms['home_team_api_id']]).agg(
        {'home_team_goal': 'mean'}).to_dict()['home_team_goal']
    return tms


def create_away_scoring_ratio(matchs):
    tms = matchs[['away_team_api_id', 'away_team_goal']]
    tms = tms.groupby([tms['away_team_api_id']]).agg(
        {'away_team_goal': 'mean'}).to_dict()['away_team_goal']
    return tms


def test_key(attr_dict, api_id, date):
    if(api_id == 0):
        return 0
    try:
        res = attr_dict[(api_id, str(date))]
    except KeyError:
        date = int(date)
        dates = [int(k[1]) for k in attr_dict if k[0] == api_id]
        if not dates:  # api_id not in keys
            return 0
        res = attr_dict[(api_id, str(
            min(dates, key=lambda key: abs(key-date))))]
    # print("Result : "+str(res))
    return res
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


def get_nbr_players_by_lines(form):
    list_form = list(form)
    list_form = [int(x) for x in list_form]
    defenders = list_form[0] + 1  # plus le gardien
    attackers = list_form[-1]
    midfielders = sum(list_form[1:-1])
    return defenders, midfielders, attackers


# matchsTrain = pd.read_csv('X_Train.csv')
matchsTrain = pd.read_csv('X_Train.csv')
matchsTest = pd.read_csv('X_Test.csv')
players = pd.read_csv('Player.csv')
teams = pd.read_csv('Team.csv')
team_attr = pd.read_csv('Team_Attributes.csv')
player_attr = pd.read_csv('Player_Attributes.csv')

matchsTrain['label'] = matchsTrain.apply(lambda row: det_label(
    row['home_team_goal'], row['away_team_goal']), axis=1)

#matchsTrain = matchsTrain.head(1)
df = Data_Engineering(matchsTrain, player_attr, teams, team_attr).run()
correlation = df.corrwith(df['label'])
df = Data_Engineering.add_labels(df)
"""


# df.drop(['Unnamed: 0', 'Unnamed: 0.1'], axis=1, inplace=True)
# df.to_csv(r'./matchsTrainFinal.csv')
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
# matchsTrain = matchsTrain.head(1000)

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

correlation = mergedDf.corrwith(mergedDf['label'])"""

###PLOT
from matplotlib import pyplot as plt
import seaborn as sns

plt.figure(figsize=(40,15))
sns.barplot(x='home_form', y='home_team_goal',data=df)
sns.heatmap(df.corr(),annot=True,cmap='coolwarm', linewidths=.5)
sns.pairplot(df.sample(1000),hue='label')