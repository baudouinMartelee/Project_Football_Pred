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

        # Fill the missing coordinates with the most recurrent ones

        self.matchs[['home_player_'+str(i) for i in range(1, 12)]].fillna(0)
        self.matchs[['away_player_'+str(i) for i in range(1, 12)]].fillna(0)

        self.matchs = self.matchs.apply(
            lambda x: x.fillna(x.value_counts().index[0]))

        # Create a formation with the Y coordinates
        print("Creating formations...")
        self.matchs['home_form'] = self.matchs.apply(
            lambda x: create_formation(x, True), axis=1)
        self.matchs['away_form'] = self.matchs.apply(
            lambda x: create_formation(x, False), axis=1)

        # Cleaning the date (take only dd-mm-yyy)
        self.matchs['date'] = self.matchs['date'].apply(
            lambda x: x.split(' ')[0])

        print('Putting overall teams ratings...')
        for i in range(1, 12):
            self.matchs['home_player_overall_'+str(i)] = self.matchs.apply(
                lambda x: dict_key_checker(self.ply_attr_overall_dict, int(x['home_player_'+str(i)]), x['date'].split('-')[0])/99, axis=1)
            self.matchs['away_player_overall_'+str(i)] = self.matchs.apply(
                lambda x: dict_key_checker(self.ply_attr_overall_dict, int(x['away_player_'+str(i)]), x['date'].split('-')[0])/99, axis=1)

        print('Putting overall teams potential...')
        for i in range(1, 12):
            self.matchs['home_player_potential_'+str(i)] = self.matchs.apply(
                lambda x: dict_key_checker(self.ply_attr_pot_dict, int(x['home_player_'+str(i)]), x['date'].split('-')[0])/99, axis=1)
            self.matchs['away_player_potential_'+str(i)] = self.matchs.apply(
                lambda x: dict_key_checker(self.ply_attr_pot_dict, int(x['away_player_'+str(i)]), x['date'].split('-')[0])/99, axis=1)

        print("Putting buildUp and defence press...")
        self.matchs['home_build_up'] = self.matchs.apply(lambda x: dict_key_checker(
            self.teams_shooting_dict, x['home_team_api_id'], x['date'].split('-')[0])/99, axis=1)
        self.matchs['away_build_up'] = self.matchs.apply(lambda x: dict_key_checker(
            self.teams_shooting_dict, x['away_team_api_id'], x['date'].split('-')[0])/99, axis=1)

        self.matchs['diff_build_up'] = self.matchs['home_build_up'] - \
            self.matchs['away_build_up']

        self.matchs['home_def_press'] = self.matchs.apply(lambda x: dict_key_checker(
            self.teams_def_dict, x['home_team_api_id'], x['date'].split('-')[0])/99, axis=1)
        self.matchs['away_def_press'] = self.matchs.apply(lambda x: dict_key_checker(
            self.teams_def_dict, x['away_team_api_id'], x['date'].split('-')[0])/99, axis=1)

        self.matchs['diff_def_press'] = self.matchs['home_def_press'] - \
            self.matchs['away_def_press']

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

        self.matchs.drop(
            ['home_team_api_id', 'away_team_api_id', 'stage'], axis=1, inplace=True)

        self.matchs.drop(self.matchs.select(
            lambda col: col.startswith('home_player'), axis=1), axis=1, inplace=True)

        self.matchs.drop(self.matchs.select(
            lambda col: col.startswith('away_player'), axis=1), axis=1, inplace=True)

        self.matchs.drop(self.matchs[['home_def_overall', 'home_mid_overall', 'home_att_overall', 'away_def_pot', 'away_mid_pot', 'away_att_pot',
                                      'home_def_pot', 'home_mid_pot', 'home_att_pot', 'away_def_overall', 'away_mid_overall', 'away_att_overall']], axis=1, inplace=True)

        return self.matchs


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
    counter = Counter(list_form)
    couter_val = counter.values()
    # concatenates the values in a string like : 442
    form = ''.join((str(e) for e in list(couter_val)))
    return form


def dict_key_checker(attr_dict, api_id, date):
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


def get_nbr_players_by_lines(form):
    list_form = list(form)
    list_form = [int(x) for x in list_form]
    defenders = list_form[0] + 1  # plus le gardien
    attackers = list_form[-1]
    midfielders = sum(list_form[1:-1])
    return defenders, midfielders, attackers


############Dictionnaries creation#################
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


"""
Hypotheses :
    -formations
    -notes des joueurs
    -



"""
