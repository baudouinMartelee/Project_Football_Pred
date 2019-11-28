# IMPORTS
import sqlite3
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from collections import Counter
# Enlever les warnings
import warnings
from sklearn.pipeline import Pipeline
warnings.simplefilter("ignore")


class Fetching(BaseEstimator, TransformerMixin):
    def __init__(self, connect_string):
        self.connect_string = connect_string

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        print("******* Data Fetching *******")
        dat = sqlite3.connect(self.connect_string)

        table = ["Country", "League", "X_Train", "X_Test", "Player", "Player_Attributes",
                 "Team", "Team_Attributes"]

        df_dict = {}

        for name in table:
            query = dat.execute("SELECT * From " + name)
            cols = [column[0] for column in query.description]
            results = pd.DataFrame.from_records(
                data=query.fetchall(), columns=cols)
            df_dict[name.lower()] = results

        return df_dict


class CleaningAndPreparing(BaseEstimator, TransformerMixin):

    def __init__(self, isMatchsTrain=None):
        self.matchs = None
        self.player_attr = None
        self.ply_attr_overall_dict = None
        self.ply_attr_pot_dict = None
        self.teams_name_dict = None
        self.teams_shooting_dict = None
        self.teams_def_dict = None
        # On regarde si on applique la transformation au train set ou au test set
        if(isMatchsTrain is None):
            self.is_test_set = False
        else:
            self.is_test_set = True
        # Nous retenons les formations les plus récurrentes pour les injecter dans le test set
        self.best_formations = {
            'home_form': "",
            'away_form': ""
        }

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        print("******* Data Cleaning and Preparing *******")
        # Initialisation apd du X
        if(self.is_test_set):
            self.matchs = X['x_test']
        else:
            self.matchs = X['x_train']

        self.ply_attr_overall_dict = create_player_overall_dict(
            X['player_attributes'])
        self.ply_attr_pot_dict = create_player_pot_dict(X['player_attributes'])
        self.teams_shooting_dict = create_team_attr_chance_dict(
            X['team_attributes'], 'buildUpPlayPassing')
        self.teams_def_dict = create_team_attr_chance_dict(
            X['team_attributes'], 'defencePressure')

        # Droping irrelevent columns
        self.matchs.drop(['country_id', 'league_id',
                          'match_api_id'], axis=1, inplace=True)

        ######## FEATURES ENGINEERING ##############

        # Create a formation with the Y coordinates
        print("Creating formations...")
        self.matchs['home_form'] = self.matchs.apply(
            lambda x: create_formation(x, True), axis=1)
        self.matchs['away_form'] = self.matchs.apply(
            lambda x: create_formation(x, False), axis=1)

        if(self.is_test_set == False):
            # Nous sauvgardons les formations
            """self.best_formations['home_form'] = self.matchs['home_form'].value_counts(
            ).index[0]
            self.best_formations['away_form'] = self.matchs['away_form'].value_counts(
            ).index[0]"""
            self.matchs[['home_form', 'away_form']] = self.matchs[['home_form', 'away_form']].apply(
                lambda x: x.fillna(x.value_counts().index[0]))
        else:
            self.matchs[['home_form', 'away_form']] = self.matchs[['home_form', 'away_form']].apply(
                lambda x: x.fillna('4231'))

        # Cleaning the date (take only dd-mm-yyy)
        self.matchs['date'] = self.matchs['date'].apply(
            lambda x: x.split(' ')[0])

        print('Putting overall teams ratings...')
        for i in range(1, 12):
            self.matchs['home_player_overall_'+str(i)] = self.matchs.apply(
                lambda x: dict_key_checker(self.ply_attr_overall_dict, x['home_player_'+str(i)], x['date'].split('-')[0])/99, axis=1)
            self.matchs['away_player_overall_'+str(i)] = self.matchs.apply(
                lambda x: dict_key_checker(self.ply_attr_overall_dict, x['away_player_'+str(i)], x['date'].split('-')[0])/99, axis=1)

        print('Putting overall teams potential...')
        for i in range(1, 12):
            self.matchs['home_player_potential_'+str(i)] = self.matchs.apply(
                lambda x: dict_key_checker(self.ply_attr_pot_dict, x['home_player_'+str(i)], x['date'].split('-')[0])/99, axis=1)
            self.matchs['away_player_potential_'+str(i)] = self.matchs.apply(
                lambda x: dict_key_checker(self.ply_attr_pot_dict, x['away_player_'+str(i)], x['date'].split('-')[0])/99, axis=1)

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

        print("Putting team overall by lines ...")
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

        self.matchs.drop(self.matchs.select(
            lambda col: col.startswith('home_player'), axis=1), axis=1, inplace=True)

        self.matchs.drop(self.matchs.select(
            lambda col: col.startswith('away_player'), axis=1), axis=1, inplace=True)

        self.matchs = self.matchs.drop(['home_build_up', 'away_build_up', 'home_def_press', 'away_def_press', 'home_def_overall',
                                        'home_mid_overall', 'home_att_overall', 'away_def_overall', 'away_mid_overall',
                                        'away_att_overall', 'home_def_pot', 'home_mid_pot', 'home_att_pot', 'away_def_pot',
                                        'away_mid_pot', 'away_att_pot', 'date'], axis=1)

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

    if(True in np.isnan(list_form)):
        return np.nan

    # Will create a dict with the occurences of the players's positions
    counter = Counter(list_form)
    couter_val = counter.values()
    # concatenates the values in a string like : 442
    form = ''.join((str(e) for e in list(couter_val)))
    return form


def dict_key_checker(attr_dict, api_id, date):
    if(api_id is np.nan):
        return np.nan
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
    if(form == "Unknowned"):
        return np.nan
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


pipe = Pipeline([
    ('fetch', Fetching('./database/database.sqlite')),
    ('prepare_clean', CleaningAndPreparing())
])

test = pipe.fit_transform(None)
