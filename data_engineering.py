import pandas as pd
from collections import Counter

class Data_Engineering:

    def __init__(self, matchs, player_attr, teams, teams_attr):
        self.matchs = matchs
        self.ply_attr_dict = create_player_overall_dict(player_attr)
        self.teams_name_dict = create_team_name_dict(teams)
        self.teams_shooting_dict = create_team_attr_chance_dict(teams_attr,'chanceCreationShooting')
        self.teams_def_dict = create_team_attr_chance_dict(teams_attr,'defenceAggression')
    
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
                lambda x: test_key(self.ply_attr_dict, int(x['home_player_'+str(i)]), x['date'].split('-')[0]), axis=1)
            self.matchs['away_player_overall_'+str(i)] = self.matchs.apply(
                lambda x: test_key(self.ply_attr_dict, int(x['away_player_'+str(i)]), x['date'].split('-')[0]), axis=1)

        self.matchs['home_team_overall'] = self.matchs.select(
            lambda col: col.startswith('home_player_overall_'), axis=1).mean(axis=1)
        self.matchs['away_team_overall'] = self.matchs.select(
            lambda col: col.startswith('away_player_overall_'), axis=1).mean(axis=1)

        self.matchs.drop(self.matchs.select(
            lambda col: col.startswith('home_player'), axis=1), axis=1, inplace=True)

        self.matchs.drop(self.matchs.select(
            lambda col: col.startswith('away_player'), axis=1), axis=1, inplace=True)

        self.matchs['best_team'] = self.matchs.apply(lambda x: get_best_team(
            x['home_team_overall'], x['away_team_overall']), axis=1)

        print("Putting shooting chances and defence agression...")
        self.matchs['home_shooting_chances'] = self.matchs.apply(lambda x: test_key(self.teams_def_dict,x['home_team_api_id'],x['date'].split('-')[0]), axis=1)    
        self.matchs['away_shooting_chances'] = self.matchs.apply(lambda x: test_key(self.teams_def_dict,x['away_team_api_id'],x['date'].split('-')[0]), axis=1)   

        self.matchs['home_def_agr'] = self.matchs.apply(lambda x: test_key(self.teams_def_dict,x['home_team_api_id'],x['date'].split('-')[0]), axis=1)    
        self.matchs['away_def_agr'] = self.matchs.apply(lambda x: test_key(self.teams_def_dict,x['away_team_api_id'],x['date'].split('-')[0]), axis=1)    

        self.matchs.drop(['home_team_api_id','away_team_api_id'], axis=1, inplace=True)    

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


def create_formation(row, home):
    list_form = list()  # We need a list for Counter
    if(home):
        list_form = row.loc[row.index.str.startswith(
            'home_player_Y')].tolist()[1:] #We don't take the goalkeeper
    else:
        list_form = row.loc[row.index.str.startswith(
            'away_player_Y')].tolist()[1:]
    # Will create a dict with the occurences of the players's positions
    couter = Counter(list_form)
    #concatenates the values in a string like : 442
    form = ''.join((str(e) for e in list(couter.values())))
    return form


def create_player_overall_dict(player_attr):
    ply_attr = player_attr[[
        'player_api_id', 'overall_rating', 'date']]
    ply_attr['date'] = ply_attr['date'].apply(lambda x: x.split('-')[0])

    ply_attr = ply_attr.groupby(
        [ply_attr['player_api_id'], ply_attr['date']]).mean()

    # Replace id of players with their overall rating at the date of the match

    return ply_attr.to_dict()['overall_rating']

def create_team_attr_chance_dict(teams_attr, shootingOrDef):
    tms_attr = teams_attr[['team_api_id', 'date','chanceCreationShooting','defenceAggression']]
    tms_attr['date'] = tms_attr['date'].apply(lambda x: x.split('-')[0])
    tms_attr = tms_attr.groupby([tms_attr['team_api_id'],tms_attr['date']]).mean()
    return tms_attr.to_dict()[shootingOrDef]

def create_team_name_dict(teams):
    tms = teams[['team_api_id', 'team_short_name']]
    return tms.set_index('team_api_id').to_dict()['team_short_name']


def test_key(attr_dict, api_id, date):
    api_id = int(api_id)
    date=int(date)
    while date>2000:
        if((api_id, str(date)) in attr_dict):
            return attr_dict[(api_id, str(date))]
        else:
            date -= 1
    return 0


#TEST"""
"""import pandas as pd
teams_attr = pd.read_csv('Team_Attributes.csv')
tms_attr = teams_attr[['team_api_id', 'date','chanceCreationShooting','defenceAggression']]
tms_attr['date'] = tms_attr['date'].apply(lambda x: x.split('-')[0])
dico = tms_attr.set_index(['team_api_id','date']).to_dict()['chanceCreationShooting']
print(dico[9987,'2012'])"""