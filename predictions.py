from sklearn.base import TransformerMixin  # gives fit_transform method for free
from sklearn.preprocessing import LabelBinarizer, OneHotEncoder, MultiLabelBinarizer
from sklearn.preprocessing import Imputer, OrdinalEncoder
from sklearn.linear_model import SGDClassifier
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
import pandas as pd
from data_engineering import Data_Engineering

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

de_train = Data_Engineering(matchsTrain, player_attr)
de_test = Data_Engineering(matchsTest, player_attr)

matchsTrain = de_train.run_engineering_data()
matchsTrain = Data_Engineering.add_labels(matchsTrain)
matchsTest = de_test.run_engineering_data()


label = matchsTrain[['label']]
matchsTrain.drop(columns=['label', 'home_team_goal',
                          'away_team_goal'], inplace=True)


###################################
#        CLEANING DATA            #
###################################


class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[self.attribute_names].values


class MyLabelBinarizer(TransformerMixin):
    def __init__(self, *args, **kwargs):
        self.encoder = LabelBinarizer(*args, **kwargs)

    def fit(self, x, y=0):
        self.encoder.fit(x)
        return self

    def transform(self, x, y=0):
        return self.encoder.transform(x)

# label


def clean_data(matchs):

    # Récupérer toutes les features du type float64
    numerical_data = matchs.select_dtypes("float64")
    num_attribs = list(numerical_data)
    # Changer le type des saisons d'objets à categorique
    matchs[['season']] = matchs[['season']].apply(
        lambda x: x.astype('category'))
    categorical_attrib = ['season']
    # Utilisation de pipeline pour clean les data

    num_pipeline = Pipeline([
        ('selector', DataFrameSelector(num_attribs)),
        ('imputer', Imputer(strategy="median")),
        ('min_max_scaler', MinMaxScaler()),
    ])

    categorical_pipeline = Pipeline([
        ('selector', DataFrameSelector(categorical_attrib)),
        ('label_binazer', MyLabelBinarizer()),
    ])

    from sklearn.pipeline import FeatureUnion
    full_pipeline = FeatureUnion(transformer_list=[
        ("num_pipeline", num_pipeline),
        ("categorical_pipeline", categorical_pipeline)
    ])

    # data are clean here
    match_cleaned = full_pipeline.fit_transform(matchs)
    return match_cleaned


"""
matchsTrain = clean_data(matchsTrain)
matchsTest = clean_data(matchsTest)


#####  SGDSVM MODEL  #####

rf = SGDClassifier(loss='hinge')

rf.fit(matchsTrain, label)

predicted_values_SVM = rf.predict(matchsTest)


match_soumission = pd.DataFrame(predicted_values_SVM)

#match_soumission['classes'] = match_soumission['classes'].apply(lambda x: str(x))
match_soumission.info()

match_soumission.to_csv(
    r"./predictionProjet1.csv")
"""
