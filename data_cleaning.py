from sklearn.base import TransformerMixin  # gives fit_transform method for free
from sklearn.preprocessing import LabelBinarizer, OneHotEncoder, MultiLabelBinarizer
from sklearn.preprocessing import Imputer, OrdinalEncoder
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline


class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[self.attribute_names].values


class MyLabelBinarizer(TransformerMixin):
    def __init__(self, *args, **kwargs):
        self.encoder = MultiLabelBinarizer(*args, **kwargs)

    def fit(self, x, y=0):
        self.encoder.fit(x)
        return self

    def transform(self, x, y=0):
        return self.encoder.transform(x)


class Data_Cleaning:
    def __init__(self, matchs):
        self.matchs = matchs

    def run(self):

        # Récupérer toutes les features du type float64
        numerical_data = self.matchs.select_dtypes("float64")
        num_attribs = list(numerical_data)
        # Changer le type des saisons d'objets à categorique
        
        categorical_attrib = ['season']#,'home_team_name','away_team_name','home_form','away_form']
        self.matchs[categorical_attrib] = self.matchs[categorical_attrib].apply(
            lambda x: x.astype('category'))
       
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
        match_cleaned = full_pipeline.fit_transform(self.matchs)
        return match_cleaned
