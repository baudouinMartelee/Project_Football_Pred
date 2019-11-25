"""
Class helper for parameter tuning 
Source: http://www.davidsbatista.net/blog/2018/02/23/model_optimization/
Modified for our needs
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import RandomizedSearchCV


class RandomizedSearchHelper:

    def __init__(self, models, params):
        if not set(models.keys()).issubset(set(params.keys())):
            missing_params = list(set(models.keys()) - set(params.keys()))
            raise ValueError(
                "Some estimators are missing parameters: %s" % missing_params)
        self.models = models
        self.params = params
        self.keys = models.keys()
        self.grid_searches = {}

    def get_gs_best_params(self, key):
        # print(self.grid_searches[key])
        return self.grid_searches[key].best_params_

    def get_gs(self, key):
        return self.grid_searches[key]

    def fit(self, X, y, cv=5, n_jobs=-1, verbose=1, scoring=None, refit=False, n_iter=30):
        for key in self.keys:
            print("Running RandomizedSearchCV for %s." % key)
            model = self.models[key]
            params = self.params[key]
            gs = RandomizedSearchCV(model, params, n_iter=n_iter, cv=cv, n_jobs=n_jobs,
                                    verbose=verbose, scoring=scoring, refit=refit,
                                    return_train_score=True)
            gs.fit(X, y)
            self.grid_searches[key] = gs

    def score_summary(self, sort_by='mean_score'):
        def row(key, scores, params):
            d = {
                'estimator': key,
                'min_score': min(scores),
                'max_score': max(scores),
                'mean_score': np.mean(scores),
                'std_score': np.std(scores),
            }
            return pd.Series({**params, **d})

        rows = []
        for k in self.grid_searches:
            print(k)
            params = self.grid_searches[k].cv_results_['params']
            scores = []
            for i in range(self.grid_searches[k].cv):
                key = "split{}_test_score".format(i)
                r = self.grid_searches[k].cv_results_[key]
                scores.append(r.reshape(len(params), 1))

            all_scores = np.hstack(scores)
            for p, s in zip(params, all_scores):
                rows.append((row(k, s, p)))

        df = pd.concat(rows, axis=1).T.sort_values([sort_by], ascending=False)

        columns = ['estimator', 'min_score',
                   'mean_score', 'max_score', 'std_score']
        columns = columns + [c for c in df.columns if c not in columns]

        return df[columns]
