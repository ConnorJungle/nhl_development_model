from sklearn.metrics import mean_absolute_error, mean_squared_error, log_loss, accuracy_score
from sklearn.model_selection import GridSearchCV
from xgboost import XGBRegressor, XGBClassifier

import pandas as pd
import numpy as np
import warnings
import joblib

from full_data_load_ep import *
from data_processing import *

from datetime import datetime

import argparse

date = datetime.now().strftime('%d-%m-%Y')

def get_player_dataset(target):

    df = pd.read_csv('../data/player_season_stats_v2.csv')

    X, y,_ = prepare_features(df, target)

    print(X.shape)

    players = X.index.droplevel(-1).unique() # get number indices
    n_players = players.shape[0] # get number of players
    train_idx, test_idx = train_test_split(players, test_size=0.3, random_state=18)

    print('--- Padding Player Data ---')

    X = pad_data(X.reset_index(), players)
    y = pad_data(y.reset_index(), players)

    X.set_index(['playerid', 'player', 'season_age'], inplace=True)
    y.set_index(['playerid', 'player', 'season_age'], inplace=True)

    print('--- Generating Player Data ---')
    train_seq, train_target = generate_players(X, y, train_idx)
    test_seq, test_target = generate_players(X, y, test_idx)

    train_idx_bool = pd.Series(list(X.index.droplevel(-1))).isin(train_idx).values
    test_idx_bool = pd.Series(list(X.index.droplevel(-1))).isin(test_idx).values

    train_real_values = (X[train_idx_bool] != -1).all(axis=1)
    test_real_values = (X[test_idx_bool] != -1).all(axis=1)

    return X, y, train_idx, test_idx, train_real_values, test_real_values, train_idx_bool, test_idx_bool

def get_loss(target, baseline, y, train_idx_bool, train_real_values, X, test_idx_bool, test_real_values ):

    if target == 'ppg_y_plus_1' :

        print('[TRAIN SET] Baseline xgboost Mean Absolute Error: {:2f}'.format(mean_absolute_error(
        y[train_idx_bool][train_real_values],
        m.predict(X[train_idx_bool][train_real_values]))
                                                            )
        )

        print('[TEST SET] Baseline xgboost Mean Absolute Error: {:2f}'.format(mean_absolute_error(
        y[test_idx_bool][test_real_values],
        m.predict(X[test_idx_bool][test_real_values]))
                                                            )
        )

    else:

        print('[TRAIN SET] Baseline xgboost Log Loss: {:2f}'.format(log_loss(
        y[train_idx_bool][train_real_values].idxmax(axis=1),
        baseline.predict_proba(X[train_idx_bool][train_real_values]),
        labels=baseline.classes_
    )
                                                            )
        )

        print('[TRAIN SET] Baseline xgboost Accuracy: {:2f}'.format(accuracy_score(
            y[train_idx_bool][train_real_values].idxmax(axis=1),
            baseline.predict(X[train_idx_bool][train_real_values]))
                                                                )
            )

        print('[TEST SET] Baseline xgboost Log Loss: {:2f}'.format(log_loss(
            y[test_idx_bool][test_real_values].idxmax(axis=1),
            baseline.predict_proba(X[test_idx_bool][test_real_values]),
            labels=baseline.classes_)
                                                                )
            )

        print('[TEST SET] Baseline xgboost Accuracy: {:2f}'.format(accuracy_score(
            y[test_idx_bool][test_real_values].idxmax(axis=1),
            baseline.predict(X[test_idx_bool][test_real_values]))
                                                                )
        )


def train_model(target, params=None):

    X, y, train_idx, test_idx, train_real_values, test_real_values, train_idx_bool, test_idx_bool = get_player_dataset(target)

    loss_metric = None if target == 'ppg_y_plus_1' else 'neg_log_loss'
    label = 'scoring' if target == 'ppg_y_plus_1' else 'league'

    gridsearch = GridSearchCV(estimator=XGBRegressor() if target == 'ppg_y_plus_1' else XGBClassifier(),
                          param_grid=params,
                          n_jobs=6,
                          cv=5,
                          verbose=25,
                          scoring= loss_metric,
                          return_train_score=True,
                          )


    gridsearch.fit(X=X[train_idx_bool][train_real_values],
                y=y[train_idx_bool][train_real_values].idxmax(axis=1))

    print(gridsearch.best_params_)

    m = gridsearch.best_estimator_.fit(X[train_idx_bool][train_real_values], 
                                    y=y[train_idx_bool][train_real_values].idxmax(axis=1))


    get_loss(target, m, y, train_idx_bool, train_real_values, X, test_idx_bool, test_real_values )


    # joblib.dump(m, f'../models/predict_next_{label}_{date}.pkl')
    m.save_model(f'../models/predict_next_{label}_{date}.xgb')

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--target',
                        default='ppg_y_plus_1', 
                        choices=['ppg_y_plus_1', 'league_y_plus_1'],
                        help='Choose a target variable to train model')

    args = parser.parse_args()

    target = args.target

    params = [{'max_depth': [2,3,4],
           'learning_rate': [1e-1, 0.2],
           'n_estimators': [150, 200]}]
    
    train_model(target, params)

