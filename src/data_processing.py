from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import warnings
import torch
import re

from datetime import datetime, date
from dateutil.relativedelta import relativedelta

warnings.filterwarnings('ignore')

pd.set_option('max_column', 0)

def load_data():

    df = pd.read_csv('player_season_stats.csv')

    X, y,_ = prepare_features(df, 'ppg_y_plus_1')

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

    return X, y, train_seq, train_target, test_seq, test_target, train_idx_bool, test_idx_bool, train_real_values, test_real_values

def is_forward(val):

    match = re.search('[c,w,lw,rw,f]', val.lower())

    if match:
        return 1
    else:
        return 0

def get_primary_position(val):

    match = re.search('[c,w,lw,rw,f]', val.lower())

    if match:
        return 'F'
    else:
        return 'D'

def is_drafted(df):

    # if a player goes into that season drafted
    df['is_drafted'] = np.where(df.start_year >= df.draft_year,
                                1,
                                0)

    return df

def extract_rankings_from_text(text):
    try:
        text_string=re.findall("[0-9]+",text)
        rankings=list(map(int,text_string))
        avg_rankings=sum(rankings)/len(rankings)
        return int(avg_rankings)
    except TypeError:
        return np.NaN

def draft_current_year(df, start_year=None):
    
    df = df.set_index(['playerid', 'player', 'year'])
    
    # get rankings information for players
    try:
        df['assigned_draft_rank'] = df.apply(lambda x: extract_rankings_from_text(x.rankings),axis=1)
        
        df['assigned_draft_round'] = pd.cut(df.assigned_draft_rank, 
                                        bins=[p for p in range(1, 219, 31)],
                                        labels=[r for r in range(1, 8, 1)],
                                        right=False)

        df['draft_year'] = df['draft_year_eligible']
        
        df['draft_round'] = np.where(df.draft_round.isnull(),
                                    df.assigned_draft_round,
                                    df.draft_round)  
        
        df['draft_pick'] = np.where(df.draft_pick.isnull(),
                                df.assigned_draft_rank,
                                df.draft_pick)  

        df = df.drop(columns=['assigned_draft_rank', 'assigned_draft_round'])

    except:
        pass

    if start_year:
        df = df[df.season_age <= start_year]
    
    return df.reset_index()
    
def get_season_age(df):
    '''Return the first year a player is NHL draft eligible'''

    df.set_index('playerid', inplace=True)

    # years = df.year.apply(lambda x: int(x.split('-')[1]))

    # create a series filled with valueus of Birth year + 18 on September 15th
    end_of_year = pd.to_datetime(dict(year=df.end_year,
                                      month=np.full((len(df), ), 12),
                                      day=np.full((len(df), ), 31)))

    end_of_year.index = df.index

    # check if player will be 18 years old by September 15th of draft year
    df['season_age'] = (((end_of_year - df['date_of_birth']) /
                         np.timedelta64(1, 'Y')) // 1).astype(int)

    df['real_season_age'] = (
        (end_of_year - df['date_of_birth']) / np.timedelta64(1, 'Y')).round(2)

    return df.reset_index()

def collapse_player_league_seasons(df):

    df.sort_values('gp', ascending=False, inplace=True)

    return df.drop_duplicates(subset=['playerid', 'player', 'season_age'], keep='first').sort_values('season_age')

def collapse_player_team_seasons(df):

    df_ = df.groupby(['playerid', 'player', 'position',
                      'league', 'year']).apply(lambda x: pd.Series({
                          'team': '/'.join(x.team),
                          'teamids': np.array(x.teamid),
                          'season_age': x.season_age.max(),
                          'real_season_age': x.real_season_age.max(),
                          'start_year': x.start_year.max(),
                          'end_year': x.end_year.max(),
                          'draft_year': x.draft_year.max(),
                          'draft_year_eligible': x.draft_year_eligible.max(),
                          'gp': x.gp.sum(),
                          'g': x.g.sum(),
                          'a': x.a.sum(),
                          'tp': x.tp.sum(),
                          'gpg': x.g.sum() / x.gp.sum(),
                          'apg': x.a.sum() / x.gp.sum(),
                          'ppg': x.tp.sum() / x.gp.sum(),
                          'perc_team_g': (x.perc_team_g * x.gp).sum() / x.gp.sum(),
                          'perc_team_a': (x.perc_team_a * x.gp).sum() / x.gp.sum(),
                          'perc_team_tp': (x.perc_team_tp * x.gp).sum() / x.gp.sum(),
                      }
                      )).reset_index()

    return df_

def get_next_season_data(df):

    df['next_season'] = df.end_year.astype(str) + '-' + \
        (df.end_year.astype(int) + 1).astype(str)

    df = df.merge(
        df[['playerid', 'player', 'year', 'league', 'g', 'a', 'tp',
            'gpg', 'apg', 'ppg', 'gp']],
        left_on=['playerid', 'player', 'next_season'],
        right_on=['playerid', 'player', 'year'],
        suffixes=('', '_y_plus_1'),
        how='left'
    )

    return df

def draft_position(df):

    df['draft_round'] = np.where(df.is_drafted == 1,
                                 np.where(df.draft_round.notnull(),
                                          df.draft_round,
                                          0),
                                 np.where(df.end_year >= df.draft_year,
                                        np.where(df.draft_round.notnull(),
                                                df.draft_round,
                                                0),
                                        0)
                                 )

    df['draft_pick'] = np.where(df.is_drafted == 1,
                                np.where(df.draft_pick.notnull(),
                                         df.draft_pick,
                                         0),
                                 np.where(df.end_year >= df.draft_year,
                                        np.where(df.draft_pick.notnull(),
                                                df.draft_pick,
                                                0),
                                        0)
                                 )

    return df

def prepare_features(df, target, scaler=None):

    features = ['forward', 'gp', 'gp_y_plus_1',
                'draft_pick', 'is_drafted',
                'height', 'weight', 'real_season_age',
                'gpg', 'apg', 'ppg', 'perc_team_g',
                'perc_team_a', 'perc_team_tp']
            
    df = get_next_season_data(df)

    df = df[(df.gp < 85)
        & (df.season_age >= 17)
        & (df.season_age <= 24)
        & (df.gp_y_plus_1.notnull())]

    df = is_drafted(df)
    df = draft_position(df)
    df['forward'] = df.position.apply(is_forward)
    df['real_season_age'] = df['real_season_age'] - \
        df['season_age']  # scale age based on birthday

    season_age = pd.get_dummies(df.season_age, drop_first=True)

    df.set_index(['playerid', 'player', 'season_age'], inplace=True)

    season_age.index = df.index
    start_league = pd.get_dummies(df.league, drop_first=True)
    draft_round = pd.get_dummies(
        df.draft_round,  drop_first=True, prefix='round')

    X = df[[f for f in features if (target ==
            'league_y_plus_1' and f != 'gp_y_plus_1') | (target == 'ppg_y_plus_1')]]

    y = df[target]
    # feature scaling
    scaler = MinMaxScaler()
    scaler.fit(X)
    X[X.columns] = scaler.transform(X)
        
    X = X.merge(start_league, left_index=True, right_index=True)\
        .merge(season_age, left_index=True, right_index=True)\
        .merge(draft_round, left_index=True, right_index=True)

    if target == 'ppg_y_plus_1':  # if we're predicting performance y+1 then we need league y+1

        next_season_league = pd.get_dummies(
            df['league_y_plus_1'], drop_first=True, prefix='next_yr')

        X = X.merge(next_season_league, left_index=True, right_index=True)
    else:
        y = pd.get_dummies(y, prefix='next_yr')

    return X, y, scaler

def prepare_features_single_season(df, scaler, target):

    features = ['forward', 'gp', 'gp_y_plus_1',
                'draft_pick', 'is_drafted',
                'height', 'weight', 'real_season_age',
                'gpg', 'apg', 'ppg', 'perc_team_g',
                'perc_team_a', 'perc_team_tp']

    df = df[(df.season_age >= 17)
            & (df.season_age <= 24)]

    df = is_drafted(df)
    df = draft_position(df)
    df['forward'] = df.position.apply(is_forward)
    df['real_season_age'] = df['real_season_age'] - \
        df['season_age']  # scale age based on birthday

    season_age = pd.get_dummies(df.season_age)

    df.set_index(['playerid', 'player', 'season_age'], inplace=True)

    season_age.index = df.index
    start_league = pd.get_dummies(df.league)
    draft_round = pd.get_dummies(
        df.draft_round, prefix='round')

    X = df[[f for f in features if (target == 
            'league_y_plus_1' and f != 'gp_y_plus_1') | (target == 'ppg_y_plus_1' )]]

    # feature scaling
    X[X.columns] = scaler.transform(X)

    if target == 'ppg_y_plus_1':
        end_league = pd.get_dummies(df.league_y_plus_1, prefix='next_yr')    
        X = X.merge(end_league, left_index=True, right_index=True)

    X = X.merge(start_league, left_index=True, right_index=True)\
        .merge(season_age, left_index=True, right_index=True)\
        .merge(draft_round, left_index=True, right_index=True)

    return X

def column_scaler(df, feature):

    df = get_next_season_data(df)

    df = df[(df.gp < 85)
        & (df.season_age >= 17)
        & (df.season_age <= 24)
        & (df.gp_y_plus_1.notnull())]

    X = df[[feature]]

    scaler = MinMaxScaler()
    scaler.fit(X)

    return scaler

def generate_players(X, y, indices):

    labels = []
    sequences = []

    for idx in indices:
        labels.append(y.loc[idx].values)
        sequences.append(X.loc[idx].values)

    return sequences, labels


def prepare_sequence(seq):

    return seq.index, torch.FloatTensor(seq.values)


def pad_data(df, players):

    pad = pd.concat([pd.DataFrame([[i] for i in range(17, 25)],
                                  columns=['age'],
                                  index=pd.MultiIndex.from_tuples([player for i in range(17, 25)],
                                                                  names=['playerid', 'player']))
                     for player in players]).reset_index()

    padded = pad.merge(df,
                       left_on=['playerid', 'player', 'age'],
                       right_on=['playerid', 'player', 'season_age'],
                       how='left').fillna(-1)

    return padded.drop(columns=['age'])

def pad_sequence(df):

    pad = pd.DataFrame([[i] for i in range(17, 25)],
                       columns=['age'])

    padded = pad.merge(df,
                       left_on=['age'],
                       right_index=True,
                       how='left').fillna(-1)

    return padded.drop(columns=['age'])

def check_late_birthday(date_of_birth):

    try:
        dt = datetime.strptime(date_of_birth, '%Y-%m-%d')
        draft = date(dt.year + 18, 9, 15)
        if relativedelta(draft, dt).years < 18 :
            return True
        else:
            return False

    except:
        return None