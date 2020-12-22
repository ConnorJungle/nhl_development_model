from data_processing import *

from multiprocessing import Pool

from scoring_lstm import Model
from xgboost import XGBClassifier, Booster 
from sklearn.linear_model import LinearRegression

from networkx.drawing.nx_agraph import graphviz_layout
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import networkx as nx
import matplotlib

from scipy.stats import mode
import pandas as pd
import numpy as np
import warnings
import joblib
import torch
import ast

from scipy import stats

import plotly.graph_objects as go

import seaborn as sns

f_role_labels = ['Depth', 'Top 9', 'Top 6', 'Top Line', 'Elite']
d_role_labels = ['Depth', 'Top 6', 'Top 4', 'Top Pair', 'Elite']

pd.set_option('max_column', 0)
warnings.filterwarnings('ignore')

def predict_scoring(features, version=0):

    scoring_lstm = Model(input_size=features.shape[1],
                        hidden_size=features.shape[1] * 2,) 
    scoring_lstm.load_state_dict(torch.load(f'../models/v2_boot_lstm_{version}.pth'))
    scoring_lstm.eval()

    return scoring_lstm(features).detach().numpy()

class GeneratePlayer(object):
    def __init__(self):
        self.dataset = pd.read_csv('../data/player_season_stats_v3.csv')
        
        # self.league_model = joblib.load('../models/predict_next_league.pkl')
        self.league_model = XGBClassifier()
        booster = Booster()
        booster.load_model('../models/predict_next_league_10-12-2020.xgb')
        self.league_model._Booster = booster
#         self.scoring_model = joblib.load('../models/scoring_model.pkl')
        # self.scoring_lstm.load_state_dict(torch.load('../models/scoring_lstm_checkpoint_v6_iter1.pth'))
        # self.scoring_lstm.eval()
        
        self.ppg_model = joblib.load('../models/ppg_team_model.pkl')
        self.gpg_model = joblib.load('../models/gpg_team_model.pkl')
        self.apg_model = joblib.load('../models/apg_team_model.pkl')
        # league to game played mapping
        self.impute_games_played()
        # instantiate league index
        self.get_league_index()
        # model feature scalers
        _, _, self.league_scaler = prepare_features(self.dataset, 'league_y_plus_1')
        _, _, self.scoring_scaler = prepare_features(self.dataset, 'ppg_y_plus_1')

        # scoring by position        
        self.scoring_by_position()

        # model features
        self.league_features = [
            'forward',              'gp',      'draft_pick',
            'is_drafted',          'height',          'weight',
       'real_season_age',             'gpg',             'apg',
                   'ppg',     'perc_team_g',     'perc_team_a',
          'perc_team_tp',            'AJHL',     'Allsvenskan',
                  'BCHL',            'CCHL',           'Czech',
                'Czech2',             'DEL',            'ECHL',
        'Jr. A SM-liiga',             'KHL',           'Liiga',
                   'MHL',          'Mestis',            'NCAA',
                   'NHL',             'NLA',             'OHL',
                  'OJHL',           'QMJHL',             'SHL',
                  'SJHL',        'Slovakia',       'SuperElit',
                  'USDP',            'USHL',             'VHL',
                   'WHL',                18,                19,
                      20,                21,                22,
                      23,                24,       'round_1.0',
             'round_2.0',       'round_3.0',       'round_4.0',
             'round_5.0',       'round_6.0',       'round_7.0',
             'round_8.0',       'round_9.0'
             ]

        self.scoring_features = [  
                'forward',                     'gp',
                  'gp_y_plus_1',             'draft_pick',
                   'is_drafted',                 'height',
                       'weight',        'real_season_age',
                          'gpg',                    'apg',
                          'ppg',            'perc_team_g',
                  'perc_team_a',           'perc_team_tp',
                         'AJHL',            'Allsvenskan',
                         'BCHL',                   'CCHL',
                        'Czech',                 'Czech2',
                          'DEL',                   'ECHL',
               'Jr. A SM-liiga',                    'KHL',
                        'Liiga',                    'MHL',
                       'Mestis',                   'NCAA',
                          'NHL',                    'NLA',
                          'OHL',                   'OJHL',
                        'QMJHL',                    'SHL',
                         'SJHL',               'Slovakia',
                    'SuperElit',                   'USDP',
                         'USHL',              'USHS-Prep',
                          'VHL',                    'WHL',
                             18,                       19,
                             20,                       21,
                             22,                       23,
                             24,              'round_1.0',
                    'round_2.0',              'round_3.0',
                    'round_4.0',              'round_5.0',
                    'round_6.0',              'round_7.0',
                    'round_8.0',              'round_9.0',
                 'next_yr_AJHL',    'next_yr_Allsvenskan',
                 'next_yr_BCHL',           'next_yr_CCHL',
                'next_yr_Czech',         'next_yr_Czech2',
                  'next_yr_DEL',           'next_yr_ECHL',
       'next_yr_Jr. A SM-liiga',            'next_yr_KHL',
                'next_yr_Liiga',            'next_yr_MHL',
               'next_yr_Mestis',           'next_yr_NCAA',
                  'next_yr_NHL',            'next_yr_NLA',
                  'next_yr_OHL',           'next_yr_OJHL',
                'next_yr_QMJHL',            'next_yr_SHL',
                 'next_yr_SJHL',       'next_yr_Slovakia',
            'next_yr_SuperElit',           'next_yr_USDP',
                 'next_yr_USHL',      'next_yr_USHS-Prep',
                  'next_yr_VHL',            'next_yr_WHL'
        ]

    def initialize_player(self, playerid, start_year=None):
        self.playerid = playerid
        self.player_df = self.dataset[self.dataset.playerid == int(playerid)]
        # for current draft eligibles, impute their average draft position
        self.player_df = draft_current_year(self.player_df, start_year)
        self.player_name = self.player_df.player.iloc[0]
        self.date_of_birth = self.player_df.date_of_birth.iloc[0]
        self.current_age = self.player_df.season_age.max()
        self.start_age = self.player_df.season_age.max()
        self.start_league = self.player_df[self.player_df.season_age == self.player_df.season_age.max()].league.iloc[0]
        self.a_proportion = (self.player_df.apg / self.player_df.ppg).median()
        self.g_proportion = 1 - self.a_proportion
        self.projections = pd.DataFrame()
        self.position = self.get_position()
    
    def scoring_by_position(self):
    
        self.dataset['primary_position'] = self.dataset.position.apply(get_primary_position)
        
        data = self.dataset

        self.position_ppg = data[(data.season_age >= 17)
                                 & (data.season_age <= 24)].groupby('primary_position').ppg.mean().to_dict()


    def impute_games_played(self):
        
        team_mask = self.dataset.team.apply(lambda x : len(ast.literal_eval(x)) > 1)

        self.gp_dict = self.dataset[(self.dataset.gp < 85)
                               & (self.dataset.season_age >= 17)
                               & ~(team_mask)
                               & (self.dataset.season_age <= 24)].groupby('league').gp.mean().round().to_dict()
        
        self.gp_full = self.dataset[(self.dataset.gp < 85)
                               & (self.dataset.season_age >= 17)
                               & ~(team_mask)
                               & (self.dataset.season_age <= 24)].groupby('league').gp.apply(lambda x: mode(x)[0][0]).round().to_dict()
        
    def get_position(self):
        
        return get_primary_position(self.player_df.position.iloc[0])
        
    def get_player_index(self):
    
        return (self.player_df.playerid.iloc[0], self.player_df.player.iloc[0])
    
    def get_league_index(self):
        
        self.league_index = { i:k for i, k in enumerate(np.unique(self.dataset['league']))}
    
    def generate_league_season(self, player_df):
        
        # Pad player career to get entries for all possible seasons
        self.pad_player = pad_data(player_df, [self.get_player_index()])
        # Get age of the last played season
        self.current_age = self.pad_player.season_age.max()
        # create retrieve data from season and produce model features
        self.current_season = self.pad_player[self.pad_player.season_age == self.current_age]
        self.current_league = self.current_season.league.iloc[0]
        self.current_season = prepare_features_single_season(self.current_season, self.league_scaler, 'league_y_plus_1')
        # append model features to entire dataset to inherent all the necessary columns
        player_league_features = pd.DataFrame(columns=self.league_features).append(
            self.current_season.droplevel([0,1])).loc[self.current_age].to_frame()
        # predict next year league probabilities for the last played season
        league_pred = self.league_model.predict( # issues with predict_proba in new xgboost --- use predict
            player_league_features.T.fillna(0)[self.league_features]).round(3)
        # write league probabilities to player dataframe to iterate over
        league_probs = pd.DataFrame([[self.current_league,
                                      self.league_index[idx], 
                                      np.squeeze(league_pred)[idx]] for idx in np.flatnonzero(league_pred > 0)],
                                    columns=['start_league', 'league', 'probability'],
                                    index=pd.MultiIndex.from_tuples(
                                        [self.get_player_index() + (self.current_age + 1,)\
                                         for i in range(len(np.flatnonzero(league_pred > 0)))],
                                        names=['playerid', 'player', 'season_age'])
                                   )# incriment year
        
        # sort by probability
        league_probs.sort_values('probability', ascending=False, inplace=True)
        # only return values that > 2% probability
        league_probs = league_probs[((league_probs.probability >= 0.1) &
                                             ~((league_probs.league == 'NCAA')
                                              & (league_probs.start_league == 'AHL'))) \
                                              ].reset_index().head(3)

        
        # redistribute probabilities 
        league_probs.probability /= league_probs.probability.sum()
    
        league_probs['gp'] = league_probs.league.map(self.gp_dict)

        self.league_probs = league_probs

    def generate_league_scoring(self, player_df):

        import time

        self.league_probs['scoring_distribution'] = [[] for _ in range(len(self.league_probs))]

        for i, sim in self.league_probs.iterrows():
            # get next year data
            player_df['gp_y_plus_1'] = player_df.gp.shift(-1)
            player_df['league_y_plus_1'] = player_df.league.shift(-1)
            player_df.loc[player_df.gp_y_plus_1.isnull(), 'gp_y_plus_1'] = sim['gp']
            player_df.loc[player_df.league_y_plus_1.isnull(), 'league_y_plus_1'] = sim['league']

            # prepare dataset for lstm
            features = prepare_features_single_season(player_df, 
                                                    self.scoring_scaler, 
                                                    'ppg_y_plus_1')

            # fill dummy values
            features[[col for col in self.scoring_features if col not in features.columns]] = 0

            # get the season index we want to extract
            current_season_index = features.shape[0] - 1

            # pass player season sequences to lstm for prediction
            # start process to get predictions for all model versions create posterior distribution
            pool = Pool(processes=8)


            sequence = features[self.scoring_features].values
            preds = [item[current_season_index].item() for item in 
                        pool.starmap(predict_scoring, [(sequence, i) for i in range(100)])]

            pool.close()

            self.league_probs.at[i, 'ppg'] = np.mean(preds)
            self.league_probs.at[i, 'scoring_distribution'] = preds

        # partition ppg into assists and goals
        self.league_probs['gpg'] = self.league_probs.ppg * self.g_proportion
        self.league_probs['apg'] = self.league_probs.ppg * self.a_proportion
            
    def generate_league_scoring_baseline(self):

        # load entire model feature set
        self.load_features('ppg_y_plus_1')
        # instantiate league index
        self.get_league_index()
        # league to game played mapping
        self.impute_games_played()
        
        # Pad player career to get entries for all possible seasons
        pad_player = pad_data(self.player_df, [self.get_player_index()])
        self.current_age = pad_player.season_age.max()

        for i, (_, sim) in enumerate(self.league_probs.iterrows()):
            # create retrieve data from season and produce model features
            self.current_season = pad_player[pad_player.season_age == self.current_age]
            self.current_season['gp_y_plus_1'] = sim['gp_y_plus_1']
            self.current_season = prepare_features_single_season(self.current_season, self.scoring_scaler, 'ppg_y_plus_1')
            self.current_season['next_yr_{}'.format(sim['league'])] = 1
            # append model features to entire dataset to inherent all the necessary columns
            player_league_features = pd.DataFrame(self.features.loc[self.get_player_index()].append(
                self.current_season.droplevel([0,1])).loc[self.current_age])

            scoring_pred = self.scoring_model.predict(
                        player_league_features.T.fillna(0)[self.scoring_features]).round(3)

            self.league_probs.at[i, 'ppg'] = scoring_pred
    
    def interpolate_team_scoring(self):

        features = self.league_probs[['ppg', 'apg', 'gpg']].merge(
            pd.get_dummies(self.league_probs.reset_index()['league']),
            left_index=True,
            right_index=True).merge(
            pd.get_dummies(self.league_probs.reset_index()['season_age'].astype(int)),
            left_index=True,
            right_index=True)

        features.index = np.full((features.shape[0],), self.current_age + 1)

        X = pd.DataFrame(columns=self.scoring_features)
        for _, row in features.iterrows():
            X = X.append(row.to_frame().T)
            
        X = X.loc[(self.current_age + 1)]

        league_feats = [v for k,v in self.league_index.items()][1:]
        age_feats = [i for i in range(18, 25)]
        
        if isinstance(X, pd.DataFrame):
            input_dim = X.shape[0]
            try:
                X.drop(columns=['AHL'], inplace=True)
            except:
                pass

        else:
            input_dim = 1
            try:
                X.drop(index=['AHL'], inplace=True)
            except:
                pass

        self.league_probs['perc_team_a'] = self.apg_model.predict(
            X.fillna(0)[['apg'] + league_feats + age_feats].values.reshape(input_dim, self.apg_model.coef_.shape[0]))
        self.league_probs['perc_team_g'] = self.gpg_model.predict(
            X.fillna(0)[['gpg'] + league_feats + age_feats].values.reshape(input_dim, self.apg_model.coef_.shape[0]))
        self.league_probs['perc_team_tp'] = self.ppg_model.predict(
            X.fillna(0)[['ppg'] + league_feats + age_feats].values.reshape(input_dim, self.ppg_model.coef_.shape[0]))
    
    def recursive_season_concat(self, season, df = pd.DataFrame()):
        
        prev_season = self.projections[self.projections.node == season['start_node']]
        
        if prev_season.empty:
            
            return df.append(season.to_frame().T)

        else:
            df = df.append(season)

            return self.recursive_season_concat(prev_season.squeeze(), df).sort_values('season_age')

    def fill_player_attributes(self, season):
        
        df = self.player_df.append(self.recursive_season_concat(season))
        df['position'] = df['position'].fillna(method = 'ffill') 
        df['height'] = df['height'].fillna(method = 'ffill')
        df['weight'] = df['weight'].fillna(method = 'ffill')
        df['draft_year_eligible'] = df['draft_year_eligible'] .fillna(method = 'ffill') 
        df['draft_year'] = df['draft_year'] .fillna(method = 'ffill') 
        df['draft_round'] = df['draft_round'].fillna(method = 'ffill')
        df['draft_pick'] = df['draft_pick'].fillna(method = 'ffill')
        df['real_season_age'] = (df['real_season_age'].min() + np.arange(0, len(df)))
        df['start_year'] = (df['start_year'].min() + np.arange(0, len(df)))
        df['end_year'] = (df['end_year'].min() + np.arange(0, len(df)))

        return df
            
    def simulate_player_development(self, node_counter=1): 
        
        while self.current_age + 1 <= (24 if check_late_birthday(self.date_of_birth) else 23):
            if node_counter == 1:
                self.generate_league_season(self.player_df)
                self.generate_league_scoring(self.player_df)
                self.interpolate_team_scoring()
                self.league_probs['start_node'] = node_counter
                self.league_probs['node'] = np.arange(node_counter + 1, node_counter + 1 + len(self.league_probs))

                self.current_simulation = self.league_probs
                self.projections = self.projections.append(self.current_simulation)
                node_counter += len(self.league_probs)
                self.current_age += 1
            else:
                simulated_seasons = pd.DataFrame()

                for _, sim in self.current_simulation.iterrows():
                    player_df = self.fill_player_attributes(sim)
                    self.generate_league_season(player_df)
                    self.generate_league_scoring(player_df)
                    self.interpolate_team_scoring()
                    self.league_probs['start_node'] = sim['node']
                    self.league_probs['node'] = np.arange(node_counter + 1, node_counter + 1 + len(self.league_probs))
                    # Append predictions to projections dataframe
                    simulated_seasons = simulated_seasons.append(self.league_probs)
                    self.projections = self.projections.append(self.league_probs)
                    node_counter += len(self.league_probs)

                self.current_simulation = simulated_seasons

                self.current_age += 1

            self.simulate_player_development(node_counter)
        
        # to calculate epoints let players play full seasons
        self.projections['gp'] = self.projections.league.map(self.gp_dict)
        self.projections['tp'] = self.projections.ppg * self.projections.gp
        self.projections['g'] = self.projections.gp * self.projections.gpg 
        self.projections['a'] = self.projections.gp * self.projections.apg 
        
    def generate_network_graph(self):
        
        self.results = pd.concat([
        self.player_df[self.player_df.season_age == self.start_age],
        self.projections])

        self.results['role'] = self.results.apply(get_role, axis=1, dataset=self.dataset, position=self.position)
        self.results['role_rank'] = self.results.apply(get_role_rank, axis=1, dataset=self.dataset, position=self.position)

        self.results.loc[(self.results.season_age ==  self.start_age), ['node']] = 1

        self.G = nx.convert_matrix.from_pandas_edgelist(self.results.loc[self.results.start_node.notnull()].round(3),
                             source='start_node',
                             target='node',
                             edge_attr=['probability', 'tp', ],
                             create_using=nx.DiGraph())

        self.G.nodes[1]['league'] = self.start_league
        self.G.nodes[1]['age'] = int(self.start_age)
        self.G.nodes[1]['tp'] = self.player_df[self.player_df.season_age == self.start_age].tp.values.item()
        self.G.nodes[1]['ppg'] = self.player_df[self.player_df.season_age == self.start_age].round(3).ppg.values.item()
        self.G.nodes[1]['cond_prob'] = float(node_probability(self.G, 1))
        self.G.nodes[1]['xvalue'] = int(round(node_expected_value(self.G, 1),3))
        self.G.nodes[1]['role'] = self.results[self.results.node == 1].role.values.item()
        self.G.nodes[1]['role_rank'] = self.results[self.results.node == 1].role_rank.values.item()

        for n in self.results.loc[self.results.start_node.notnull()].node.values:
            self.G.nodes[n]['league'] = self.results[self.results.node == n].league.values.item()
            self.G.nodes[n]['age'] = int(self.results[self.results.node == n].season_age.values.item())
            self.G.nodes[n]['tp'] = self.results[self.results.node == n].tp.values.item()
            self.G.nodes[n]['ppg'] = self.results[self.results.node == n].ppg.round(3).values.item()
            self.G.nodes[n]['cond_prob'] = float(node_probability(self.G, n))
            self.G.nodes[n]['xvalue'] = int(round(node_expected_value(self.G, n),3))
            self.G.nodes[n]['role'] = self.results[self.results.node == n].role.values.item()
            self.G.nodes[n]['role_rank'] = self.results[self.results.node == n].role_rank.values.item()
            
        self.calculate_value_metrics()

        
    def plot_network_graph(self):
        
        matplotlib.rc('font',family='monospace')

        fig, ax = plt.subplots(figsize=(20,10))

        # create tree layout and rescale based on age
        pos = graphviz_layout(self.G, prog='dot')
        # array of ages for y-axis
        ages = np.arange(self.start_age,
                         self.projections.season_age.astype(int).max() + 1,
                         1)
        # rescale x,y coordinates
        y = np.array([xy[1] for _, xy in pos.items()])
        y_rescale = (y.max() / y.min()) / (ages.max() / ages.min())
        y_map = { k:v for k,v in zip(sorted(np.unique(y))[::-1], ages)}
        pos = {u :(v[0] / y_rescale, y_map[v[1]]) for u,v in pos.items()}

        # set params
#         node_sizes = [int(self.G.nodes[1]['ppg'] / self.position_ppg[self.position] * 300)] \
#         + [int(v * 300) for v in self.projections.ppg.div(self.position_ppg[self.position]).values.tolist()]
        node_sizes = [self.G.nodes[g]['ppg'] / self.position_ppg[self.position] * 500 for g in self.G.nodes]
        self.node_sizes = node_sizes

        # draw nodes
        nodes = nx.draw_networkx_nodes(self.G,
                                       pos,
                                       node_size=node_sizes,
                                       node_color='dodgerblue',
                                       ax=ax)
        node_labels = nx.get_node_attributes(self.G, 'league')
        nx.draw_networkx_labels(self.G,
                                pos,
                                node_labels,
                                font_size=14,
                                ax=ax, 
                                font_family='monospace', 
                                font_color='black', 
                                font_weight='bold',)

        # draw edges
        edges = nx.draw_networkx_edges(self.G,
                                       pos, 
                                       arrowstyle='->', 
                                       arrowsize=20, 
                                       edge_cmap=plt.cm.Greys, 
                                       width=3,
                                       ax=ax)
        edge_labels = nx.get_edge_attributes(self.G, 'probability')
        nx.draw_networkx_edge_labels(self.G, 
                                     pos, 
                                     edge_labels, 
                                     font_size=12,
                                     ax=ax, 
                                     font_family='monospace')

        # set alpha value for each edge
        for i in range(len(self.projections)):
            edges[i].set_alpha(self.projections.probability.values[i])

        # colourscale for colourbar
        pc = matplotlib.collections.PatchCollection(edges, cmap=plt.cm.Greys)
        pc.set_array([(5 + i) / (30 + 4) for i in range(30)])
        plt.colorbar(pc).ax.set_title('Transition %')
        # create legend for projected points
        custom_leg = [mlines.Line2D([0], [0], marker='o', color='dodgerblue', linestyle='None') for _ in range(0,5)]
        # 4 part distirubtion
        Q1 = np.percentile([self.G.nodes[g]['ppg'] for g in self.G.nodes], 25, interpolation = 'midpoint').round(1) 
        Q2 = np.percentile([self.G.nodes[g]['ppg'] for g in self.G.nodes], 50, interpolation = 'midpoint').round(1) 
        Q3 = np.percentile([self.G.nodes[g]['ppg'] for g in self.G.nodes], 75, interpolation = 'midpoint').round(1)
        Q4 = np.percentile([self.G.nodes[g]['ppg'] for g in self.G.nodes], 100, interpolation = 'midpoint').round(1)
        point_range=[Q1, Q2, Q3, Q4]
        self.point_range = point_range
        custom_leg = [mlines.Line2D([0], [0],
                                    marker='o',
                                    color='dodgerblue',
                                    markersize=point_range[i] / self.position_ppg[self.position] * 10 ,
                                    linestyle='None') for i in range(0,4)]
        leg = plt.legend(custom_leg, point_range,
                         title = 'Projected Points (PPG)',
                         labelspacing=2.5,
                         handletextpad=1.5,
                         borderpad=1.5,
                         ncol=4,)
            
        ax.annotate('', xytext=(-.075, 0),  xy=(-.075, 1),  xycoords='axes fraction', arrowprops=dict(color='black', width=2))

        # axis changes
        ax.set_yticks(ages)
        ax.tick_params(left=True, bottom=False, labelleft=True, labelbottom=False)
        ax.set_ylabel('Season Age', size=16, weight='bold')
        ax.set_xlabel('MeaningleX Axis', size=12)
        plt.gca().yaxis.grid(True, linestyle='--',)
        plt.title(f"{self.player_name} Player Development", fontsize=18, fontweight='bold')
        footnote = """By: Connor Jung @ConnorJungle, Data Source: Elite Prospects"""
        plt.figtext(0.125, 0.05, footnote)
        # save image
        fname = '_'.join(self.player_name.split()).lower()
        plt.savefig(f"../images/{fname}_{self.playerid}_player_development_network_graph.png",
                    dpi=300,
                    bbox_inches='tight',
                    pad_inches=1.2,
                    format='png')
        
        plt.show()
        
    def get_nhl_value(self):

        nodes = self.projections[(self.projections.start_node==1)].node

        return dict(nhl_expected_value=int(sum([nhl_expected_value(self.G, n) for n in nodes])))

    def get_nhl_path(self):

        nodes = self.projections[(self.projections.season_age==(24 if check_late_birthday(self.date_of_birth) else 23))
                                  & (self.projections.league=='NHL')].node

        probs = []
        max_prob = 0
        max_node = 1

        for node in nodes:
            node_prob = node_probability(self.G, node)
            if node_prob > max_prob:
                max_prob = node_prob
                max_node = node
            probs.append(node_prob)

        return dict(nhl_likelihood=round(sum(probs),2),
                     most_likely_nhl_node=max_node,
                     most_likely_nhl_prob=round(float(node_probability(self.G, max_node)),2),
                     nhl_floor=int(nhl_value(self.G, max_node))
                     )
    
    def get_nhl_path_ceiling(self):
        nodes = self.projections[(self.projections.season_age==(24 if check_late_birthday(self.date_of_birth) else 23))].node
        max_points = 0
        max_node = 1
        for node in nodes:
            node_points = nhl_value(self.G, node)
            if node_points > max_points:
                max_points = node_points
                max_node = node
        ceiling_prob = round(node_probability(self.G, max_node),2)
        return dict(nhl_ceiling=round(max_points, 0), nhl_maximizing_node=max_node, nhl_ceiling_prob=float(ceiling_prob))
    
    def calculate_value_metrics(self):
            
        player_value = dict(playerid = self.playerid,
                            player_name = self.player_name,
                            position = self.position)

        player_value.update(self.get_nhl_path())

        player_value.update(self.get_nhl_value())
        
        player_value.update(self.get_nhl_path_ceiling())

        self.player_value = player_value
            
def node_probability(G, node):
    
    probabilities = []
    for n in sorted(nx.ancestors(G, node))[::-1]:
        probabilities.append(G.edges[(n, node)]['probability'])
        node=n
    return np.prod(probabilities)

def nhl_value(G, node):

    return np.sum([G.nodes[node]['tp'] if (G.nodes[node]['league'] == 'NHL') & (node != 1) else 0] +
                  [G.nodes[n]['tp'] for n in nx.ancestors(G, node) if (G.nodes[n]['league'] == 'NHL') & (n != 1)]).round(1)

def node_expected_value(G, node):

    return sum(G.nodes[node]['tp'] * G.nodes[node]['cond_prob'] for n in nx.ancestors(G, node) | {node})

def nhl_expected_value(G, node):
        
    return np.sum([G.nodes[node]['tp'] * G.nodes[node]['cond_prob'] if G.nodes[node]['league'] == 'NHL' else 0] + \
                [G.nodes[n]['tp'] * G.nodes[n]['cond_prob']\
                   for n in nx.descendants(G, node) if G.nodes[n]['league'] == 'NHL']).round(1)

def get_role(row, dataset, position):
    
    subset = dataset.loc[(dataset.gp >= 20)
                                & (dataset.league == row['league'])
                                & (dataset.primary_position == position), ['ppg']]
    
    if position == 'D':
        return np.where(row['ppg'] > np.percentile(subset.ppg, 25),
                        np.where(row['ppg'] > np.percentile(subset.ppg, 50),
                                 np.where(row['ppg'] > np.percentile(subset.ppg, 75),
                                          np.where(row['ppg'] > np.percentile(subset.ppg, 90),
                                                   'Elite',
                                                   'Top Pair'
                                                  ),
                                          'Top 4'
                                         ),
                                 'Top 6'
                                ),
                        'Depth'
                       ).item()
    
    else:
        return np.where(row['ppg'] > np.percentile(subset.ppg, 25),
                        np.where(row['ppg'] > np.percentile(subset.ppg, 50),
                                 np.where(row['ppg'] > np.percentile(subset.ppg, 75),
                                          np.where(row['ppg'] > np.percentile(subset.ppg, 90),
                                                   'Elite',
                                                   'Top Line'
                                                  ),
                                          'Top 6'
                                         ),
                                 'Top 9'
                                ),
                        'Depth'
                       ).item()
    
def get_role_rank(row, dataset, position):
    
    subset = dataset.loc[(dataset.gp >= 20)
                                & (dataset.league == row['league'])
                                & (dataset.primary_position == position), ['ppg']]
    
    return round(stats.percentileofscore(subset.ppg, row['ppg'], kind='strict'))
