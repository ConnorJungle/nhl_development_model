from data_processing import *

from scoring_lstm import Model

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

league_model_col_order = ['forward', 'gp', 'draft_pick', 'is_drafted', 'height', 'weight', 'real_season_age',
                        'gpg', 'apg', 'ppg', 'perc_team_g', 'perc_team_a', 'perc_team_tp', 'AJHL', 
                        'Allsvenskan', 'BCHL', 'CCHL', 'Czech', 'Czech2', 'Jr. A SM-liiga', 'KHL', 'Liiga',
                        'MHL', 'NCAA', 'NHL', 'NLA', 'OHL', 'OJHL', 'QMJHL', 'SHL', 'SJHL', 'SuperElit', 
                        'USHL', 'VHL', 'WHL', 18, 19, 20, 21, 22, 23, 'round_1.0', 
                        'round_2.0', 'round_3.0', 'round_4.0', 'round_5.0', 
                        'round_6.0', 'round_7.0', 'round_8.0', 'round_9.0']

scoring_model_col_order =['forward', 'gp', 'gp_y_plus_1', 'draft_pick', 'is_drafted',
                          'height', 'weight', 'real_season_age', 'gpg', 'apg', 'ppg', 'perc_team_g', 
                          'perc_team_a', 'perc_team_tp', 'AJHL', 'Allsvenskan', 'BCHL', 'CCHL', 'Czech',
                          'Czech2', 'Jr. A SM-liiga', 'KHL', 'Liiga', 'MHL', 'NCAA', 'NHL', 'NLA',
                          'OHL', 'OJHL', 'QMJHL', 'SHL', 'SJHL', 'SuperElit', 'USHL', 'VHL', 'WHL', 
                          18, 19, 20, 21, 22, 23, 'round_1.0', 'round_2.0',
                          'round_3.0', 'round_4.0', 'round_5.0', 'round_6.0', 'round_7.0',
                          'round_8.0', 'round_9.0', 'next_yr_AJHL', 'next_yr_Allsvenskan', 
                          'next_yr_BCHL', 'next_yr_CCHL', 'next_yr_Czech', 'next_yr_Czech2', 
                          'next_yr_Jr. A SM-liiga', 'next_yr_KHL', 'next_yr_Liiga', 
                          'next_yr_MHL', 'next_yr_NCAA', 'next_yr_NHL', 'next_yr_NLA',
                          'next_yr_OHL', 'next_yr_OJHL', 'next_yr_QMJHL', 
                          'next_yr_SHL', 'next_yr_SJHL', 'next_yr_SuperElit', 
                          'next_yr_USHL', 'next_yr_VHL', 'next_yr_WHL']

pd.set_option('max_column', 0)
warnings.filterwarnings('ignore')

class GeneratePlayer(object):
    def __init__(self):
        self.dataset = pd.read_csv('../data/player_season_stats.csv')
        
        self.league_model = joblib.load('../models/predict_next_league.pkl')
        self.scoring_model = joblib.load('../models/scoring_model.pkl')
        self.scoring_lstm = Model() 
        self.scoring_lstm.load_state_dict(torch.load('../models/scoring_lstm_checkpoint.pth'))
        self.scoring_lstm.eval()
        
        self.ppg_model = joblib.load('../models/ppg_team_model.pkl')
        self.gpg_model = joblib.load('../models/gpg_team_model.pkl')
        self.apg_model = joblib.load('../models/apg_team_model.pkl')
        
    def initialize_player(self, playerid):
        self.playerid = playerid
        self.player_df = self.dataset[self.dataset.playerid == int(playerid)]
        self.player_name = self.player_df.player.iloc[0]
        self.current_age = self.player_df.season_age.max()
        self.start_age = self.player_df.season_age.max()
        self.start_league = self.player_df[self.player_df.season_age == self.player_df.season_age.max()].league.iloc[0]
        self.a_proportion = (self.player_df.apg / self.player_df.ppg).median()
        self.g_proportion = 1 - self.a_proportion
        self.projections = pd.DataFrame()
        
    def impute_games_played(self):
        
        team_mask = self.dataset.team.apply(lambda x : len(ast.literal_eval(x)) > 1)

        self.gp_dict = self.dataset[(self.dataset.gp < 85)
                               & (self.dataset.season_age >= 17)
                               & ~(team_mask)
                               & (self.dataset.season_age <= 23)].groupby('league').gp.mean().round().to_dict()
        
        self.gp_full = self.dataset[(self.dataset.gp < 85)
                               & (self.dataset.season_age >= 17)
                               & ~(team_mask)
                               & (self.dataset.season_age <= 23)].groupby('league').gp.apply(lambda x: mode(x)[0][0]).round().to_dict()
        
        
    def load_features(self, target):
        
        self.features, self.target, self.scaler = prepare_features(self.dataset, target)
        
    def get_player_index(self):
    
        return (self.player_df.playerid.iloc[0], self.player_df.player.iloc[0])
    
    def get_league_index(self):
        
        self.league_index = { i:k for i, k in enumerate(np.unique(self.dataset['league']))}
    
    def generate_league_season(self, player_df):
        
        # load entire model feature set
        self.load_features('league_y_plus_1')
        # instantiate league index
        self.get_league_index()
        # league to game played mapping
        self.impute_games_played()
        # Pad player career to get entries for all possible seasons
        self.pad_player = pad_data(player_df, [self.get_player_index()])
        # Get age of the last played season
        self.current_age = self.pad_player.season_age.max()
        # create retrieve data from season and produce model features
        self.current_season = self.pad_player[self.pad_player.season_age == self.current_age]
        self.current_league = self.current_season.league.iloc[0]
        self.current_season = prepare_features_single_season(self.current_season, self.scaler, 'league_y_plus_1')
        # append model features to entire dataset to inherent all the necessary columns
        player_league_features = pd.DataFrame(self.features.loc[self.get_player_index()].append(
            self.current_season.droplevel([0,1])).loc[self.current_age])
        # predict next year league probabilities for the last played season
        league_pred = self.league_model.predict_proba(
            player_league_features.T.fillna(0)[league_model_col_order]).round(3)
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
        league_probs = league_probs[(league_probs.probability >= 0.1) |
                                              (league_probs.league == 'NHL')].reset_index().head(3)
        # redistribute probabilities 
        league_probs.probability /= league_probs.probability.sum()
    
        league_probs['gp'] = league_probs.league.map(self.gp_dict)

        self.league_probs = league_probs
        
    def generate_league_scoring(self, player_df):
        # load entire model feature set
        self.load_features('ppg_y_plus_1')

        for i, (_, sim) in enumerate(self.league_probs.iterrows()):
            # create retrieve data from season and produce model features
            self.current_season = self.pad_player[self.pad_player.season_age == self.current_age]
            self.current_season['gp_y_plus_1'] = sim['gp']
            self.current_season = prepare_features_single_season(self.current_season, 
                                                                       self.scaler, 
                                                                       'ppg_y_plus_1')
            self.current_season['next_yr_{}'.format(sim['league'])] = 1
            # player features for lstm
            player_league_features = pd.DataFrame(
                self.features.loc[self.get_player_index()]\
                                        .append(self.current_season.droplevel([0,1]))
            ).fillna(0)[scoring_model_col_order]

            # pad sequence for LSTM
            player_league_features = pad_sequence(player_league_features).values

            with torch.no_grad():
                lstm_preds = torch.Tensor([])  # to store our predictions for season t+1
                pred_t, _ = self.scoring_lstm(player_league_features,
                                                 np.ndarray((player_league_features.shape[0], 1))) # dummy target array
                lstm_preds = torch.cat((lstm_preds, pred_t))

            self.league_probs.ix[i, 'ppg'] = lstm_preds.detach().numpy()[-1]
        
        # partition ppg into assists and goals
        self.league_probs['gpg'] = self.league_probs.ppg * self.g_proportion
        self.league_probs['apg'] = self.league_probs.ppg * self.a_proportion
        # Append predictions to projections dataframe
#         self.projections = self.projections.append(self.league_probs.reset_index())
            
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
            self.current_season = prepare_features_single_season(self.current_season, self.scaler, 'ppg_y_plus_1')
            self.current_season['next_yr_{}'.format(sim['league'])] = 1
            # append model features to entire dataset to inherent all the necessary columns
            player_league_features = pd.DataFrame(self.features.loc[self.get_player_index()].append(
                self.current_season.droplevel([0,1])).loc[self.current_age])

            scoring_pred = self.scoring_model.predict(
                        player_league_features.T.fillna(0)[scoring_model_col_order]).round(3)

            self.league_probs.ix[i, 'ppg'] = scoring_pred
    
    def interpolate_team_scoring(self):

        features = self.league_probs[['ppg', 'apg', 'gpg']].merge(
            pd.get_dummies(self.league_probs.reset_index()['league']),
            left_index=True,
            right_index=True).merge(
            pd.get_dummies(self.league_probs.reset_index()['season_age'].astype(int)),
            left_index=True,
            right_index=True)

        features.index = np.full((features.shape[0],), self.current_age + 1)

        X = self.features.loc[self.get_player_index()]
        for _, row in features.iterrows():
            X = X.append(row.to_frame().T)
            
        X = X.loc[(self.current_age + 1)]

        league_feats = [v for k,v in self.league_index.items()][1:]
        age_feats = [i for i in range(18, 24)]
        
        input_dim = X.shape[0] if X.shape[0] != self.features.shape[1] else 1
        
        self.league_probs['perc_team_a'] = self.apg_model.predict(
            X.fillna(0)[['apg'] + league_feats + age_feats].values.reshape(input_dim, self.apg_model.coef_.shape[0]))
        self.league_probs['perc_team_g'] = self.gpg_model.predict(
            X.fillna(0)[['gpg'] + league_feats + age_feats].values.reshape(input_dim, self.apg_model.coef_.shape[0]))
        self.league_probs['perc_team_tp'] = self.ppg_model.predict(
            X.fillna(0)[['ppg'] + league_feats + age_feats].values.reshape(input_dim, self.ppg_model.coef_.shape[0]))

    def collect_expected_points(self):
    
        return self.league_probs.groupby(['playerid', 'player', 'age', 'league']).prod(axis=1).prod(axis=1)
    
    def recursive_season_concat(self, season, df = pd.DataFrame()):
        
        prev_season = self.projections[self.projections.node == season['start_node']]
        
        if prev_season.empty:
            
            return df.append(season.to_frame().T)

        else:
            df = df.append(season)
#             season = df.loc[df.season_age == df.season_age.min()].squeeze()

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
        df['real_season_age'] = (df['real_season_age'].min() + np.arange(1, len(df) + 1))
        df['start_year'] = (df['start_year'].min() + np.arange(1, len(df) + 1))
        df['end_year'] = (df['end_year'].min() + np.arange(1, len(df) + 1))

        return df
            
    def simulate_player_development(self, node_counter=1): 
    
        while self.current_age + 1 <= 23:

            print(f'--- Simulating Seasons --- {self.player_name} --- Age: {int(self.current_age)}')

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
                print(f'--- Simulation Complete --- {self.player_name} --- Age: {int(self.current_age)}')


            print(f'--- Simulating Seasons --- {self.player_name} --- Age: {int(self.current_age)}')
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
            print(f'--- Simulation Complete --- {self.player_name} --- Age: {int(self.current_age)}')
            self.current_age += 1

            self.simulate_player_development(node_counter)
        
        # to calculate epoints let players play full seasons
        self.projections['gp'] = self.projections.league.map(self.gp_full)
        self.projections['epoints'] = self.projections.ppg * self.projections.gp
        
    def generate_network_graph(self):
        
        self.G = nx.convert_matrix.from_pandas_edgelist(self.projections.round(3),
                             source='start_node',
                             target='node',
                             edge_attr=['probability', 'epoints'],
                             create_using=nx.DiGraph())
        
        self.G.nodes[1]['league'] = self.start_league
        self.G.nodes[1]['age'] = self.start_age
        self.G.nodes[1]['epoints'] = self.player_df[self.player_df.season_age == self.player_df.season_age.max()].tp.values.item()
        self.G.nodes[1]['cond_prob'] = node_probability(self.G, 1)
        self.G.nodes[1]['xvalue'] = round(node_expected_value(self.G, 1),3)
        
        for n in self.projections.node.values:
            self.G.nodes[n]['league'] = self.projections[self.projections.node == n].league.values.item()
            self.G.nodes[n]['age'] = self.projections[self.projections.node == n].season_age.values.item()
            self.G.nodes[n]['epoints'] = self.projections[self.projections.node == n].epoints.values.item()
            self.G.nodes[n]['cond_prob'] = node_probability(self.G, n)
            self.G.nodes[n]['xvalue'] = round(node_expected_value(self.G, n),3)
        
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
        node_sizes = [int(self.G.nodes[1]['epoints'] *10)] + [int(v * 10) for v in self.projections.epoints.values.tolist()]

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
                                font_weight='bold',
                                font_stretch = 'semi-expanded')

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
        plt.colorbar(pc)
        # create legend for projected points
        custom_leg = [mlines.Line2D([0], [0], marker='o', color='dodgerblue', linestyle='None') for _ in range(0,5)]
        # 4 part distirubtion
        Q1 = np.percentile(self.projections.epoints.unique(), 25, interpolation = 'midpoint').round(1) 
        Q2 = np.percentile(self.projections.epoints.unique(), 50, interpolation = 'midpoint').round(1) 
        Q3 = np.percentile(self.projections.epoints.unique(), 75, interpolation = 'midpoint').round(1)
        Q4 = np.percentile(self.projections.epoints.unique(), 100, interpolation = 'midpoint').round(1)
        point_range=[Q4, Q3, Q2, Q1]
        leg = plt.legend(custom_leg, point_range, title = 'Projected Points')
        for i, points in enumerate(point_range):
            leg.legendHandles[i]._legmarker.set_markersize(points / 5)

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
           format='png')
        
        plt.show()
        
    def calculate_value_metrics(self):
            
        self.projections = self.projections.assign(nhl_xvalue=np.nan,xvalue=np.nan,cond_probability=np.nan,)
            
        for n in self.projections.node.values:
            # add expected value to data frame
            self.projections.loc[self.projections.node == n, ['nhl_xvalue']] = nhl_expected_value(self.G, n)
            self.projections.loc[self.projections.node == n, ['xvalue']] = node_expected_value(self.G, n)
            self.projections.loc[self.projections.node == n, ['cond_probability']] = node_probability(self.G, n)
            
    def run(self, playerid):
    
        self.initialize_player(playerid)
        self.simulate_player_development()
        self.generate_network_graph()
        self.plot_network_graph()

        return self.projections
        
def node_probability(G, node):
    
    probabilities = []
    for n in sorted(nx.ancestors(G, node))[::-1]:
        probabilities.append(G.edges[(n, node)]['probability'])
        node=n
    return np.prod(probabilities)

def node_expected_value(G, node):

    return sum(G.nodes[node]['epoints'] * G.nodes[node]['cond_prob'] for n in nx.ancestors(G, node) | {node})

def nhl_expected_value(G, node):
    
    value = []
    for n in sorted(nx.descendants(G, node)):
        value.append(G.nodes[n]['xvalue'] if G.nodes[n]['league'] == 'NHL' else 0)
    return np.sum(value)

def nhl_expected_value(G, node):
        
    return np.sum([G.nodes[n]['xvalue'] for n in nx.descendants(G, node) if G.nodes[n]['league'] == 'NHL']).round(1)
