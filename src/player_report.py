from IPython.display import display, Markdown
import matplotlib.pyplot as plt
import pandas as pd

import generate_player_seasons as g

def player_header(player):
    
    first = ['date_of_birth', 'nation', 'height', 'weight', 'shoots'] # first row of player info
    second = ['rights', 'draft_year', 'draft_year', 'draft_round', 'draft_team'] # second row of player info

    blurb = f'''
    APPLE projects {player.player_name} to have a {player.player_value['nhl_likelihood'] * 100}% chance to make the NHL by age 23. 
    {player.player_name}'s simulated development path over the 5 seasons since being drafted 
    has a expected NHL Value of ~{player.player_value['nhl_expected_value']} points.

    {player.player_name}'s "Most Likely NHL Development Path" or projected floor is ~{player.player_value['nhl_floor']} points
    {player.player_name}'s "Maximum NHL Value Development Path" or projected ceiling is ~{player.player_value['nhl_ceiling']} points
            '''

    display(Markdown(f'# {player.player_name}'))
    display(Markdown(blurb))
    display(Markdown('&ensp;'.join([': '.join([f'**{col}**', player.player_df[col].astype(str).iloc[0]]) for col in first])))
    display(Markdown('&ensp;'.join([': '.join([f'**{col}**', player.player_df[col].astype(str).iloc[0]]) for col in second])))

def player_seasons_table(player):
    
    col_order = ['year', 'league', 'season_age', 'gp', 'g', 'a', 'tp', 'gpg', 'apg', 'ppg']

    return player.player_df[col_order].astype({'gp':int,'g':int, 'a':int, 
                                               'tp': int,'ppg': float, 
                                               'apg':float, 'gpg':float,
                                               'season_age':int}).round({'ppg': 2, 'apg':2, 'gpg':2}).to_markdown()

def player_ceiling_table(player):

    season = player.projections[player.projections.node == player.player_value['nhl_maximizing_node']]

    col_order = ['league', 'season_age', 'probability', 'gp', 'egoals', 
                 'eassists', 'epoints', 'gpg', 'apg', 'ppg']
    
    return player.recursive_season_concat(season.squeeze())[col_order].astype({
        'egoals':int, 'eassists':int, 
        'epoints': int,'ppg': float, 
        'apg':float, 'gpg':float,
        'probability':float}).round({'ppg': 2, 'apg':2, 'gpg':2, 'probability' : 2}).to_markdown()

def player_floor_table(player):

    season = player.projections[player.projections.node == player.player_value['most_likely_nhl_node']]

    col_order = ['league', 'season_age', 'probability', 'gp', 'egoals', 
                 'eassists', 'epoints', 'gpg', 'apg', 'ppg']
    
    return player.recursive_season_concat(season.squeeze())[col_order].astype({
        'egoals':int, 'eassists':int, 
        'epoints': int,'ppg': float, 
        'apg':float, 'gpg':float,
        'probability':float}).round({'ppg': 2, 'apg':2, 'gpg':2, 'probability' : 2}).to_markdown()


def create_player_report(playerid):
    
    player = g.GeneratePlayer()

    player.initialize_player(playerid)
    player.simulate_player_development()
    player.generate_network_graph()
    
    player.projections['egoals'] = player.projections.gp * player.projections.gpg 
    player.projections['eassists'] = player.projections.gp * player.projections.apg 

    # print player information in markdown
    player_header(player)
    # print dataframe of player current stats
    display(Markdown(f'### Player Stats'))
    display(Markdown(player_seasons_table(player)))
    # plot player development graph
    player.plot_network_graph()
    # print player floor table
    display(Markdown(f'### Most Likely Player Development Path'))
    display(Markdown(player_floor_table(player)))
    # print player ceiling table
    display(Markdown(f'### Maximum NHL Value Development Path'))
    display(Markdown(player_ceiling_table(player)))
    fig, ax = plt.subplots(figsize=(20,10))
    plt.title('Notes', fontsize=18, fontweight='bold', loc='left')
    plt.tick_params(bottom=False, left=False, labelbottom = False, labelleft=False)
    plt.show()