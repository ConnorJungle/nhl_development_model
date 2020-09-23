import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import os
import json
import argparse
import logging
import sys

import optuna
from optuna.trial import FixedTrial

from data_processing import *

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 10
LOG_INTERVAL = 10

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

# Get the player dataset.
X, y, train_sequences, train_targets, test_sequences, test_targets, train_idx, test_idx, train_real_values, test_real_values, train_idx_bool, test_idx_bool = get_player_dataset()


def get_player_dataset(target='ppg_y_plus_1'):

    df = pd.read_csv('../data/player_season_stats.csv')

    X, y,_ = prepare_features(df, target)

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

    return X, y, train_seq, train_target, test_seq, test_target, train_idx, test_idx, train_real_values, test_real_values, train_idx_bool, test_idx_bool

def save_model(model, model_dir, trial_number):
    logger.info("Saving the model_{}.".format(trial_number))
    path = os.path.join(model_dir, 'model_{}.pth'.format(trial_number))
    # recommended way from http://pytorch.org/docs/master/notes/serialization.html
    torch.save(model.cpu().state_dict(), path)
    
def define_model(trial):
    # We optimize the number of layers, hidden untis and dropout ratio in each layer.
    lstm_layers = trial.suggest_int("lstm_layers", 2, 4)
    lstm_hidden = trial.suggest_int("lstm_hidden", low=73, high=73*3, step=73)

    return Model(lstm_layers=lstm_layers, hidden_size=lstm_hidden)

class Model(nn.Module):
    def __init__(self, input_size=74, hidden_size=74, lstm_layers=3, output_size=1, drop=0.2):
        
        super().__init__()
        self.start = time.time()
        self.hidden_size = hidden_size
        self.lstm_layers = lstm_layers
        self.lstm = nn.LSTM(input_size, hidden_size, lstm_layers, batch_first=True,
#                             dropout=drop
                           )
        self.linear = nn.Linear(hidden_size, output_size)  
        self.relu = nn.ReLU()

    def forward(self, seasons):

        ht = torch.zeros(self.lstm_layers, 1, self.hidden_size)   # initialize hidden state
        ct = torch.zeros(self.lstm_layers, 1, self.hidden_size)  # initialize cell state
        predictions = torch.Tensor([]) # to store our predictions for season t+1
        
        hidden = (ht, ct)
        
        for idx, season in enumerate(seasons):  # here we want to iterate over the time dimension
            lstm_input = torch.FloatTensor(season).view(1,1,len(season)) # LSTM takes 3D tensor
            out, hidden = self.lstm(lstm_input, hidden) # LSTM updates hidden state and returns output
            pred_t = self.linear(out) # pass LSTM output through a linear activation function
            pred_t = self.relu(pred_t) # since performance is non-negative we apply ReLU
            
            predictions = torch.cat((predictions, pred_t)) # concatenate all the predictions

        return predictions
    
def objective(trial):

    # Generate the model.
    model = define_model(trial).to(DEVICE)

    # Generate the optimizers.
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD"])
    lr = trial.suggest_float("lr", 1e-3, 1e-1, log=True)
    optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr)
    
    loss_fn = nn.MSELoss()
    
    # to track the average training loss per epoch as the model trains
    avg_train_losses = []
    # to track the average validation loss per epoch as the model trains
    avg_valid_losses = [] 

    # Training of the model.
    model.train()
    for epoch in range(EPOCHS):
        
        train_loss = []
        test_loss = []
        print(f'Running epoch: {epoch}')
        for seasons, targets in zip(train_sequences, train_targets): # data is a list returning tuple of X, y

            optimizer.zero_grad()

            mask = torch.tensor([(season > -1).all() for season in seasons]) # create mask for real seasons only      
            targets = torch.FloatTensor(targets)[mask]

            predictions = model(seasons)    
            # now here, we want to compute the loss between the predicted values
            # for each season and the actual values for each season
            # TO-DO: random select grouth truth or predicted value in next timestep
            loss = loss_fn(predictions[mask].squeeze(1), targets) 
            loss.backward()
            optimizer.step() 
            train_loss.append(loss.item())
            
        avg_train_losses.append(np.nanmean(train_loss))
            
    # validate with test set
    model.eval()
    with torch.no_grad():
        for seasons, targets in zip(test_sequences, test_targets):
            mask = torch.tensor([(season > -1).all() for season in seasons]) # create mask for real seasons only      
            targets = torch.FloatTensor(targets)[mask]

            predictions = model(seasons)    
            # now here, we want to compute the loss between the predicted values
            # for each season and the actual values for each season
            # TO-DO: random select grouth truth or predicted value in next timestep
            loss = self.loss_fn(predictions[mask].squeeze(1), targets) 
            test_loss.append(loss.item())

    
    print(f'epoch: {ep:3} train avg. loss: {np.nanmean(avg_train_losses):10.8f}')
    print(f'epoch: {ep:3} test loss: {np.nanmean(test_loss):10.8f}')

    save_model(model, '/tmp', trial.number)
    
    trial.set_user_attr('job_name', args.training_env['job_name'])
    
    return np.nanmean(avg_train_losses)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # To configure Optuna db
    parser.add_argument('--host', type=str)
    parser.add_argument('--db-name', type=str, default='optuna')
    parser.add_argument('--db-secret', type=str, default='demo/optuna/db')
    parser.add_argument('--study-name', type=str, default='lstm-pytorch-optune')
    parser.add_argument('--region-name', type=str, default='us-east-1')
    parser.add_argument('--n-trials', type=int, default=50)
    
    # Data, model, and output directories These are required.
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--data-dir', type=str, default=os.environ['SM_CHANNEL_TRAINING'])
    parser.add_argument('--training-env', type=str, default=json.loads(os.environ['SM_TRAINING_ENV']))
    
    args, _ = parser.parse_known_args()    

    secret = get_secret(args.db_secret, args.region_name)
    connector = 'pymysql'
    db = 'mysql+{}://{}:{}@{}/{}'.format(connector, secret['username'], secret['password'], args.host, args.db_name)

    study = optuna.study.load_study(study_name=args.study_name, storage=db, direction='minimize', load_if_exists=True)
    study.optimize(objective, n_trials=args.n_trials)

    logger.info("Number of finished trials: {}".format(len(study.trials)))

    logger.info("Best trial:")
    trial = study.best_trial

    logger.info("  Value: {}".format(trial.value))

    logger.info("  Params: ")
    for key, value in trial.params.items():
        logger.info("    {}: {}".format(key, value))
        
    # retrieve and save the best model
    try:
        model = define_model(FixedTrial(trial.params)).to(DEVICE)
        with open(os.path.join('/tmp', 'model_{}.pth'.format(trial.number)), 'rb') as f:
            model.load_state_dict(torch.load(f))
            
        torch.save(model.cpu().state_dict(), os.path.join(args.model_dir, 'model.pth'))
        torch.save(trial.params, os.path.join(args.model_dir, 'params.pth'))
        logger.info('    Model saved: model_{}.npz'.format(trial.number))
    except Exception as e: 
        logger.info('    Save failed: {}'.format(e))