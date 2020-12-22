import torch.optim as optim
import torch.nn as nn
import torch
import time

class Model(nn.Module):
    def __init__(self, input_size=80, hidden_size=160, lstm_layers=2, output_size=1, drop=0.2, use_cuda=False):
        
        super().__init__()
        self.start = time.time()
        self.hidden_size = hidden_size
        self.lstm_layers = lstm_layers
        self.lstm = nn.LSTM(input_size, hidden_size, lstm_layers, batch_first=True,
#                             dropout=drop
                           )
        self.linear = nn.Linear(hidden_size, output_size)  
        self.relu = nn.ReLU()
        
        device = torch.cuda.is_available()
        self.device = torch.device("cuda:0" if use_cuda and device else "cpu")

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

class Trainer(object):
    def __init__(self, train_sequences, train_targets, test_sequences, test_targets, model, epochs=200, lr=0.01, batch_size=1, log_per=10000):
        self.start = time.time()
        self.epochs = epochs
        self.log_per = log_per
        self.train_sequences = train_sequences
        self.train_targets = train_targets
        self.test_sequences = test_sequences
        self.test_targets = test_targets
        self.batch_size = batch_size
        self.lr = lr
        self.loss_fn = nn.MSELoss()
        self.model = model
        
    def train(self):

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        
        train_epoch_loss = []
        test_epoch_loss = []
        for ep in range(1, self.epochs):
            step = 0
            print(f'Running epoch: {ep}')
            for seasons, targets in zip(self.train_sequences, self.train_targets): # data is a list returning tuple of X, y
                step += 1
                self.optimizer.zero_grad()
                predictions, ground_truth = self.model(seasons, targets)    
                # now here, we want to compute the loss between the predicted values
                # for each season and the actual values for each season
                # TO-DO: random select grouth truth or predicted value in next timestep
                loss = self.loss_fn(predictions, ground_truth) 
                loss.backward()
                self.optimizer.step() 

            # validate with test set
            with torch.no_grad():
                for seasons, targets in zip(self.test_sequences, self.test_targets):
                    predictions, ground_truth = self.model(seasons, targets)    
                    # now here, we want to compute the loss between the predicted values
                    # for each season and the actual values for each season
                    # TO-DO: random select grouth truth or predicted value in next timestep
                    test_loss = self.loss_fn(predictions, ground_truth) 

            train_epoch_loss.append(loss.item())
            test_epoch_loss.append(test_loss.item())

            if ep%5 == 1:
                print(f'epoch: {ep:3} loss: {loss.item():10.8f}')
                print(f'epoch: {ep:3} test loss: {test_loss.item():10.8f}')

        self.train_loss = train_epoch_loss
        self.test_loss = test_epoch_loss

        print(f'epoch: {ep:3} loss: {loss.item():10.10f}')
        print(f'epoch: {ep:3} test loss: {test_loss.item():10.8f}')
        print(f'Total Model Training Runtime: {(time.time() - self.start)//60:10.8f} mins')
