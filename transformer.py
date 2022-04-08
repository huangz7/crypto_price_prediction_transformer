import os
import random
import math
import datetime
import time
import numpy as np
import h5py
import matplotlib.pyplot as plt
from IPython.display import HTML
from PIL import Image
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.utils import data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torchvision import transforms as T
from torchvision.datasets import ImageFolder
from torch.autograd import Variable
from torchvision.utils import save_image
from torch.backends import cudnn

import torchvision
import torchvision.transforms as transforms
from torch import nn, optim

import requests
import pandas as pd
import json

# Change this part to locate the correct directory where dataset resides
load_from_drive = False

if load_from_drive:
    from google.colab import drive
    drive.mount('/content/gdrive')
    ## Update the experiments directory
    EXPERIMENTS_DIRECTORY = '/content/gdrive/My Drive/Datasets/shared/experiments/'
    DATA_DIRECTORY = '/content/gdrive/My Drive/Datasets/shared/experiments/data/'
    CELEBA_GOOGLE_DRIVE_PATH = DATA_DIRECTORY + 'data.hdf5'
    IMDB_REVIEWS_FILE_PATH = DATA_DIRECTORY + 'data/'

%matplotlib inline

cudnn.benchmark = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set random seed for reproducibility
manualSeed = 999
# manualSeed = random.randint(1, 10000) # use if you want new results
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
np.random.seed(manualSeed)
torch.manual_seed(manualSeed)

input_window = 100
output_window = 5
batch_size = 10
eval_batch_size = 1

## PositionalEncoding Class 
# Reference: https://github.com/pytorch/tutorials/blob/master/beginner_source/transformer_tutorial.py
class TransformerModel(nn.Module):
    def __init__(self, ntoken: int, d_model: int, nhead: int, d_hid: int,
                 nlayers: int, dropout: float = 0.5):
        super().__init__()
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, d_hid, dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, nlayers)
        self.encoder = nn.Embedding(ntoken, d_model)
        self.d_model = d_model
        self.decoder = nn.Linear(d_model, ntoken)

        self.init_weights()

        self.src_mask = None

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange) 
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src, src_mask):
        #src = self.encoder(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, src_mask)
        output = self.decoder(output)
        return output

def generate_square_subsequent_mask(sz: int):
    """Generates an upper-triangular matrix of -inf, with zeros on diag."""
    return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

def data_process():
    # Reference: https://github.com/dushyant18033/BTC-Price-Prediction-ML-Project/blob/master/ARIMA.py
    # Fetching data from the server
    url = "https://web-api.coinmarketcap.com/v1/cryptocurrency/ohlcv/historical"
    param = {"convert":"USD","slug":"bitcoin","time_end":"1601510400","time_start":"1367107200"}
    content = requests.get(url=url, params=param).json()
    df = pd.json_normalize(content['data']['quotes'])

    # Extracting and renaming the important variables
    df['Date'] = pd.to_datetime(df['quote.USD.timestamp']).dt.tz_localize(None)
    df['Low'] = df['quote.USD.low']
    df['High'] = df['quote.USD.high']
    df['Open'] = df['quote.USD.open']
    df['Close'] = df['quote.USD.close']
    df['Volume'] = df['quote.USD.volume']

    # Drop original and redundant columns
    df = df.drop(columns=['time_open','time_close','time_high','time_low', 'quote.USD.low', 'quote.USD.high', 'quote.USD.open', 'quote.USD.close', 'quote.USD.volume', 'quote.USD.market_cap', 'quote.USD.timestamp'])
    df = df.dropna()
    series = df['Open'].to_numpy()

    max_time = 800
    time = np.arange(0, min(max_time, 0.1 * len(series)), 0.1)
    series = series[:len(time)]
    
    # Normalize
    scaler = MinMaxScaler(feature_range=(-1, 1))
    series = scaler.fit_transform(series.reshape(-1, 1)).reshape(-1) 
    
    #split dara
    train_data_ratio = 0.8
    data_split_index = int(0.8 * len(series))
    train_data = series[:data_split_index]
    test_data = series[data_split_index:]

    # convert our train data into a pytorch train tensor
    def create_inout_sequences(input_data, input_window):
        inout_seq = []
        L = len(input_data)
        for i in range(L-input_window):
            train_seq = np.append(input_data[i:i+input_window][:-output_window] , output_window * [0])
            train_label = input_data[i:i+input_window]
            inout_seq.append((train_seq ,train_label))
        return torch.FloatTensor(inout_seq)
    train_sequence = create_inout_sequences(train_data, input_window)
    train_sequence = train_sequence[:-output_window]

    test_data = create_inout_sequences(test_data, input_window)
    test_data = test_data[:-output_window]

    print(f'training data size: {train_sequence.size()} testing data size: {test_data.size()}')
    return train_sequence.to(device), test_data.to(device)

def get_batch(source, idx, manual_batch_size):
    seq_len = min(manual_batch_size, len(source) - 1 - idx)
    data = source[idx:idx+seq_len]    
    input = torch.stack(torch.stack([item[0] for item in data]).chunk(input_window,1)) # 1 is feature size
    target = torch.stack(torch.stack([item[1] for item in data]).chunk(input_window,1))
    return input, target

def train(train_data):
    model.train() # Turn on the train mode
    total_loss = 0.
    log_interval = int(len(train_data) / batch_size / 5)
    start_time = time.time()

    for batch, i in enumerate(range(0, len(train_data) - 1, batch_size)):
        data, targets = get_batch(train_data, i, batch_size)
        src_mask = generate_square_subsequent_mask(len(data)).to(device)
        output = model(data, src_mask)
        loss = criterion(output[-output_window:], targets[-output_window:])
    
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

        total_loss += loss.item()
        if batch % log_interval == 0 and batch > 0:
            cur_loss = total_loss / log_interval
            ms_per_batch = (time.time() - start_time) * 1000 / log_interval
            ppl = math.exp(cur_loss)
            print(f'| epoch {epoch:3d} | {batch:5d}/{batch:5d} batches | '
                  f'lr {lr:02.2f} | ms/batch {ms_per_batch:5.2f} | '
                  f'loss {cur_loss:5.2f} | ppl {ppl:8.2f}')
            total_loss = 0
            start_time = time.time()

            
## Reference: https://github.com/oliverguhr/transformer-time-series-prediction
def evaluate(eval_model, data_source, require_plot = False):
    eval_model.eval() 
    total_loss = 0.
    test_result = torch.Tensor(0)    
    truth = torch.Tensor(0)
    with torch.no_grad():
        for i in range(0, data_source.size(0) - 1):
            data, target = get_batch(data_source, i, eval_batch_size)
            # look like the model returns static values for the output window
            src_mask = generate_square_subsequent_mask(data.size(0)).to(device) 
            output = eval_model(data, src_mask)    
            total_loss += criterion(output[-output_window:], target[-output_window:]).item()
            
            test_result = torch.cat((test_result, output[-1].view(-1).cpu()), 0)
            truth = torch.cat((truth, target[-1].view(-1).cpu()), 0)

    if require_plot:
        subplot[0].clear()
        subplot[0].plot(test_result,color="red")
        subplot[0].plot(truth,color="blue")
        subplot[0].grid(True, which='both')
        subplot[0].axhline(y=0, color='k')
        #plt.close()
    
    return total_loss / len(data_source)

def predict(eval_model, data_source, steps):
    eval_model.eval() 
    total_loss = 0.
    test_result = torch.Tensor(0)    
    truth = torch.Tensor(0)
    _ , data = get_batch(data_source, 0, 1)
    with torch.no_grad():
        for i in range(0, steps, 1):
            input = torch.clone(data[-input_window:])
            input[-output_window:] = 0
            src_mask = generate_square_subsequent_mask(data[-input_window:].size(0)).to(device)   
            output = eval_model(data[-input_window:], src_mask)                        
            data = torch.cat((data, output[-1:]))
    data = data.cpu().view(-1)

    subplot[1].clear()
    subplot[1].plot(data,color="red")
    subplot[1].plot(data[:input_window],color="blue")
    subplot[1].grid(True, which='both')
    subplot[1].axhline(y=0, color='k')
    #plt.savefig('graph/transformer-predict%d.png'%steps)
    #plt.close()

## Main loop
train_data, val_data = data_process()
fig, subplot = plt.subplots(2, 1)

# Reference: https://github.com/pytorch/tutorials/blob/master/beginner_source/transformer_tutorial.py
ntokens = 1  # size of data label
emsize = 250  # embedding dimension
d_hid = 2048  # dimension of the feedforward network model in nn.TransformerEncoder
nlayers = 1  # number of nn.TransformerEncoderLayer in nn.TransformerEncoder
nhead = 10  # number of heads in nn.MultiheadAttention
dropout = 0.1  # dropout probability
model = TransformerModel(ntokens, emsize, nhead, d_hid, nlayers, dropout).to(device)

#criterion = nn.CrossEntropyLoss()
criterion = nn.MSELoss()
lr = 0.00005

#optimizer = torch.optim.SGD(model.parameters(), lr=lr)
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.98)

best_val_loss = float("inf")
epochs = 100
best_model = None

for epoch in range(1, epochs + 1):
    epoch_start_time = time.time()
    train(train_data)

    if(epoch % 10 is 0):
        val_loss = evaluate(model, val_data, True)
        predict(model, val_data, 200)
    else:
        val_loss = evaluate(model, val_data)
        
        
    val_ppl = math.exp(val_loss)
    elapsed = time.time() - epoch_start_time
    print('-' * 89)
    print(f'| end of epoch {epoch:3d} | time: {elapsed:5.2f}s | '
          f'valid loss {val_loss:5.2f} | valid ppl {val_ppl:8.2f}')
    print('-' * 89)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model = model

    scheduler.step()
