import torch
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
import argparse
import pickle
import os
import random

from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

from data_utils import scaling_window_nstep, scaling_window
from pytorchtools import set_seed, save_model, clean_models, find_best_model_file, EarlyStopping
from stock_info import stock_kline_day, qfq
from models_defination import TimeSeriesModel_NSteps_v2
from config import *

### Hyper-Parameter ###
seed = 1989
seq_length = 20
test_size = 0.2
feature_columns = ['open', 'high', 'low', 'close', 'volume']
predict_columns = ['high', 'low']
lr = 0.00001
min_lr = 0.1e-11
patience = 100
max_epoch = 100000
encoder_settings = {}
encoder_settings['hidden_size'] = 64
encoder_settings['layer'] = 4
encoder_settings['dropout'] = 0.2

decoder_settings = {}
decoder_settings['hidden_size'] = 128
decoder_settings['layer'] = 2
decoder_settings['dropout'] = 0.2

### Init            ###
verbose = True
device = torch.device("cuda:0")
set_seed(seed)
model_name = 'multi_step_v2'
train_steps = 3
validate_ids = ['sh600115', 'sh600221', 'sz002928']
train_ids1 = [
    'sh600000',
]
train_ids = [
    'sh600000',
    'sh600004',
    'sh600009',
    'sh600010',
    'sh600011',
    'sh600015',
    'sh600016',
    'sh600018',
    'sh600019',
    'sh600025',
    'sh600027',
    'sh600028',
    'sh600029',
    'sh600030',
    'sh600031',
    'sh600036',
    'sh600038',
    'sh600048',
    'sh600050',
    'sh600061',
    'sh600066',
    'sh600068',
    'sh600085',
    'sh600089',
    'sh600377',
    'sh601021',
    'sh601111',
    'sh601333'
]

### Feature-Engineering
def process_features(stock_df):
    processors = {}
    values_list = []
    for column in feature_columns:
        values = stock_df.loc[:,[column]].values
        if not column in processors:
            processors[column] = MinMaxScaler()
        sc = processors[column]

        values = sc.fit_transform(values)
        values_list.append(values)

    values = np.dstack(values_list)
    values = values.reshape((-1, len(feature_columns)))

    labels_list = []
    for column in predict_columns:
        labels = stock_df.loc[:,[column]].values
        if not column in processors:
            processors[column] = MinMaxScaler()
        sc = processors[column]

        labels = sc.fit_transform(labels)
        labels_list.append(labels)

    labels = np.dstack(labels_list)
    labels = labels.reshape((-1, len(predict_columns)))

    return values, labels, processors

def to_time_series(values, labels):
    # return scaling_window(values, labels, seq_length)
    return scaling_window_nstep(values, labels, seq_length, train_steps)

def data_to_tensor(X_train, X_test, y_train, y_test):
    X_train = torch.tensor(X_train).float().to(device=device)
    X_test = torch.tensor(X_test).float().to(device=device)
    y_train = torch.tensor(y_train).float().to(device=device)
    y_test = torch.tensor(y_test).float().to(device=device)
    return X_train, X_test, y_train, y_test

def generate_dataset(stock_ids):
    x_list = []
    y_list = []

    for stock_id in stock_ids:
        stock_df = stock_kline_day(stock_id, qfq)
        values, labels, _ = process_features(stock_df)
        inputs_x, inputs_y = to_time_series(values, labels)
        x_list.append(inputs_x)
        y_list.append(inputs_y)

    c = list(zip(x_list, y_list))
    random.shuffle(c)
    x_list, y_list = zip(*c)

    x = np.concatenate(x_list, axis=0)
    y = np.concatenate(y_list, axis=0)

    return x, y

def load_train_data(stock_ids):
    X_train, y_train = generate_dataset(stock_ids)
    X_test, y_test = generate_dataset(validate_ids)

    return data_to_tensor(X_train, X_test, y_train, y_test)

### Model       #########
def build_model():
    return TimeSeriesModel_NSteps_v2(len(predict_columns), len(feature_columns), encoder_settings, decoder_settings, 0.2, 0, device).to(device=device)

### Train       #########
class BasicDataset(Dataset):
    def __init__(self, inputs_x: list, inputs_y: list):
        self.inputs_x = inputs_x
        self.inputs_y = inputs_y

    def __len__(self):
        return len(self.inputs_y)

    def __getitem__(self, index):
        x = self.inputs_x[index]
        y = self.inputs_y[index]

        return x, y

def train(model, stock_ids):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.MSELoss()
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=patience, verbose=verbose, cooldown=1, min_lr=min_lr, eps=min_lr)

    earlyStop = EarlyStopping(model_name, models_folder, patience=10)
    X_train, X_test, y_train, y_test = load_train_data(stock_ids)
    pbar = tqdm(range(0, max_epoch))
    clean_models(model_name, models_folder)

    for epoch in pbar:
        optimizer.zero_grad()
        model.train()
        # forward + backward + optimize
        steps = y_train.shape[1] // len(predict_columns)
        dataset = BasicDataset(X_train, y_train)
        dataloader = DataLoader(dataset, batch_size=10240, shuffle=True, num_workers=0)

        total_train_loss = []
        for _, items in enumerate(dataloader):
            train_outputs = model(items[0], steps)

            train_loss = criterion(train_outputs, items[1])
            train_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5, norm_type=2)
            optimizer.step()
            total_train_loss.append(train_loss)
           
        train_loss = torch.mean(torch.stack(total_train_loss))

        with torch.no_grad():
            model.eval()
            outputs = model(X_test, steps)
            validate_loss = criterion(outputs, y_test)

        if epoch % 100 == 99:
            earlyStop(validate_loss, model)
            if earlyStop.early_stop:
                break
        
        scheduler.step(train_loss)
        pbar.set_description("{0:.6f}, {1:.6f}".format(train_loss, validate_loss))
    
    return model

### predict     ########
def load_pre_trained_model(model):
    best_model_parameters = find_best_model_file(model_name, models_folder)
    model.load_state_dict(torch.load(best_model_parameters))

    return model

def predict(model, stock_id, predict_steps = train_steps, look_back_days = seq_length):
    model = load_pre_trained_model(model)
    model.eval()

    stock_df = stock_kline_day(stock_id, qfq)
    values, _, processors = process_features(stock_df)
    index = len(values)
    values = np.array([values[index - look_back_days:index]])
    values = torch.tensor(values).float().to(device=device)

    with torch.no_grad():
        outputs = model(values, predict_steps)
        outputs = pd.DataFrame(outputs.reshape(-1, 2).cpu().numpy(), columns=['high', 'low'])
        for column in outputs.columns:
            sc = processors[column]
            values = outputs.loc[:,[column]].values
            values = sc.inverse_transform(values)
            outputs[column] = pd.Series(values.reshape(-1))

        return outputs

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-train', dest='train_flag', action='store_true')
    parser.add_argument('-step', dest='step', type=int, default=train_steps)
    parser.add_argument('-look_back', dest='look_back', type=int, default=seq_length)
    parser.add_argument('-stockids', nargs='+')
    parser.set_defaults(train_flag=False)
    
    args = parser.parse_args()
    stockids = train_ids if not args.stockids else args.stockids

    result = {}
    model = build_model()
    #args.train_flag = True
    if args.train_flag:
        train(model, stockids)

    for stock_id in stockids:
        result = predict(model, stock_id, args.step, args.look_back)
        print(stock_id)
        print(result)

