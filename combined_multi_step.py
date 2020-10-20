import torch
from torch.optim import lr_scheduler
import numpy as np
import pandas as pd
import argparse
import pickle
import os

from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

from data_utils import scaling_window_nstep, scaling_window
from pytorchtools import set_seed, save_model, clean_models, find_best_model_file, EarlyStopping
from stock_info import stock_kline_day, qfq
from models_defination import TimeSeriesModel_NStep_Combined
from config import *

### Hyper-Parameter ###
seed = 1989
seq_length = 30
test_size = 0.2
feature_columns = ['open', 'high', 'low', 'close', 'volume']
predict_columns = ['high', 'low']
encoder_output = 8
decoder_output = 16
lr = 0.01
min_lr = 0.1e-8
patience = 100
max_epoch = 100000

### Init            ###
verbose = True
device = torch.device("cuda:0")
set_seed(seed)
n_steps = 3
model_name = 'combined_multi_{0}_step'.format(n_steps)

### Feature-Engineering
def process_features(stock_df, processors):
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

    return values, labels

def to_time_series(values, labels):
    # return scaling_window(values, labels, seq_length)
    return scaling_window_nstep(values, labels, seq_length, n_steps)

def data_to_tensor(X_train, X_test, y_train, y_test):
    X_train = torch.tensor(X_train).float().to(device=device)
    X_test = torch.tensor(X_test).float().to(device=device)
    y_train = torch.tensor(y_train).float().to(device=device)
    y_test = torch.tensor(y_test).float().to(device=device)
    return X_train, X_test, y_train, y_test

def save_data_context(data_context):
    file_name = "data_context_{0}.pkl".format(model_name)
    if not os.path.exists(models_folder):
        os.makedirs(models_folder)
    file_name = os.path.join(models_folder, file_name)
        
    pickle.dump(data_context, open(file_name, 'wb'))

def load_data_context():
    file_name = "data_context_{0}.pkl".format(model_name)
    file_name = os.path.join(models_folder, file_name)
    return pickle.load(open(file_name, 'rb'))

def load_train_data(stock_ids):
    inputs_x_list = []
    inputs_y_list = []
    data_context = {}
    for stock_id in stock_ids:
        if not stock_id in data_context:
            data_context[stock_id] = {}

        stock_df = stock_kline_day(stock_id, qfq)
        values, labels = process_features(stock_df, data_context[stock_id])
        inputs_x, inputs_y = to_time_series(values, labels)
        inputs_x_list.append(inputs_x)
        inputs_y_list.append(inputs_y)

    save_data_context(data_context)
    input_x = np.concatenate(inputs_x_list, axis=0)
    input_y = np.concatenate(inputs_y_list, axis=0)

    X_train, X_test, y_train, y_test = train_test_split(input_x, input_y, test_size=test_size, random_state=seed)

    return data_to_tensor(X_train, X_test, y_train, y_test)

### Model       #########
def build_model():
    return TimeSeriesModel_NStep_Combined(len(predict_columns), len(feature_columns), encoder_output, decoder_output, seq_length, n_steps, 0.2, 0.1, device).to(device=device)

def get_model_name(stock_id):
    return "{}_{}".format(model_name, stock_id)

### Train       #########
def train(model, stock_ids):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.MSELoss()
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=patience, verbose=verbose, cooldown=1, min_lr=min_lr, eps=1e-05)

    earlyStop = EarlyStopping(model_name, models_folder, patience=3)
    X_train, X_test, y_train, y_test = load_train_data(stock_ids)
    pbar = tqdm(range(0, max_epoch))
    clean_models(model_name, models_folder)

    for epoch in pbar:
        optimizer.zero_grad()

        # forward + backward + optimize
        train_outputs = model(X_train)

        train_loss = criterion(train_outputs, y_train)
        train_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5, norm_type=2)
        optimizer.step()

        with torch.no_grad():
            outputs = model(X_test)
            validate_loss = criterion(outputs, y_test)

        if epoch % 300 == 299:
            earlyStop(validate_loss, model)
            if earlyStop.early_stop:
                break

        pbar.set_description("{0:.6f}, {1:.6f}".format(train_loss, validate_loss))
        scheduler.step(validate_loss)
    
    return model

### predict     ########
def load_pre_trained_model(model):
    best_model_parameters = find_best_model_file(model_name, models_folder)
    model.load_state_dict(torch.load(best_model_parameters))

    return model

def predict(model, stock_id):
    data_context = load_data_context()
    model = load_pre_trained_model(model)
    model.eval()

    stock_df = stock_kline_day(stock_id, qfq)
    values, _ = process_features(stock_df, data_context)
    index = len(values)
    values = np.array([values[index - seq_length:index]])
    values = torch.tensor(values).float().to(device=device)

    with torch.no_grad():
        outputs = model(values)
        outputs = pd.DataFrame(outputs.reshape(-1, 2).cpu().numpy(), columns=['high', 'low'])
        for column in outputs.columns:
            sc = data_context[stock_id][column]
            values = outputs.loc[:,[column]].values
            values = sc.inverse_transform(values)
            outputs[column] = pd.Series(values.reshape(-1))

        return outputs

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-train', dest='train_flag', action='store_true')
    parser.add_argument('-stockids', nargs='+')
    parser.set_defaults(train_flag=False)
    
    args = parser.parse_args()
    stockids = default_stock_ids if not args.stockids else args.stockids

    result = {}
    model = build_model()
    args.train_flag = True
    if args.train_flag:
        train(model, stockids)

    for stock_id in stockids:
        result = predict(model, stock_id)
        print(stock_id)
        print(result)

