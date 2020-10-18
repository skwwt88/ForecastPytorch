import torch
from torch.optim import lr_scheduler
import numpy as np
import pandas as pd
import argparse

from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

from data_utils import scaling_window_nstep, scaling_window
from pytorchtools import set_seed, save_model, clean_models, find_best_model_file, EarlyStopping
from stock_info import stock_kline_day, qfq
from models_defination import TimeSeriesModel_NStep
from config import *

### Hyper-Parameter ###
seed = 1988
seq_length = 50
test_size = 0.2
feature_columns = ['open', 'high', 'low', 'close', 'volume']
predict_columns = ['high', 'low']
LATENT_DIM = 8
lr = 0.01
min_lr = 0.1e-9
patience = 280
max_epoch = 100000

### Init            ###
verbose = False
device = torch.device("cuda:0")
set_seed(seed)
n_steps = 15
model_name = 'multi_{}_step'.format(n_steps)

### Feature-Engineering
def process_features(stock_df):
    values_list = []
    for column in feature_columns:
        values = stock_df.loc[:,[column]].values
        sc = MinMaxScaler()
        values = sc.fit_transform(values)
        values_list.append(values)

    values = np.dstack(values_list)
    values = values.reshape((-1, len(feature_columns)))

    labels = stock_df.loc[:, predict_columns].values

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

def load_train_data(stock_id):
    stock_df = stock_kline_day(stock_id, qfq)
    values, labels = process_features(stock_df)
    inputs_x, inputs_y = to_time_series(values, labels)
    X_train, X_test, y_train, y_test = train_test_split(inputs_x, inputs_y, test_size=test_size, random_state=seed)
    return data_to_tensor(X_train, X_test, y_train, y_test)

### Model       #########
def build_model():
    return TimeSeriesModel_NStep(len(predict_columns), len(feature_columns), LATENT_DIM, seq_length, n_steps, device).to(device=device)

def get_model_name(stock_id):
    return "{}_{}".format(model_name, stock_id)

### Train       #########
def train(model, stock_id):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.MSELoss()
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=patience, verbose=verbose, cooldown=1, min_lr=min_lr, eps=1e-05)

    earlyStop = EarlyStopping(get_model_name(stock_id), models_folder, patience=4)
    X_train, X_test, y_train, y_test = load_train_data(stock_id)
    pbar = tqdm(range(0, max_epoch))
    clean_models(get_model_name(stock_id), models_folder)

    for epoch in pbar:
        optimizer.zero_grad()

        # forward + backward + optimize
        train_outputs = model(X_train)
        train_loss = criterion(train_outputs, y_train)
        train_loss.backward()
        optimizer.step()

        with torch.no_grad():
            outputs = model(X_test)
            validate_loss = criterion(outputs, y_test)

        if epoch % 300 == 299:
            earlyStop(validate_loss, model)
            if earlyStop.early_stop:
                break

        pbar.set_description("{0}:{1:.6f}, {2:.6f}".format(stock_id, train_loss, validate_loss))
        scheduler.step(validate_loss)
    
    return model

### predict     ########
def load_pre_trained_model(model, stock_id):
    best_model_parameters = find_best_model_file(get_model_name(stock_id), models_folder)
    model.load_state_dict(torch.load(best_model_parameters))

    return model

def predict(model, stock_id):
    stock_df = stock_kline_day(stock_id, qfq)
    values, _ = process_features(stock_df)
    index = len(values)
    values = np.array([values[index - seq_length:index]])
    values = torch.tensor(values).float().to(device=device)

    with torch.no_grad():
        outputs = model(values)
        return pd.DataFrame(outputs.reshape(-1, 2).cpu().numpy(), columns=['high', 'low'])

def print_predict(result):
    pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-train', dest='train_flag', action='store_true')
    parser.add_argument('-stockids', nargs='+')
    parser.set_defaults(train_flag=False)
    
    args = parser.parse_args()
    stockids = default_stock_ids if not args.stockids else args.stockids

    result = {}
    for stock_id in stockids:
        model = build_model()
        if args.train_flag:
            model = train(model, stock_id)
        else:
            try:
                model = load_pre_trained_model(model, stock_id)
            except:
                model = train(model, stock_id)

        stock_predict = predict(model, stock_id)
        result[stock_id] = stock_predict

    for stock, price in result.items():
        print(stock)
        print("{}".format(price))
