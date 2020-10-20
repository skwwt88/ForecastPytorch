import torch
from torch.optim import lr_scheduler
import numpy as np
import argparse
import os
import pickle
import pandas as pd

from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

from data_utils import scaling_window
from pytorchtools import set_seed, save_model, clean_models, find_best_model_file, EarlyStopping, model_exist
from stock_info import stock_kline_day, qfq
from models_defination import TimeSeriesModel_1Step_ES
from config import *

### Hyper-Parameter ###
seed = 1988
seq_length = 20
test_size = 0.2
feature_columns = ['open', 'high', 'low', 'close', 'volume']
predict_columns = ['high', 'low']
LATENT_DIM = 12
lr = 0.001
min_lr = 0.1e-8
patience = 200
max_epoch = 20000

### Init            ###
verbose = False
device = torch.device("cuda:0")
set_seed(seed)
model_name = 'single_step_es'

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
    return scaling_window(values, labels, seq_length)

def save_data_context(data_context, stock_id):
    file_name = "data_context_{0}_{1}.pkl".format(model_name, stock_id)
    if not os.path.exists(models_folder):
        os.makedirs(models_folder)
    file_name = os.path.join(models_folder, file_name)
        
    pickle.dump(data_context, open(file_name, 'wb'))

def load_data_context(stock_id):
    file_name = "data_context_{0}_{1}.pkl".format(model_name, stock_id)
    file_name = os.path.join(models_folder, file_name)
    return pickle.load(open(file_name, 'rb'))

def data_to_tensor(X_train, X_test, y_train, y_test):
    X_train = torch.tensor(X_train).float().to(device=device)
    X_test = torch.tensor(X_test).float().to(device=device)
    y_train = torch.tensor(y_train).float().to(device=device)
    y_test = torch.tensor(y_test).float().to(device=device)
    return X_train, X_test, y_train, y_test

def load_train_data(stock_id):
    data_context = {}
    stock_df = stock_kline_day(stock_id, qfq)
    values, labels = process_features(stock_df, data_context)
    inputs_x, inputs_y = to_time_series(values, labels)
    X_train, X_test, y_train, y_test = train_test_split(inputs_x, inputs_y, test_size=test_size, random_state=seed)

    save_data_context(data_context, stock_id)
    return data_to_tensor(X_train, X_test, y_train, y_test)

### Model       #########
def build_model():
    return TimeSeriesModel_1Step_ES(len(predict_columns), len(feature_columns), LATENT_DIM, seq_length, device, alpha=0.2, beta=0.1).to(device=device)

def get_model_name(stock_id):
    return "{}_{}".format(model_name, stock_id)

### Train       #########
def train(model, stock_id):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.MSELoss()
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=patience, verbose=verbose, cooldown=1, min_lr=min_lr, eps=1e-05)

    X_train, X_test, y_train, y_test = load_train_data(stock_id)
    pbar = tqdm(range(0, max_epoch))
    earlyStop = EarlyStopping(get_model_name(stock_id), models_folder, patience=4, delta = 0.00001)
    clean_models(get_model_name(stock_id), models_folder)

    for epoch in pbar:
        optimizer.zero_grad()

        # forward + backward + optimize
        train_outputs = model(X_train)
        train_loss = criterion(train_outputs, y_train)
        train_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10, norm_type=2)
        optimizer.step()

        with torch.no_grad():
            outputs = model(X_test)
            validate_loss = criterion(outputs, y_test)

        if epoch % 100 == 99:
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
    model = load_pre_trained_model(model, stock_id)
    data_context = load_data_context(stock_id)

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
            sc = data_context[column]
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
    for stock_id in stockids:
        model = build_model()
        if args.train_flag or not model_exist(get_model_name(stock_id), models_folder):
            train(model, stock_id)

        stock_predict = predict(model, stock_id)
        print(stock_id)
        print(stock_predict)
