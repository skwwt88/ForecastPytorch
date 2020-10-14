from config import *

import pandas as pd
import numpy as np
import torch

from sklearn.preprocessing import MinMaxScaler


def extract_features(input_file: str):
    df = pd.read_csv(input_file)
    
    values_list = []
    for i in range(0, F):
        values = df.iloc[:,i + 1 : i + 2].values
        sc = MinMaxScaler()
        values = sc.fit_transform(values)
        values_list.append(values)
    
    labels = df.iloc[:, 1: 5].values

    values = np.dstack(values_list)
    values = values.reshape((-1, F))

    return values, labels

def load_data(input_file: str):
    values, labels = extract_features(input_file)
    
    return scaling_window(values, labels, T)

def scaling_window(data, labels, seq_length):
    x = []
    y = []

    for i in range(len(data)-seq_length-1):
        _x = data[i:(i+seq_length)]
        _y = labels[i+seq_length, 0:4]
        x.append(_x)
        y.append(_y)

    return np.array(x),np.array(y)

def latest():
    values, _ = extract_features(data_file)
    index = len(values)
    return np.array([values[index - T:index]])

if __name__ == "__main__":
    inputs_x, inputs_y = load_data(data_file)
    print(inputs_x[:10])
    print(inputs_y[:10])
    print(latest())