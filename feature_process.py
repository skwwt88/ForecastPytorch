from config import *

import pandas as pd
import numpy as np
import torch

from sklearn.preprocessing import MinMaxScaler


def load_data(input_file: str):
    df = pd.read_csv(input_file)
    
    values_1 = df.iloc[:,1:2].values
    sc_1 = MinMaxScaler()
    values_1 = sc_1.fit_transform(values_1)

    values_2 = df.iloc[:,2:3].values
    sc_2 = MinMaxScaler()
    values_2 = sc_2.fit_transform(values_2)

    values = np.dstack((values_1, values_2))
    values = values.reshape((-1, 2))
    
    return scaling_window(values, T)

def scaling_window(data, seq_length):
    x = []
    y = []

    for i in range(len(data)-seq_length-1):
        _x = data[i:(i+seq_length)]
        _y = data[i+seq_length, 0:1]
        x.append(_x)
        y.append(_y)

    return np.array(x),np.array(y)

if __name__ == "__main__":
    inputs_x, inputs_y = load_data('data/energy.csv')
    print(inputs_x[:10])
    print(inputs_y[:10])