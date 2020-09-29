from config import *

import pandas as pd
import numpy as np
import torch


def load_data(input_file: str):
    df = pd.read_csv(input_file)  
    return transform_to_time_series(df)

def transform_to_time_series(df: pd.DataFrame):
    inputs_x = []
    inputs_y = []
    for index in range(0, len(df)):
        feature_matrix = np.empty((T, F))
        feature_matrix[:] = np.nan

        for t in range(1, T + 1):
            if index - t < 0:
                break

            feature_matrix[t - 1, 1] = df.iloc[index - t]['temp']
            feature_matrix[t - 1, 0] = df.iloc[index - t]['load']

        if (True in np.isnan(feature_matrix)):
            continue
                        
        feature_matrix = torch.tensor(feature_matrix)

        inputs_x.append(df.iloc[index]['load'])
        inputs_y.append(feature_matrix)

    return inputs_x, inputs_y

if __name__ == "__main__":
    inputs_x, inputs_y = load_data('data/energy.csv')
    print(inputs_x[:10])
    print(inputs_y[:10])