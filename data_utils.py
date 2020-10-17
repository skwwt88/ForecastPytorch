import numpy as np
import torch

from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

def scaling_window(data, labels, seq_length):
    x = []
    y = []

    for i in range(len(data)-seq_length-1):
        _x = data[i:(i+seq_length)]
        _y = labels[i+seq_length, :]
        x.append(_x)
        y.append(_y)

    return np.array(x), np.array(y)

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