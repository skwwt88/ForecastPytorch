import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from feature_process import load_data
from config import data_file, seed, BatchSize

class TimeSeriesDataset(Dataset):
    def __init__(self, inputs_x: list, inputs_y: list):
        self.inputs_x = inputs_x
        self.inputs_y = inputs_y

    def __len__(self):
        return len(self.inputs_y)

    def __getitem__(self, index):
        x = self.inputs_x[index]
        y = self.inputs_y[index]

        return x, y

def train_data():
    inputs_x, inputs_y = load_data(data_file)
    return train_test_split(inputs_x, inputs_y, test_size=0.2, random_state=seed)

def train_data_loaders(X_train, X_test, y_train, y_test):
    trainloader = torch.utils.data.DataLoader(TimeSeriesDataset(X_train, y_train), batch_size=BatchSize, shuffle=True, num_workers = 8)
    validateloader = torch.utils.data.DataLoader(TimeSeriesDataset(X_test, y_test), batch_size=BatchSize, shuffle=False, num_workers = 8)
    return trainloader, validateloader

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = train_data()
    print(X_train)

