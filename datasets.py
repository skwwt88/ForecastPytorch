import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from feature_process import load_data
from config import data_file, seed

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

def train_data_loaders():
    inputs_x, inputs_y = load_data(data_file)
    X_train, X_test, y_train, y_test = train_test_split(inputs_x, inputs_y, test_size=0.2, random_state=seed)
    trainloader = torch.utils.data.DataLoader(TimeSeriesDataset(X_train, y_train), batch_size=16, shuffle=True, num_workers = 8)
    validateloader = torch.utils.data.DataLoader(TimeSeriesDataset(X_test, y_test), batch_size=16, shuffle=False, num_workers = 8)
    return trainloader, validateloader

if __name__ == "__main__":
    import pandas as pd
    
    from feature_process import transform_to_time_series
    df = pd.read_csv(data_file)  
    inputs_x, inputs_y = transform_to_time_series(df[0:10])
    dataset = TimeSeriesDataset(inputs_x, inputs_y)
    print(dataset[3])
    print(len(dataset))

    train_data_loaders()

