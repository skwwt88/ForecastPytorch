import torch
from torch.optim import lr_scheduler
import numpy as np

from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

from data_utils import scaling_window
from utils import set_seed
from stock_info import stock_kline_day, qfq
from models import TimeSeriesModel_1Step

### Hyper-Parameter ###
seed = 1988
seq_length = 20
test_size = 0.2
feature_columns = ['open', 'high', 'low', 'close', 'volume']
predict_columns = ['high', 'low']
LATENT_DIM = 6
lr = 0.01
min_lr = 0.1e-8
patience = 200
max_epoch = 4000

### Init            ###
stock_id = 'sh600029'
device = torch.device("cuda:0")
set_seed(seed)

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
    return scaling_window(values, labels, seq_length)

def data_to_tensor(X_train, X_test, y_train, y_test):
    X_train = torch.tensor(X_train).float().to(device=device)
    X_test = torch.tensor(X_test).float().to(device=device)
    y_train = torch.tensor(y_train).float().to(device=device)
    y_test = torch.tensor(y_test).float().to(device=device)
    return X_train, X_test, y_train, y_test

def load_train_data():
    stock_df = stock_kline_day(stock_id, qfq)
    values, labels = process_features(stock_df)
    inputs_x, inputs_y = to_time_series(values, labels)
    X_train, X_test, y_train, y_test = train_test_split(inputs_x, inputs_y, test_size=test_size, random_state=seed)
    return data_to_tensor(X_train, X_test, y_train, y_test)

### Model       #########
def build_model():
    return TimeSeriesModel_1Step(len(predict_columns), len(feature_columns), LATENT_DIM, seq_length, device).to(device=device)

### Train       #########
def train(model):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.MSELoss()
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=patience, verbose=True, cooldown=1, min_lr=min_lr, eps=1e-05)

    X_train, X_test, y_train, y_test = load_train_data()
    pbar = tqdm(range(0, max_epoch))
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

        pbar.set_description("{0:.6f}, {1:.6f}".format(train_loss, validate_loss))
        scheduler.step(validate_loss)
    
    return model

### predict     ########
def predict(model):
    stock_df = stock_kline_day(stock_id, qfq)
    values, _ = process_features(stock_df)
    index = len(values)
    values = np.array([values[index - seq_length:index]])
    values = torch.tensor(values).float().to(device=device)

    with torch.no_grad():
        outputs = model(values)
        print(outputs)

if __name__ == "__main__":
    model = build_model()
    model = train(model)
    predict(model)

