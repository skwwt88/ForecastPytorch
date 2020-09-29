import torch

from tqdm import tqdm
from models import TimeSeriesModel
from datasets import train_data_loaders


device = torch.device("cuda: 0")
model = TimeSeriesModel().to(device=device)
