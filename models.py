
import torch.nn as nn
import torch.nn.functional as F
import torch

from config import F, LATENT_DIM

class TimeSeriesModel(nn.Module):
    def __init__(self):
        self.rnn = nn.RNN(F, LATENT_DIM, 1)
        self.out = nn.Linear(LATENT_DIM, 1)
        
    def forward(self, x, h):
        x, h = self.rnn(x, h)
        x = self.out(x)
        return x