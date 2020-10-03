
import torch.nn as nn
import torch.nn.functional as F
import torch

from config import F, LATENT_DIM, T

class GRUModel(nn.Module):
    def __init__(self, num_classes, input_size, hidden_size, num_layers):
        super(GRUModel, self).__init__()
        
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.seq_length = T
        
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True)
        
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        h_0 = torch.zeros(
            self.num_layers, x.size(0), self.hidden_size).cuda()
        
        # Propagate input through LSTM
        ula, h_out = self.gru(x, h_0)
        h_out = h_out.view(-1, self.hidden_size)
        out = self.fc(h_out)
        
        return out