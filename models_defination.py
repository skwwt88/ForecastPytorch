
import torch.nn as nn
import torch.nn.functional as F
import torch

class TimeSeriesModel_1Step(nn.Module):
    def __init__(self, num_classes, input_size, hidden_size, seq_length, device, num_layers = 1):
        super(TimeSeriesModel_1Step, self).__init__()
        
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.seq_length = seq_length
        self.device = device
        
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True)
        
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        h_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device=self.device)
        
        # Propagate input through LSTM
        _, h_out = self.gru(x, h_0)
        h_out = h_out.view(-1, self.hidden_size)
        out = self.fc(h_out)
        
        return out

class TimeSeriesModel_NStep(nn.Module):
    def __init__(self, num_classes, input_size, hidden_size, seq_length, steps, device, num_layers = 1):
        super(TimeSeriesModel_NStep, self).__init__()
        
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.seq_length = seq_length
        self.device = device
        self.steps = steps
        
        self.encoder = nn.GRU(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True)
        self.decoder = nn.GRU(input_size=hidden_size, hidden_size=hidden_size * 2,
                            num_layers=num_layers, batch_first=True)
        
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        h_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device=self.device)
        
        # Propagate input through LSTM
        _, h_out = self.encoder(x, h_0) 
        h_out = h_out.repeat((1, 1, self.steps))
        h_out = h_out.view(-1, self.steps, self.hidden_size)
        h_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size * 2).to(device=self.device)
        h_out, _ = self.decoder(h_out, h_0) 
        h_out = h_out.reshape(-1, self.hidden_size * 2)
        out = self.fc(h_out)
        out = out.reshape(-1, self.num_classes * self.steps)
        
        return out