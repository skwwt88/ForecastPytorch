
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

class TimeSeriesModel_1Step_ES(nn.Module):
    def __init__(self, num_classes, input_size, hidden_size, seq_length, device, alpha, beta, num_layers = 1):
        super(TimeSeriesModel_1Step_ES, self).__init__()
        
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.seq_length = seq_length
        self.device = device
        
        self.es_normalize1 = ES_Normalize(alpha)
        self.es_normalize2 = ES_Normalize(beta)
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True)
        
        self.fc = nn.Linear(hidden_size, num_classes)
        self.es_denormalize1 = ES_DeNormalize(alpha, [1,2])
        self.es_denormalize2 = ES_DeNormalize(beta, [1,2])

    def forward(self, x):
        h_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device=self.device)
        
        x, st_11 = self.es_normalize1(x)
        #x, st_12 = self.es_normalize2(x)
        # Propagate input through LSTM
        _, h_out = self.gru(x, h_0)
        h_out = h_out.view(-1, self.hidden_size)
        out = self.fc(h_out)
        
        #out = self.es_denormalize2(out, st_12)
        out = self.es_denormalize1(out, st_11)
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

class TimeSeriesModel_NStep_Combined(nn.Module):
    def __init__(self, num_classes, input_size, hidden_size_encoder, hidden_size_decoder, seq_length, steps, device, num_layers = 1):
        super(TimeSeriesModel_NStep_Combined, self).__init__()
        
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size_encoder = hidden_size_encoder
        self.hidden_size_decoder = hidden_size_decoder
        self.seq_length = seq_length
        self.device = device
        self.steps = steps
        
        self.encoder = nn.GRU(input_size=input_size, hidden_size=hidden_size_encoder,
                            num_layers=num_layers, batch_first=True)
        self.decoder = nn.GRU(input_size=hidden_size_encoder, hidden_size=hidden_size_decoder,
                            num_layers=num_layers, batch_first=True)
        
        self.fc = nn.Linear(hidden_size_decoder, num_classes)

    def forward(self, x):
        h_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size_encoder).to(device=self.device)
        
        # Propagate input through LSTM
        _, h_out = self.encoder(x, h_0) 
        h_out = h_out.repeat((1, 1, self.steps))
        h_out = h_out.view(-1, self.steps, self.hidden_size_encoder)
        h_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size_decoder).to(device=self.device)
        h_out, _ = self.decoder(h_out, h_0) 
        h_out = h_out.reshape(-1, self.hidden_size_decoder)
        out = self.fc(h_out)
        out = out.reshape(-1, self.num_classes * self.steps)
        
        return out

class ES_Normalize(nn.Module):
    def __init__(self, alpha):
        super(ES_Normalize, self).__init__()
        
        self.alpha = alpha

    def forward(self, x):
        
        for i in range(1, x.shape[1]):
            x[:, i, :] = (1 - self.alpha) * x[:, i, :] + self.alpha * x[:, i - 1, :]
        
        return x, x[:, -1, :]

class ES_DeNormalize(nn.Module):
    def __init__(self, alpha, columns):
        super(ES_DeNormalize, self).__init__()
        
        self.alpha = alpha
        self.columns = columns

    def forward(self, x, st_1):
        
        x = (x - self.alpha * st_1[:, self.columns]) / (1 - self.alpha)
        
        return x