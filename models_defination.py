
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
        
        self.es_normalize1 = ES_Normalize(alpha, [1,2])
        self.es_normalize2 = ES_Normalize(beta, [1,2])
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True)
        
        self.fc = nn.Linear(hidden_size, num_classes)
        self.es_denormalize1 = ES_DeNormalize(alpha)
        self.es_denormalize2 = ES_DeNormalize(beta)

    def forward(self, x):
        h_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device=self.device)
        
        x, st_11 = self.es_normalize1(x)
        x, st_12 = self.es_normalize2(x)
        # Propagate input through LSTM
        _, h_out = self.gru(x, h_0)
        h_out = h_out.view(-1, self.hidden_size)
        out = self.fc(h_out)
        out = out.view(-1, 1, self.num_classes)
        
        out = self.es_denormalize2(out, st_12)
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
    def __init__(self, num_classes, input_size, hidden_size_encoder, hidden_size_decoder, seq_length, steps, alpha, beta, device, num_layers = 1):
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
                            num_layers=num_layers, batch_first=True, dropout=0.2)
        self.decoder = nn.GRU(input_size=hidden_size_encoder * num_layers, hidden_size=hidden_size_decoder,
                            num_layers=num_layers, batch_first=True, dropout=0.2)        

        self.es_normalize1 = ES_Normalize(alpha, [1, 2])
        self.es_normalize2 = ES_Normalize(beta, [1,2])
        self.es_denormalize1 = ES_DeNormalize(alpha)
        self.es_denormalize2 = ES_DeNormalize(beta)
        
        self.fc_init_hidden_status = nn.Linear(hidden_size_encoder, hidden_size_decoder)
        self.fc_map_encode_decode = nn.Linear(hidden_size_encoder * num_layers, hidden_size_decoder)
        self.fc = nn.Linear(hidden_size_decoder, num_classes)

    def forward(self, x):
        h_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size_encoder).to(device=self.device)
        
        x, st_11 = self.es_normalize1(x)
        x, st_12 = self.es_normalize2(x)

        # Propagate input through LSTM
        _, h_out = self.encoder(x, h_0) 
        h_0 = self.fc_init_hidden_status(h_out)

        h_out = h_out.repeat((1, 1, self.steps))
        h_out = h_out.view(-1, self.steps, self.hidden_size_encoder * self.num_layers)
        
        h_out, _ = self.decoder(h_out, h_0) 
        h_out = h_out.reshape(-1, self.hidden_size_decoder)
        out = self.fc(h_out)
        out = out.view(-1, self.steps, self.num_classes)
        
        st_12 = torch.cat((st_12, out), 1)
        out = self.es_denormalize2(out, st_12)

        st_11 = torch.cat((st_11, out), 1)
        out = self.es_denormalize1(out, st_11)

        out = out.view(-1, self.steps * self.num_classes)

        return out

class ES_Normalize(nn.Module):
    def __init__(self, alpha, columns = None, pre_step = 0):
        super(ES_Normalize, self).__init__()
        
        self.alpha = alpha
        self.columns = columns
        self.pre_step = pre_step

    def forward(self, x):
        out = x.clone().detach()
        for i in range(1, out.shape[1]):
            out[:, i, :] = (1 - self.alpha) * out[:, i, :] + self.alpha * out[:, i - 1, :]

        return out, out[:, out.shape[1] - 1 - self.pre_step : out.shape[1], self.columns]

class ES_DeNormalize(nn.Module):
    def __init__(self, alpha):
        super(ES_DeNormalize, self).__init__()
        
        self.alpha = alpha

    def forward(self, x, st):
        for i in range(x.shape[1]):
            x[:, i, :] = (x[:, i, :] - self.alpha * st[:, i, :]) / (1 - self.alpha)
        
        return x