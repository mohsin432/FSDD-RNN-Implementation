# Copyright (C) 2026, Muhammad Mohsin Ghaffar, All rights reserved.
# Only for Job Application as Embedded AI Engineer Position at Innatera Nanosystems

import torch
import torch.nn as nn


class LSTMClassifier(nn.Module):
    def __init__(self, input_size=64, hidden_size=64, num_layers=2, num_classes=10):
        super(LSTMClassifier, self).__init__()

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # x: (batch, time_steps, features)
        out, (hn, cn) = self.lstm(x)

        # Use last hidden state
        final_hidden = hn[-1]  # shape: (batch, hidden_size)

        out = self.fc(final_hidden)
        return out

class GRUClassifier(nn.Module):
    def __init__(self, input_size=64, hidden_size=64, num_layers=2, num_classes=10, dropout=0.3):
        super(GRUClassifier, self).__init__()
        
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.fc = nn.Linear(hidden_size, num_classes)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        out, hn = self.gru(x)
        final_hidden = hn[-1]
        out = self.dropout(final_hidden)
        out = self.fc(out)
        return out
        
        
        
class LSTM36kBClassifier(nn.Module):
    def __init__(self, input_size=64, hidden_size=25, num_layers=2, num_classes=10):
        super(LSTM36kBClassifier, self).__init__()

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # x: (batch, time_steps, features)
        out, (hn, cn) = self.lstm(x)

        # Use last hidden state
        final_hidden = hn[-1]  # shape: (batch, hidden_size)

        out = self.fc(final_hidden)
        return out
        

