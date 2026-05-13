# Copyright (C) 2026, Muhammad Mohsin Ghaffar, All rights reserved.
# Only for Job Application as Embedded AI Engineer Position at Innatera Nanosystems
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from torch.utils.data import Dataset



class SpokenMNISTDataset(Dataset):
    def __init__(self, X_path, y_path):
        self.X = np.load(X_path)
        self.y = np.load(y_path)

        # Convert to torch tensors
        self.X = torch.tensor(self.X, dtype=torch.float32)
        self.y = torch.tensor(self.y, dtype=torch.long).squeeze()
        
        #print("Shape before processing:", self.X.shape)
        if self.X.dim() == 4:
            N, T, F, C = self.X.shape
            #self.X = self.X.view(N, T, F * C)
            self.X = self.X[:, :, :, 0] # remove the channel data because of MatPlotLib function
            self.X = (self.X - self.X.mean(dim=0, keepdim=True)) / (self.X.std(dim=0, keepdim=True) + 1e-5)   
        #print(f"Shape after processing: {self.X.shape}")

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
