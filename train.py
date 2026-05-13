# Copyright (C) 2026, Muhammad Mohsin Ghaffar, All rights reserved.
# Only for Job Application as Embedded AI Engineer Position at Innatera Nanosystems

import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from models.rnnmodels import LSTMClassifier, GRUClassifier, LSTM36kBClassifier
from dataprocessing.SpokenMNISTData import SpokenMNISTDataset
from utils.Evaluate import evaluate
from models.Power2LSTM import initialize_power_of_two, apply_power_of_two_constraint 
from torch.utils.data import DataLoader


def train(model_train="lstm"): 
    # Check device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Hyperparameters
    num_epochs = 50
    batch_size=32
    lr=0.0005
    criterion = nn.CrossEntropyLoss()
    
    save_path = "output/best_model.pth"

    #Load Model
    if model_train=="lstm": 
        model = LSTMClassifier().to(device)
    elif model_train == "lstm36K":
        model = LSTM36kBClassifier().to(device)
    elif model_train == "pow2lstm":
        model = LSTM36kBClassifier().to(device)
        initialize_power_of_two(model)
    else: 
        print(f"Unknown Model")
    #model = GRUClassifier().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # Store Best Model for testing 
    
    # Load Train and Validate Data
    train_dataset = SpokenMNISTDataset("dataset/X_train.npy", "dataset/y_train.npy")
    val_dataset   = SpokenMNISTDataset("dataset/X_val.npy", "dataset/y_val.npy")


    train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size)

    #scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            if model_train == "pow2lstm":
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            if model_train == "pow2lstm":
                apply_power_of_two_constraint(model)
            total_loss += loss.item()
        
        #scheduler.step()  
        
        if (epoch + 1) % 10 == 0:
            val_accuracy = evaluate(model, val_loader, device)
            best_acc=0.0
            if val_accuracy > best_acc:
                torch.save(model.state_dict(), save_path)
            best_acc = val_accuracy
            print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%")
        else:
            print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")



