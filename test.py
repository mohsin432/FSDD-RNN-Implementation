# Copyright (C) 2026, Muhammad Mohsin Ghaffar, All rights reserved.
# Only for Job Application as Embedded AI Engineer Position at Innatera Nanosystems
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from models.rnnmodels import LSTMClassifier, GRUClassifier
from dataprocessing.SpokenMNISTData import SpokenMNISTDataset
from utils.Evaluate import evaluate
from torch.utils.data import DataLoader

def test(model="lstm"): 
    # Check device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    test_dataset  = SpokenMNISTDataset("dataset/X_test.npy", "dataset/y_test.npy")
    test_loader  = DataLoader(test_dataset, batch_size=32)

    if model=="lstm": 
        model = LSTMClassifier().to(device)
    else: 
        print(f"Unknown Model")

    def test_model(model, test_loader, device, model_path=None):
        if model_path is not None:
            model.load_state_dict(torch.load(model_path))
            model.to(device)
            model.eval()
            print(f"Loaded model from {model_path}")

        test_acc = evaluate(model, test_loader, device)
        print(f"Test Accuracy: {test_acc:.2f}%")
        return test_acc


    test_accuracy = test_model(model, test_loader, device, model_path="output/best_model.pth")
