# Copyright (C) 2026, Muhammad Mohsin Ghaffar, All rights reserved.
# Only for Job Application as Embedded AI Engineer Position at Innatera Nanosystems
import torch

def evaluate(model, data_loader, device):
    model.eval()  # set to evaluation mode
    correct = 0
    total = 0

    with torch.no_grad():  # no gradients needed
        for x, y in data_loader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)  # shape: (batch, num_classes)
            _, predicted = torch.max(outputs, dim=1)  # get class with highest score
            total += y.size(0)
            correct += (predicted == y).sum().item()

    accuracy = 100 * correct / total
    return accuracy
