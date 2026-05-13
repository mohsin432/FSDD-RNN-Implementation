# Copyright (C) 2026, Muhammad Mohsin Ghaffar, All rights reserved.
# Only for Job Application as Embedded AI Engineer Position at Innatera Nanosystems
import torch
import warnings
import torch.nn as nn
from collections import defaultdict
from models.rnnmodels import LSTMClassifier

warnings.filterwarnings("ignore", category=DeprecationWarning)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def quantize_model(model):
    model.eval()
    quantized_model = torch.quantization.quantize_dynamic(model,{nn.LSTM, nn.Linear},dtype=torch.qint8)
    return quantized_model

def dynamic_quantize_lstm():
    
    model_path = "output/best_modelFP.pth"  
    model = LSTMClassifier().to(device)
    model.load_state_dict(torch.load(model_path))
    print("Quantizing model to Int8")
    quantized_model = quantize_model(model)
    
    print("Model Quantized to Int8")
    for name, module in quantized_model.named_modules():
        if hasattr(module, 'dtype'):
            print(f"{name}: dtype={module.dtype}")
        if hasattr(module, 'weight') and hasattr(module.weight, 'dtype'):
            print(f"weight dtype={module.weight.dtype}")
    
 
