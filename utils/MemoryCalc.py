# Copyright (C) 2026, Muhammad Mohsin Ghaffar, All rights reserved.
# Only for Job Application as Embedded AI Engineer Position at Innatera Nanosystems

import torch
import torch.nn as nn
from collections import defaultdict
from models.rnnmodels import LSTMClassifier, GRUClassifier, LSTM36kBClassifier


def load_model(model_path, model):
    if model=="lstm" or model == "qlstm": 
        model = LSTMClassifier()
    elif model == "lstm36K":
        model = LSTM36kBClassifier()
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    return model


def count_params_per_layer(model):
    layer_params = defaultdict(int)
    for name, param in model.named_parameters():
        if "lstm" in name:
            layer_num = name.split('_l')[-1][0]
            layer_key = f"lstm_layer_{layer_num}"
        elif "fc" in name:
            layer_key = "fc_layer"
        else:
            layer_key = "other"

        layer_params[layer_key] += param.numel()

    return layer_params


def print_layerwise_memory(model, bytes_per_param=4, title="Model"):
    layer_params = count_params_per_layer(model)

    for layer, params in layer_params.items():
        memory_kb = (params * bytes_per_param) / 1024
        print(f"{layer}:")
        print(f"  Params: {params}")
        print(f"  Memory: {memory_kb:.2f} KB")
        print()

def memory_calc(model="lstm"): 
    if model=="lstm": 
        model_path = "output/best_modelFP.pth"
    elif model == "lstm36K":
        model_path = "output/best_model.pth"
    elif model == "qlstm":
        model_path = "output/best_modelFP.pth"

    model_load = load_model(model_path, model)

    if model=="lstm" or model == "lstm36K" : 
        print_layerwise_memory(model_load, bytes_per_param=4, title="FP32 Model")
    elif model == "qlstm":
        print_layerwise_memory(model_load, bytes_per_param=1, title="Int8 Model")
