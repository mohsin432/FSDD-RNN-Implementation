# Copyright (C) 2026, Muhammad Mohsin Ghaffar, All rights reserved.
# Only for Job Application as Embedded AI Engineer Position at Innatera Nanosystems

import torch 
import matplotlib.pyplot as plt 
from models.rnnmodels import LSTMClassifier 


def disply_weights():
    MODEL_PATH = "output/power2_model.pth" 
    model = LSTMClassifier() 
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu")) 
    model.eval() 
    all_weights = [] 
    for name, param in model.named_parameters(): 
        if "weight" in name: 
            all_weights.extend(param.data.cpu().flatten().tolist()) 
            weights = torch.tensor(all_weights).abs() 
        
    unique_vals = torch.unique(weights) 
    print(unique_vals[:20])
    print(f"  Min absolute weight: {weights.min():.6f}")
    print(f"  Max absolute weight: {weights.max():.6f}")
    
    log2_weights = torch.log2(weights + 1e-8)
    
    plt.hist(log2_weights.numpy(), bins=50, alpha=0.7, edgecolor='black')

    min_exp = int(log2_weights.min().item())
    max_exp = int(log2_weights.max().item())
    ticks = list(range(min_exp, max_exp + 1))
    plt.xticks(ticks, rotation=45)
    plt.xlabel("log 2 weights")
    plt.ylabel("Number of weights")
    plt.grid(True, alpha=0.3)
    plt.savefig("output/log2_weights_distribution.png", dpi=150)
    plt.close()
    

if __name__ == "__main__":
    display()

