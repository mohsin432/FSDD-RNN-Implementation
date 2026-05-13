# Copyright (C) 2026, Muhammad Mohsin Ghaffar, All rights reserved.
# Only for Job Application as Embedded AI Engineer Position at Innatera Nanosystems
import sys
from train import train
from test import test
from models.QuantizeLSTM import dynamic_quantize_lstm
from utils.MemoryCalc import memory_calc
from utils.DisplayWeights import disply_weights


def main(): 
    if len(sys.argv) < 2:
        print("Usage: python main.py [TaskA | TaskB | TaskC]")
        sys.exit(1)
        
    task = sys.argv[1]
    if task == "TaskA":
        print("Running TaskA: Model Training on Train and Validate Dataset")
        train(model_train="lstm")
        
        print("Testing on Test Dataset")
        test(model="lstm")
    
    elif task == "TaskB":
        print("Original Model Memory Footprint")
        memory_calc(model="lstm")
        
        print("Training with reduced sized model")
        train(model_train="lstm36K")
        
        print("36KB Model Memory Constraints")
        memory_calc(model="lstm36K")
        
        print("Dynamic Int8 Quantization of LSTM Model")
        dynamic_quantize_lstm()
        
        print("36KB Model Memory Constraints")
        memory_calc(model="qlstm")
        print("After Int8 Quantization original FP model achieved the 36KB Memory/Layer Constraint")
    
    elif task == "TaskC":
        print("Running TaskC: Weights forced to power of 2")
        train(model_train="pow2lstm")
        print("Verifying the Weights")
        disply_weights()
    
    else:
        print(f"Unknown task: {task}")
        print("Valid options: TaskA, TaskB, TaskC")
        sys.exit(1)

if __name__ == "__main__":
    main()
    
