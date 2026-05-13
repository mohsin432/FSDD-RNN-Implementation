# Copyright (C) 2026, Muhammad Mohsin Ghaffar, All rights reserved.
# Only for Job Application as Embedded AI Engineer Position at Innatera Nanosystems

import torch
import torch.nn as nn

def project_to_power_of_two(tensor, min_exp=-8, max_exp=8):
    sign = tensor.sign()
    abs_tensor = tensor.abs() + 1e-8

    log2_val = torch.log2(abs_tensor)
    log2_val = torch.clamp(log2_val, min_exp, max_exp)

    rounded = torch.round(log2_val)

    return sign * (2 ** rounded)


def apply_power_of_two_constraint(model):
    with torch.no_grad():
        for name, param in model.named_parameters():
            if "weight" in name:
                param.data = project_to_power_of_two(param.data)


def initialize_power_of_two(model):
    with torch.no_grad():
        for param in model.parameters():
            param.data = project_to_power_of_two(param.data)

