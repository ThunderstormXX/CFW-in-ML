import torch.nn as nn

def count_all_params(model):
    return sum(p.numel() for p in model.parameters())