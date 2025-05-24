import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, input_dim=28*28, output_dim=10, 
                 hidden_dims=[512, 512], 
                 dropout_list=None, 
                 use_relu_list=None):
        super(MLP, self).__init__()

        num_hidden_layers = len(hidden_dims)

        if dropout_list is None:
            dropout_list = [0.0] * num_hidden_layers
        if use_relu_list is None:
            use_relu_list = [True] * num_hidden_layers

        assert len(dropout_list) == num_hidden_layers, "dropout_list must match number of hidden layers"
        assert len(use_relu_list) == num_hidden_layers, "use_relu_list must match number of hidden layers"

        layers = []
        in_dim = input_dim

        for i in range(num_hidden_layers):
            out_dim = hidden_dims[i]
            layers.append(nn.Linear(in_dim, out_dim))
            if use_relu_list[i]:
                layers.append(nn.ReLU())
            if dropout_list[i] > 0:
                layers.append(nn.Dropout(dropout_list[i]))
            in_dim = out_dim

        layers.append(nn.Linear(in_dim, output_dim))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.model(x)