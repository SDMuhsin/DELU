import torch
import torch.nn as nn
class DELU(nn.Module): # Dampened Exponential Linear Unit
    def forward(self, x):
        return x * torch.exp(-torch.exp(-x))

