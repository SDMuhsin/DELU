import torch
import torch.nn as nn
class DELU(nn.Module): # Dampened Exponential Linear Unit

    def __init__(self, a: float = 1.0, b: float = 1.0):
        super(DELU, self).__init__()
        self.a = a
        self.b = b

    def forward(self, x):
        return x * torch.exp(-self.a * torch.exp(-self.b * x))


class LDELU(nn.Module):  # Learned Dampened Exponential Linear Unit
    def __init__(self, initial_a: float = 1.0, initial_b: float = 1.0):
        super(LDELU, self).__init__()
        self.a = nn.Parameter(torch.tensor(initial_a, dtype=torch.float))
        self.b = nn.Parameter(torch.tensor(initial_b, dtype=torch.float))

    def forward(self, x):
        return x * torch.exp(-self.a * torch.exp(-self.b * x))

