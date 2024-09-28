import torch
import torch.nn as nn
class DELU(nn.Module): # Dampened Exponential Linear Unit

    def __init__(self, a: float = 1.0, b: float = 1.0):
        super(DELU, self).__init__()
        self.a = a
        self.b = b

    def forward(self, x):
        return x * torch.exp(-self.a * torch.exp(-self.b * x))


class ADELU(nn.Module):  # Learned Dampened Exponential Linear Unit
    def __init__(self, initial_a: float = 1.0, initial_b: float = 1.0):
        super(ADELU, self).__init__()
        self.a = nn.Parameter(torch.tensor(initial_a, dtype=torch.float))
        self.b = nn.Parameter(torch.tensor(initial_b, dtype=torch.float))

    def forward(self, x):
        return x * torch.exp(-self.a * torch.exp(-self.b * x))

class DELU_faster(nn.Module):
    def __init__(self, a: float = 1.0, b: float = 1.0):
        super(DELU_faster, self).__init__()
        self.a = a
        self.b = b

    def forward(self, x):
        return x * torch.exp(-self.a * torch.exp(-self.b * x))


class TDELU(nn.Module):  # Tanh Dampened Exponential Linear Unit
    def __init__(self, a: float = 0.5, b: float = 1.0):
        super(TDELU, self).__init__()
        self.a = a
        self.b = b

    def forward(self, x):
        return 0.5 * self.a * x * (1 + torch.tanh(self.b * x))


class ATDELU(nn.Module):  # Adaptive Tanh Dampened Exponential Linear Unit
    def __init__(self, initial_a: float = 1.0, initial_b: float = 1.0):
        super(ATDELU, self).__init__()
        self.a = nn.Parameter(torch.tensor(initial_a, dtype=torch.float))
        self.b = nn.Parameter(torch.tensor(initial_b, dtype=torch.float))

    def forward(self, x):
        return 0.5 * self.a * x * (1 + torch.tanh(self.b * x))
