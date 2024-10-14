import torch
import torch.nn as nn
import math
import torch.nn.functional as F

class SRGELU(nn.Module):
    def __init__(self, beta=0.1):
        super(SRGELU, self).__init__()
        self.beta = beta

    def forward(self, x):
        # Ensure beta is on the same device as x
        beta = torch.tensor(self.beta, device=x.device, dtype=x.dtype)
        
        # Compute the sign of x
        sign_x = torch.sign(x)
        
        # Compute the argument for the Gaussian CDF
        arg = x + beta * sign_x
        
        # Compute SRGELU
        return x * torch.special.ndtr(arg)

class RGELU(nn.Module):
    def __init__(self, sigma=0.1):
        super(RGELU, self).__init__()
        self.alpha = nn.Parameter(torch.ones(1))
        self.sigma = sigma

    def forward(self, x):
        # Generate random noise on the same device as x
        epsilon = torch.randn_like(x, device=x.device) * self.sigma
        
        # Ensure alpha is on the same device as x
        alpha = self.alpha.to(x.device)
        
        # Apply alpha and add noise
        x_modified = alpha * x + epsilon
        
        # GELU approximation using tanh
        return 0.5 * x_modified * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x_modified + 0.044715 * torch.pow(x_modified, 3))))

    def extra_repr(self):
        return f'sigma={self.sigma}'
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

class FastDELU(nn.Module):
    def __init__(self, a: float = 1.0, b: float = 1.0):
        super(FastDELU, self).__init__()
        self.a = nn.Parameter(torch.tensor(a))
        self.b = nn.Parameter(torch.tensor(b))

    @torch.jit.script_method
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


class FADELU(nn.Module):
    def __init__(self, init_a=1.0, init_b=1.0, init_c=0.1, init_d=1.0):
        super(FADELU, self).__init__()
        self.a = nn.Parameter(torch.tensor(init_a, dtype=torch.float32))
        self.b = nn.Parameter(torch.tensor(init_b, dtype=torch.float32))
        self.c = nn.Parameter(torch.tensor(init_c, dtype=torch.float32))
        self.d = nn.Parameter(torch.tensor(init_d, dtype=torch.float32))

    def forward(self, x):
        delu_part = x * torch.exp(-self.a * torch.exp(-self.b * x))
        tanh_part = self.c * torch.tanh(self.d * x)
        return delu_part + tanh_part
class SGELU(nn.Module):
    def __init__(self, alpha=1.702, beta=0.044):
        super(SGELU, self).__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, x):
        # Calculate the argument for the sigmoid function
        sigmoid_arg = self.alpha * x + self.beta * x ** 3
        # Apply the sigmoid function
        sigmoid = torch.sigmoid(sigmoid_arg)
        # Return the SGELU activation
        return x * sigmoid


class SWLU(nn.Module):
    def __init__(self, alpha=1.0):
        super(SWLU, self).__init__()
        self.alpha = nn.Parameter(torch.tensor(alpha))

    def forward(self, x):
        return x * torch.sigmoid(self.alpha * x)
