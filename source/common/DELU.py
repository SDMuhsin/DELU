import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from typing import Optional
import numpy as np

class AnalyticalGELUApprox(nn.Module):
    def __init__(self, k1=1.702, k2=0.147):
        super().__init__()
        self.k1 = k1
        self.k2 = k2

    def forward(self, x):
        return x * (1 / (1 + torch.exp(-self.k1 * x - self.k2 * x**3)))

import torch
import torch.nn as nn
import numpy as np
from typing import Optional

class HardwareGELUApprox(nn.Module):
    """PyTorch module that simulates hardware-efficient GELU approximation."""
    def __init__(self, data_width: int, frac_bits: int, lut_addr_bits: int = 6):
        super().__init__()
        
        self.data_width = data_width
        self.frac_bits = frac_bits
        self.int_bits = data_width - frac_bits - 1  # 1 bit for sign
        self.lut_addr_bits = lut_addr_bits
        
        # Compute range limits for fixed-point representation
        self.max_val = 2.0 ** self.int_bits - 2.0 ** (-self.frac_bits)
        self.min_val = -2.0 ** self.int_bits
        self.scale = 2.0 ** self.frac_bits
        
        # Initialize LUT for critical region (-0.5 to 0.5)
        self.register_buffer(
            'lut', 
            self._initialize_lut(),
            persistent=True
        )
        
        # Register region boundaries as buffers so they automatically move with the module
        self.register_buffer('lut_bound', torch.tensor(self.scale * 1.0))
        self.register_buffer('transition_bound', torch.tensor(self.scale * 3.0))
        
    def _initialize_lut(self) -> torch.Tensor:
        """Initialize lookup table with precise GELU values for critical region."""
        num_entries = 2 ** self.lut_addr_bits
        x = torch.linspace(-1.0, 1.0, num_entries)
        # True GELU values
        y = 0.5 * x * (1 + torch.erf(x / np.sqrt(2)))
        # Quantize to fixed-point precision
        y_fixed = self._quantize(y)
        return y_fixed
        
    def _quantize(self, x: torch.Tensor) -> torch.Tensor:
        """Simulate fixed-point quantization."""
        # Scale to fixed-point
        x_scaled = x * self.scale
        # Round to nearest integer
        x_scaled = torch.round(x_scaled)
        # Clamp to valid range
        max_int = (2 ** (self.data_width - 1)) - 1
        min_int = -(2 ** (self.data_width - 1))
        x_scaled = torch.clamp(x_scaled, min_int, max_int)
        return x_scaled
        
    def _fixed_to_float(self, x_fixed: torch.Tensor) -> torch.Tensor:
        """Convert fixed-point values back to floating point."""
        return x_fixed / self.scale
        
    def _lut_lookup(self, x: torch.Tensor) -> torch.Tensor:
        """Perform LUT lookup for values in critical region."""
        # Scale input to LUT address range
        scale_factor = (2 ** self.lut_addr_bits - 1) / 2.0
        indices = ((x / self.scale + 1.0) * scale_factor).long()
        indices = torch.clamp(indices, 0, 2 ** self.lut_addr_bits - 1)
        # Move indices to same device as LUT
        indices = indices.to(self.lut.device)
        return self.lut[indices]
        
    def _transition_region(self, x_fixed: torch.Tensor) -> torch.Tensor:
        """Compute transition region values using shifts and adds."""
        # Implement x/2 + x³/8 with fixed-point arithmetic
        x_half = torch.div(x_fixed, 2, rounding_mode='trunc')
        x_squared = torch.div(x_fixed * x_fixed, self.scale, rounding_mode='trunc')
        x_cubed = torch.div(x_squared * x_fixed, self.scale, rounding_mode='trunc')
        x_cubed_eighth = torch.div(x_cubed, 8, rounding_mode='trunc')
        return x_half + x_cubed_eighth

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass implementing hardware GELU approximation."""
        # Create result tensor on same device as input
        result = torch.zeros_like(x)
        
        # Clamp input to valid fixed-point range
        x_clamped = torch.clamp(x, self.min_val, self.max_val)
        
        # Convert to fixed-point representation
        x_fixed = self._quantize(x_clamped)
        
        # Compute absolute value for region detection
        x_abs = torch.abs(x_fixed)
        
        # LUT region
        lut_mask = x_abs < self.lut_bound
        if lut_mask.any():
            result[lut_mask] = self._lut_lookup(x_fixed[lut_mask])
        
        # Transition region
        trans_mask = (x_abs >= self.lut_bound) & (x_abs < self.transition_bound)
        if trans_mask.any():
            result[trans_mask] = self._transition_region(x_fixed[trans_mask])
        
        # Linear region
        linear_mask = x_abs >= self.transition_bound
        if linear_mask.any():
            result[linear_mask] = x_fixed[linear_mask]
            # Zero out large negative values
            result[linear_mask & (x_fixed < 0)] = 0
        
        # Convert back to floating point
        return self._fixed_to_float(result)
    
    def extra_repr(self) -> str:
        """Additional information for string representation."""
        return f'data_width={self.data_width}, frac_bits={self.frac_bits}, lut_addr_bits={self.lut_addr_bits}'


class RAF(nn.Module):
    def __init__(self):
        super(RAF, self).__init__()

    def forward(self, x):
        return torch.sign(x) * (1 - torch.exp(-torch.abs(x))) * (2 / (1 + torch.exp(-torch.abs(x))) - 1)
class LOGGELU(torch.nn.Module):
    def __init__(self, lambda_param=0.1):
        super(LOGGELU, self).__init__()
        self.lambda_param = lambda_param

    def forward(self, x):
        return x * torch.special.ndtr(x + self.lambda_param * torch.sign(x) * torch.log1p(torch.abs(x)))

    @staticmethod
    def gelu(x):
        return x * 0.5 * (1 + torch.erf(x / math.sqrt(2)))

    def extra_repr(self):
        return f'lambda_param={self.lambda_param}'

class HGELU(nn.Module):
    def __init__(self, delta=0.1, omega=math.pi):
        super(HGELU, self).__init__()
        self.delta = delta
        self.omega = omega

    def forward(self, x):
        # Ensure all operations are done on the same device as the input
        device = x.device
        
        # Standard GELU computation
        gelu = 0.5 * x * (1 + torch.erf(x / math.sqrt(2)))
        
        # Harmonic component
        harmonic = 1 + self.delta * torch.sin(self.omega * x)
        
        # Combine GELU with harmonic component
        return gelu * harmonic

class SQGELU(nn.Module):
    def __init__(self, epsilon=0.1):
        super(SQGELU, self).__init__()
        self.epsilon = epsilon

    def forward(self, x):
        return x * 0.5 * (1 + torch.erf((x + self.epsilon * x.pow(2)) / torch.sqrt(torch.tensor(2.0))))

class SEGELU(nn.Module):
    def __init__(self, gamma=0.1):
        super(SEGELU, self).__init__()
        self.gamma = gamma

    def forward(self, x):
        # Exact GELU implementation
        gelu = 0.5 * x * (1 + torch.erf(x / math.sqrt(2)))
        
        # SE-GELU modification
        return gelu * (1 + self.gamma * torch.exp(-torch.abs(x)))


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

import math

class SimpleDELU(nn.Module):
    def __init__(self, a: float = 1.0, b: float = 1.0):
        super(SimpleDELU, self).__init__()
        self.b = b
        # Pre-compute A = e^(-a * e)
        self.A = math.exp(-a * math.e)

    def forward(self, x):
        return x * torch.pow(self.A, -self.b * x)

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
