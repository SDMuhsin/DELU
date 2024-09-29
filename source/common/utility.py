import torch.nn as nn
from common.DELU import DELU,ADELU,TDELU,ATDELU,FADELU

def get_activation_by_name(activation_name,a=1.0,b=1.0,c=1.0,d=1.0):
    activation_map = {
        'ReLU': nn.ReLU(),
        'LeakyReLU': nn.LeakyReLU(),
        'ELU': nn.ELU(),
        'SELU': nn.SELU(),
        'GELU': nn.GELU(),
        'Tanh': nn.Tanh(),
        'Sigmoid': nn.Sigmoid(),
        'Hardswish': nn.Hardswish(),
        'Mish': nn.Mish(),
        'SiLU': nn.SiLU(),
        'Softplus': nn.Softplus(),
        'Softsign': nn.Softsign(),
        'Hardshrink': nn.Hardshrink(),
        'Softshrink': nn.Softshrink(),
        'Tanhshrink': nn.Tanhshrink(),
        'PReLU': nn.PReLU(),
        'RReLU': nn.RReLU(),
        'CELU': nn.CELU(),
        'Hardtanh': nn.Hardtanh(),
        'DELU' : DELU(a,b),
        'ADELU' : ADELU(),
        'TDELU' : TDELU(),
        'ATDELU' : ATDELU(),
        'FADELU' : FADELU(a,b,c,d)
    }
    print(f"a = {a}, d = {d}") 
    if activation_name in activation_map:
        return activation_map[activation_name]
    else:
        raise ValueError(f"Unsupported activation function: {activation_name}")
def replace_activations(module,old_activation,new_activation):
    """
    Recursively replaces all ReLU activations in the model with SmoothExponentialLinear (SEL).
    Args:
        module (nn.Module): The PyTorch model or module where replacements are needed.
    """
    for name, child in module.named_children():
        if isinstance(child, old_activation):
            # If the child is old_activation, replace with new activation
            print(f"Replacing {name}")
            setattr(module, name, new_activation)
        else:
            # Recursively apply to child modules
            replace_activations(child,old_activation,new_activation)
