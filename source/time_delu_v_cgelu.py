import torch
import torch.nn as nn
from common.DELU import DELU
from tabulate import tabulate
import time
import math

class CustomGELU(nn.Module):
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

def run_comparison(input_size, num_runs=1000):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    delu = DELU().to(device)
    custom_gelu = CustomGELU().to(device)
    
    input_tensor = torch.randn(input_size).to(device)
    input_tensor.requires_grad = True
    
    def measure_time(func):
        start = time.perf_counter()
        result = func()
        torch.cuda.synchronize()
        end = time.perf_counter()
        return result, (end - start) * 1000  # Convert to milliseconds
    
    delu_forward_times = []
    delu_backward_times = []
    gelu_forward_times = []
    gelu_backward_times = []
    
    for _ in range(num_runs):
        # DELU forward and backward
        _, delu_forward_time = measure_time(lambda: delu(input_tensor))
        delu_output = delu(input_tensor)
        _, delu_backward_time = measure_time(lambda: delu_output.sum().backward(retain_graph=True))
        
        # Reset gradients
        input_tensor.grad = None
        
        # Custom GELU forward and backward
        _, gelu_forward_time = measure_time(lambda: custom_gelu(input_tensor))
        gelu_output = custom_gelu(input_tensor)
        _, gelu_backward_time = measure_time(lambda: gelu_output.sum().backward(retain_graph=True))
        
        # Reset gradients
        input_tensor.grad = None
        
        delu_forward_times.append(delu_forward_time)
        delu_backward_times.append(delu_backward_time)
        gelu_forward_times.append(gelu_forward_time)
        gelu_backward_times.append(gelu_backward_time)
    
    avg_delu_forward = sum(delu_forward_times) / num_runs
    avg_delu_backward = sum(delu_backward_times) / num_runs
    avg_gelu_forward = sum(gelu_forward_times) / num_runs
    avg_gelu_backward = sum(gelu_backward_times) / num_runs
    
    delu_forward_speedup = (avg_gelu_forward - avg_delu_forward) / avg_gelu_forward * 100
    delu_backward_speedup = (avg_gelu_backward - avg_delu_backward) / avg_gelu_backward * 100
    
    return [
        ["DELU", avg_delu_forward, avg_delu_backward, delu_forward_speedup, delu_backward_speedup],
        ["Custom GELU", avg_gelu_forward, avg_gelu_backward, 0, 0]
    ]

def main():
    input_sizes = [(1000,), (10000,), (100000,)]
    headers = ["Activation", "Avg Forward (ms)", "Avg Backward (ms)", "Forward Speedup (%)", "Backward Speedup (%)"]
    
    for size in input_sizes:
        print(f"\nInput size: {size}")
        results = run_comparison(size)
        print(tabulate(results, headers=headers, floatfmt=".4f"))

if __name__ == "__main__":
    main()
