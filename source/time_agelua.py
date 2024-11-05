import torch
import torch.nn as nn
import time
from tabulate import tabulate
from common.DELU import *

class AnalyticalGELUApprox(nn.Module):
    def __init__(self, k1=1.702, k2=0.147):
        super().__init__()
        self.k1 = k1
        self.k2 = k2

    def forward(self, x):
        return x * (1 / (1 + torch.exp(-self.k1 * x - self.k2 * x**3)))

def measure_performance(func, x, device, backward=False):
    x = x.to(device)
    x.requires_grad = backward
    
    start = time.perf_counter()
    y = func(x)
    if backward:
        y.sum().backward()
    torch.cuda.synchronize() if device == 'cuda' else None
    end = time.perf_counter()
    
    return end - start

def calculate_metrics(approx, true):
    mse = nn.MSELoss()(approx, true)
    rmse = torch.sqrt(mse)
    mae = nn.L1Loss()(approx, true)
    return mse.item(), rmse.item(), mae.item()

def run_comparison(n_runs=1000, n_elements=1000000):
    devices = ['cpu', 'cuda'] if torch.cuda.is_available() else ['cpu']
    functions = {
        'Approx GELU': AnalyticalGELUApprox(),
        'True GELU': nn.GELU(),
        'Tanh GELU': nn.GELU(approximate='tanh')
    }
    
    results = []
    
    for device in devices:
        x = torch.randn(n_elements, device=device)
        true_gelu = nn.GELU()(x).detach()
        
        for name, func in functions.items():
            forward_time = 0
            backward_time = 0
            
            for _ in range(n_runs):
                forward_time += measure_performance(func, x, device)
                backward_time += measure_performance(func, x, device, backward=True)
            
            forward_time /= n_runs
            backward_time /= n_runs
            
            output = func(x)
            mse, rmse, mae = calculate_metrics(output, true_gelu)
            
            results.append([
                device, name, mse, rmse, mae,
                forward_time * 1000, backward_time * 1000
            ])
    
    return results

results = run_comparison()

headers = ['Device', 'Function', 'MSE', 'RMSE', 'MAE', 'Forward (ms)', 'Backward (ms)']
table = tabulate(results, headers=headers, floatfmt='.6f')
print(table)

# Calculate percentage speedups
for device in ['cpu', 'cuda']:
    device_results = [r for r in results if r[0] == device]
    true_gelu_forward = next(r[5] for r in device_results if r[1] == 'True GELU')
    true_gelu_backward = next(r[6] for r in device_results if r[1] == 'True GELU')
    
    print(f"\nPercentage speedups for {device}:")
    for r in device_results:
        if r[1] != 'True GELU':
            forward_speedup = (true_gelu_forward - r[5]) / true_gelu_forward * 100
            backward_speedup = (true_gelu_backward - r[6]) / true_gelu_backward * 100
            print(f"{r[1]}:")
            print(f"  Forward: {forward_speedup:.2f}%")
            print(f"  Backward: {backward_speedup:.2f}%")
