import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from common.DELU import DELU

class CustomGELU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return 0.5 * input * (1 + torch.tanh(np.sqrt(2 / np.pi) * (input + 0.044715 * torch.pow(input, 3))))

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        tanh = torch.tanh(np.sqrt(2 / np.pi) * (input + 0.044715 * torch.pow(input, 3)))
        return grad_output * (0.5 * (1 + tanh) + 0.5 * input * (1 - tanh**2) * np.sqrt(2 / np.pi) * (1 + 0.134145 * input**2))

def GELU(x):
    return CustomGELU.apply(x)

def compute_gradient(func, x):
    x = x.clone().detach().requires_grad_(True)
    y = func(x)
    grad = torch.autograd.grad(y.sum(), x, create_graph=True)[0]
    return grad.detach()

def analyze_activation(func, x):
    with torch.no_grad():
        grad = compute_gradient(func, x)
        
        smoothness = np.mean(np.abs(np.diff(grad.cpu().numpy(), 2)))
        non_vanishing = torch.mean((grad.abs() > 1e-5).float()).item()
        saturation = (torch.mean(grad[x < -5].abs()).item() + torch.mean(grad[x > 5].abs()).item()) / 2
        magnitude = torch.mean(grad.abs()).item()
        linearity = np.abs(np.polyfit(x.cpu().numpy(), grad.cpu().numpy(), 1)[0])
    
    return {
        'smoothness': smoothness,
        'non_vanishing': non_vanishing,
        'saturation': saturation,
        'magnitude': magnitude,
        'linearity': linearity
    }

def compare_activations(delu_metrics, GELU_metrics):
    better_count = 0
    for metric in ['smoothness', 'non_vanishing', 'saturation', 'magnitude', 'linearity']:
        if metric in ['smoothness', 'saturation', 'linearity']:
            if delu_metrics[metric] < GELU_metrics[metric]:
                better_count += 1
        elif metric in ['non_vanishing']:
            if delu_metrics[metric] > GELU_metrics[metric]:
                better_count += 1
        elif metric == 'magnitude':
            if abs(delu_metrics[metric] - 1) < abs(GELU_metrics[metric] - 1):
                better_count += 1
    return better_count

def find_optimal_delu_params():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.linspace(-10, 10, 1000, device=device)
    
    GELU_metrics = analyze_activation(GELU, x)
    
    best_params = None
    best_better_count = 0
    best_metrics = None
    
    for a in np.arange(0.1, 2.1, 0.1):
        for b in np.arange(0.1, 2.1, 0.1):
            delu = DELU(a, b)
            delu_metrics = analyze_activation(delu, x)
            better_count = compare_activations(delu_metrics, GELU_metrics)
            
            if better_count > best_better_count:
                best_better_count = better_count
                best_params = (a, b)
                best_metrics = delu_metrics
    
    return best_params, best_better_count, best_metrics, GELU_metrics

def plot_activations(best_params):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.linspace(-10, 10, 1000, device=device)
    
    best_delu = DELU(*best_params)
    
    with torch.no_grad():
        plt.figure(figsize=(12, 8))
        plt.plot(x.cpu().numpy(), GELU(x).cpu().numpy(), label='GELU')
        plt.plot(x.cpu().numpy(), best_delu(x).cpu().numpy(), label=f'DELU (a={best_params[0]:.2f}, b={best_params[1]:.2f})')
        plt.title('GELU vs Optimized DELU')
        plt.xlabel('Input')
        plt.ylabel('Output')
        plt.legend()
        plt.grid(True)
        plt.savefig('GELU_vs_optimized_delu.png')
        plt.close()

if __name__ == "__main__":
    best_params, best_better_count, best_delu_metrics, GELU_metrics = find_optimal_delu_params()
    
    print(f"Best DELU parameters: a = {best_params[0]:.2f}, b = {best_params[1]:.2f}")
    print(f"Number of metrics where DELU is better: {best_better_count} out of 5")
    
    print("\nDELU Metrics:")
    for metric, value in best_delu_metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    print("\nGELU Metrics:")
    for metric, value in GELU_metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    plot_activations(best_params)
    print("\nPlot saved as 'GELU_vs_optimized_delu.png'")
