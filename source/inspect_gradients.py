import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from common.DELU import DELU
from common.utility import get_activation_by_name

def analyze_gradients(compare_func_name='gelu', a=0.5, b=1.2, x_range=(-10, 10), num_points=1000):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define activation functions
    delu = DELU(a, b)
    compare_func = get_activation_by_name(compare_func_name)

    # Function to compute gradient
    def compute_gradient(func, x):
        x.requires_grad_(True)
        y = func(x)
        y.backward(torch.ones_like(x))
        return x.grad.clone()

    # Generate input values
    x = torch.linspace(x_range[0], x_range[1], num_points, device=device)

    # Compute gradients
    delu_grad = compute_gradient(delu, x.clone())
    compare_grad = compute_gradient(compare_func, x.clone())

    # Plot gradients
    plt.figure(figsize=(12, 8))
    plt.plot(x.cpu().numpy(), delu_grad.cpu().numpy(), label='DELU')
    plt.plot(x.cpu().numpy(), compare_grad.cpu().numpy(), label=compare_func_name.upper())
    plt.title(f'Gradients of DELU vs {compare_func_name.upper()}')
    plt.xlabel('Input')
    plt.ylabel('Gradient')
    plt.legend()
    plt.grid(True)
    plt.savefig('activation_gradients.png')
    plt.close()

    # Analysis functions
    def analyze_smoothness(grad):
        return np.mean(np.abs(np.diff(grad.cpu().numpy(), 2)))

    def analyze_non_vanishing(grad):
        return torch.mean((grad.abs() > 1e-5).float()).item()

    def analyze_saturation(grad, x):
        left_saturation = torch.mean(grad[x < -5].abs()).item()
        right_saturation = torch.mean(grad[x > 5].abs()).item()
        return (left_saturation + right_saturation) / 2

    def analyze_magnitude(grad):
        return torch.mean(grad.abs()).item()

    def analyze_symmetry(grad):
        return np.abs(np.mean(grad.cpu().numpy()) - np.median(grad.cpu().numpy()))

    def analyze_inflection_points(grad):
        return np.sum(np.diff(np.sign(np.diff(grad.cpu().numpy()))) != 0)

    def analyze_zero_gradient(grad, x):
        near_zero = (x.abs() < 1e-3)
        if near_zero.sum() == 0:
            return 0.0
        return torch.mean((grad[near_zero].abs() > 1e-5).float()).item()

    def analyze_linearity(grad):
        x_np = x.cpu().numpy()
        grad_np = grad.cpu().numpy()
        slope, _ = np.polyfit(x_np, grad_np, 1)
        return np.abs(slope)

    # Perform analysis
    analyses = {
        'Smoothness (lower is smoother)': (analyze_smoothness(delu_grad), analyze_smoothness(compare_grad)),
        'Non-vanishing gradient (higher is better)': (analyze_non_vanishing(delu_grad), analyze_non_vanishing(compare_grad)),
        'Saturation (lower is less saturated)': (analyze_saturation(delu_grad, x), analyze_saturation(compare_grad, x)),
        'Gradient magnitude (closer to 1 is better)': (analyze_magnitude(delu_grad), analyze_magnitude(compare_grad)),
        'Symmetry (lower is more symmetric)': (analyze_symmetry(delu_grad), analyze_symmetry(compare_grad)),
        'Inflection points': (analyze_inflection_points(delu_grad), analyze_inflection_points(compare_grad)),
        'Non-zero gradient near zero (higher is better)': (analyze_zero_gradient(delu_grad, x), analyze_zero_gradient(compare_grad, x)),
        'Linearity (closer to 0 is more linear)': (analyze_linearity(delu_grad), analyze_linearity(compare_grad))
    }

    # Print results
    print(f"\nAnalysis of DELU({a}, {b}) vs {compare_func_name.upper()} gradients:")
    for criterion, (delu_value, compare_value) in analyses.items():
        print(f"\n{criterion}:")
        print(f"  DELU: {delu_value:.4f}")
        print(f"  {compare_func_name.upper()}: {compare_value:.4f}")
        
        if criterion in ['Non-vanishing gradient (higher is better)', 'Non-zero gradient near zero (higher is better)']:
            better = 'DELU' if delu_value > compare_value else compare_func_name.upper()
        elif criterion in ['Smoothness (lower is smoother)', 'Saturation (lower is less saturated)', 'Symmetry (lower is more symmetric)', 'Linearity (closer to 0 is more linear)']:
            better = 'DELU' if delu_value < compare_value else compare_func_name.upper()
        elif criterion == 'Gradient magnitude (closer to 1 is better)':
            better = 'DELU' if abs(delu_value - 1) < abs(compare_value - 1) else compare_func_name.upper()
        else:
            better = 'Comparison inconclusive'
        
        print(f"  Better: {better}")

    print("\nGradient analysis completed. Plot saved as 'activation_gradients.png'.")

# Run the analysis
analyze_gradients('GELU', a=0.5, b=1.2)
