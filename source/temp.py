import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf
from scipy.optimize import minimize
import os

def true_gelu(x):
    """Implementation of the true GELU function"""
    return 0.5 * x * (1 + erf(x / np.sqrt(2)))

def proposed_gelu(x, k1, k2):
    """Our proposed GELU approximation"""
    return x * (1 / (1 + np.exp(-k1*x - k2*x**3)))

def compute_metrics(func1, func2, x_range):
    """Compute MSE, MAE, and RMSE between two functions"""
    y1 = func1(x_range)
    y2 = func2(x_range)
    mse = np.mean((y1 - y2) ** 2)
    mae = np.mean(np.abs(y1 - y2))
    rmse = np.sqrt(mse)
    return {'MSE': mse, 'MAE': mae, 'RMSE': rmse}

# Generate data points for optimization
x_train = np.linspace(-5, 5, 1000)
y_train = true_gelu(x_train)

def objective(params):
    """Objective function for optimization"""
    k1, k2 = params
    y_pred = proposed_gelu(x_train, k1, k2)
    return np.mean((y_train - y_pred) ** 2)

# Optimize parameters
initial_guess = [1.0, 0.1]
result = minimize(objective, initial_guess, method='Nelder-Mead')
k1_opt, k2_opt = result.x

# Create visualization
x_test = np.linspace(-5, 5, 1000)
y_true = true_gelu(x_test)
y_approx = proposed_gelu(x_test, k1_opt, k2_opt)

# Calculate metrics
metrics = compute_metrics(
    lambda x: true_gelu(x),
    lambda x: proposed_gelu(x, k1_opt, k2_opt),
    x_test
)

# Create the visualization
plt.figure(figsize=(12, 8))
plt.plot(x_test, y_true, 'b-', label='True GELU', linewidth=2)
plt.plot(x_test, y_approx, 'r--', label='Proposed Approximation', linewidth=2)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=12)
plt.xlabel('x', fontsize=12)
plt.ylabel('y', fontsize=12)
plt.title('GELU vs Proposed Approximation\n' + 
          f'MSE: {metrics["MSE"]:.2e}, MAE: {metrics["MAE"]:.2e}, RMSE: {metrics["RMSE"]:.2e}\n' +
          f'k₁: {k1_opt:.4f}, k₂: {k2_opt:.4f}', 
          fontsize=14)

# Create error plot
plt.figure(figsize=(12, 4))
plt.plot(x_test, np.abs(y_true - y_approx), 'g-', label='Absolute Error')
plt.grid(True, alpha=0.3)
plt.legend(fontsize=12)
plt.xlabel('x', fontsize=12)
plt.ylabel('Absolute Error', fontsize=12)
plt.title('Approximation Error Distribution', fontsize=14)

# Save plots
os.makedirs('./saves', exist_ok=True)
plt.figure(1)
plt.savefig('./saves/gelu_comparison.png', dpi=300, bbox_inches='tight')
plt.figure(2)
plt.savefig('./saves/gelu_error.png', dpi=300, bbox_inches='tight')

# Print optimized parameters and metrics
print(f"Optimized parameters: k₁ = {k1_opt:.6f}, k₂ = {k2_opt:.6f}")
for metric_name, value in metrics.items():
    print(f"{metric_name}: {value:.2e}")
