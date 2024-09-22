import numpy as np
from scipy import optimize

def delu_prime(x, a, b):
    return np.exp(-a * np.exp(-b * x)) * (1 + a * b * x * np.exp(-b * x))

def delu_prime_second_derivative(x, a, b):
    # This is a numerical approximation of the second derivative of DELU'(x)
    h = 1e-5
    return (delu_prime(x + h, a, b) - 2 * delu_prime(x, a, b) + delu_prime(x - h, a, b)) / (h ** 2)

def integrated_squared_jerk(params):
    a, b = params
    x = np.linspace(-10, 10, 1000)
    jerk = delu_prime_second_derivative(x, a, b)
    return np.trapz(jerk ** 2, x)

def optimize_smoothness():
    result = optimize.minimize(
        integrated_squared_jerk,
        x0=[1, 1],  # Initial guess for [a, b]
        method='Nelder-Mead',
        options={'maxiter': 1000}
    )
    return result.x

# Find optimal a and b
optimal_a, optimal_b = optimize_smoothness()

print(f"Optimal a: {optimal_a}")
print(f"Optimal b: {optimal_b}")

# Evaluate smoothness at optimal values
smoothness = integrated_squared_jerk([optimal_a, optimal_b])
print(f"Smoothness measure at optimal values: {smoothness}")

# Compare with some other values
print("\nComparison with other values:")
for a, b in [(1, 1), (0.5, 0.5), (2, 2)]:
    smoothness = integrated_squared_jerk([a, b])
    print(f"a = {a}, b = {b}: Smoothness = {smoothness}")
