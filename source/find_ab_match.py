import numpy as np
from scipy import optimize, special

def gelu(x):
    return x * 0.5 * (1 + special.erf(x / np.sqrt(2)))

def delu(x, a, b):
    return x * np.exp(-a * np.exp(-b * x))

def gelu_derivative(x):
    return 0.5 * (1 + special.erf(x / np.sqrt(2))) + x * (1 / np.sqrt(2 * np.pi)) * np.exp(-x**2 / 2)

def delu_derivative(x, a, b):
    return np.exp(-a * np.exp(-b * x)) * (1 + a * b * x * np.exp(-b * x))


DISTANCE_RANGE = 2
def distance(params):
    a, b = params
    x = np.linspace(-DISTANCE_RANGE, DISTANCE_RANGE, 1000)
    return np.sum((gelu(x) - delu(x, a, b))**2)

def distance_derivative(params):
    a, b = params
    x = np.linspace(-DISTANCE_RANGE, DISTANCE_RANGE, 1000)
    return np.sum((gelu_derivative(x) - delu_derivative(x, a, b))**2)

AB_RANGE = 2
# Optimize for objective 1
result1 = optimize.minimize(distance, [1, 1], method='Nelder-Mead')
optimal_a1, optimal_b1 = result1.x

# Optimize for objective 2
result2 = optimize.minimize(distance_derivative, [1, 1], method='Nelder-Mead')
optimal_a2, optimal_b2 = result2.x

print(f"Optimal (a, b) for GELU vs DELU: ({optimal_a1}, {optimal_b1})")
print(f"Optimal (a, b) for GELU' vs DELU': ({optimal_a2}, {optimal_b2})")
