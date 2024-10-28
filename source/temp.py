import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf
from sklearn.metrics import mean_squared_error, mean_absolute_error

BW =4
def true_gelu(x):
    """True GELU implementation"""
    return 0.5 * x * (1 + erf(x / np.sqrt(2)))

def analytical_gelu_approx(x, k1=1.702, k2=0.147):
    """Analytical approximation of GELU"""
    return x * (1 / (1 + np.exp(-k1*x - k2*x**3)))

def hardware_gelu_approx(x):
    """Hardware approximation of GELU based on the VHDL implementation"""
    DATA_WIDTH = BW
    FRAC_BITS = BW // 2
    NEG_TWO = -2 * (2**FRAC_BITS)
    POS_TWO = 2 * (2**FRAC_BITS)

    # Convert to fixed-point representation
    x_fixed = np.round(x * (2**FRAC_BITS)).astype(int)
    
    result = np.zeros_like(x_fixed)
    
    for i, val in enumerate(x_fixed):
        if val < NEG_TWO:
            # Region 1: Near-zero linear approximation
            result[i] = val >> 4  # x/16
        elif val > POS_TWO:
            # Region 3: Linear approximation
            result[i] = val - (val >> 3)  # x - x/8
        else:
            # Region 2: Quadratic approximation
            x_shifted = val >> 1  # x/2
            x_squared = (val * val) >> FRAC_BITS
            quad_term = x_squared >> 3
            
            if val < 0:
                result[i] = x_shifted - quad_term
            else:
                result[i] = x_shifted + quad_term
    
    # Convert back to floating-point
    return result.astype(float) / (2**FRAC_BITS)

# Generate x values
x = np.linspace(-3, 3, 1000)

# Calculate y values for each function
y_true = true_gelu(x)
y_analytical = analytical_gelu_approx(x)
y_hardware = hardware_gelu_approx(x)

# Calculate error metrics
mse_analytical = mean_squared_error(y_true, y_analytical)
rmse_analytical = np.sqrt(mse_analytical)
mae_analytical = mean_absolute_error(y_true, y_analytical)

mse_hardware = mean_squared_error(y_true, y_hardware)
rmse_hardware = np.sqrt(mse_hardware)
mae_hardware = mean_absolute_error(y_true, y_hardware)

# Print error metrics
print("Analytical Approximation:")
print(f"MSE: {mse_analytical:.6f}")
print(f"RMSE: {rmse_analytical:.6f}")
print(f"MAE: {mae_analytical:.6f}")
print("\nHardware Approximation:")
print(f"MSE: {mse_hardware:.6f}")
print(f"RMSE: {rmse_hardware:.6f}")
print(f"MAE: {mae_hardware:.6f}")

# Plot the results
plt.figure(figsize=(12, 8))
plt.plot(x, y_true, label='True GELU', linewidth=2)
plt.plot(x, y_analytical, label='Analytical Approximation', linestyle='--')
plt.plot(x, y_hardware, label=f'Hardware Approximation ({BW}-bit)', linestyle=':', linewidth=2)
plt.xlabel('x')
plt.ylabel('GELU(x)')
plt.title('GELU Approximations')
plt.legend()

plt.savefig("./saves/gelu_vs_hardware.png")
