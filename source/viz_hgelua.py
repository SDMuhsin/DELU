import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import os
from common.DELU import HardwareGELUApprox

# Ensure the saves directory exists
os.makedirs('./saves', exist_ok=True)

# Initialize GELU functions
hardware_gelu = HardwareGELUApprox(data_width=16, frac_bits=8)
true_gelu = nn.GELU()

# Generate input data in the range [-4, 4]
x_vals = torch.linspace(-4, 4, steps=1000)

# Compute outputs for both true and hardware GELU approximations
with torch.no_grad():
    y_true = true_gelu(x_vals)
    y_approx = hardware_gelu(x_vals)

# Plot the results
plt.figure(figsize=(8, 6))
plt.plot(x_vals, y_true, label='True GELU', color='blue', linewidth=2)
plt.plot(x_vals, y_approx, label='Hardware GELU Approximation', color='orange', linestyle='--', linewidth=2)
plt.xlabel("Input")
plt.ylabel("Output")
plt.title("True GELU vs Hardware GELU Approximation")
plt.legend()
plt.grid(True)

# Save the plot as a PNG file
output_path = './saves/hardware_gelu_approximation.png'
plt.savefig(output_path, format='png', dpi=300)
plt.close()

print(f"Plot saved to {output_path}")

