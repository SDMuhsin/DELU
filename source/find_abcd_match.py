import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from common.DELU import FADELU
from common.utility import get_activation_by_name
import numpy as np
import matplotlib.pyplot as plt

def parse_args():
    parser = argparse.ArgumentParser(description='Find optimal FADELU parameters')
    parser.add_argument('--activation', type=str, default='ReLU',
                        choices=['ADELU', 'ReLU', 'LeakyReLU', 'ELU', 'SELU', 'GELU', 'Tanh', 'Sigmoid',
                                 'Hardswish', 'Mish', 'SiLU', 'Softplus', 'Softsign', 'Hardshrink',
                                 'Softshrink', 'Tanhshrink', 'PReLU', 'RReLU', 'CELU', 'Hardtanh', 'DELU'],
                        help='Activation function to approximate')
    return parser.parse_args()

def optimize_fadelu(target_activation):
    # Initialize FADELU
    fadelu = FADELU()
    
    # Create target activation function
    target_func = get_activation_by_name(target_activation)
    
    # Define optimizer
    optimizer = optim.Adam(fadelu.parameters(), lr=0.001)
    
    # Define loss function
    loss_fn = nn.MSELoss()
    
    # Generate input range
    x = torch.linspace(-10, 10, 1000).unsqueeze(1)
    
    # Training loop
    num_epochs = 10000
    best_loss = float('inf')
    best_params = None
    
    for epoch in range(num_epochs):
        # Forward pass
        y_fadelu = fadelu(x)
        y_target = target_func(x)
        
        # Compute loss
        loss = loss_fn(y_fadelu, y_target)
        
        # Check for NaN loss
        if torch.isnan(loss):
            print(f"NaN loss detected at epoch {epoch+1}. Resetting parameters.")
            fadelu = FADELU()  # Reset FADELU
            optimizer = optim.Adam(fadelu.parameters(), lr=0.001)  # Reset optimizer
            continue
        
        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        
        # Clip gradients to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(fadelu.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # Save best parameters
        if loss.item() < best_loss:
            best_loss = loss.item()
            best_params = {name: param.clone().detach() for name, param in fadelu.named_parameters()}
        
        if (epoch + 1) % 1000 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
    
    # Load best parameters
    fadelu.load_state_dict(best_params)
    
    return fadelu

def plot_results(fadelu, target_activation):
    x = torch.linspace(-10, 10, 1000)
    y_fadelu = fadelu(x)
    y_target = get_activation_by_name(target_activation)(x)
    
    # Calculate the Squared Error
    squared_error = (y_fadelu - y_target) ** 2

    # Calculate the Mean Squared Error
    mse = squared_error.mean().item()

    # Get DELU parameters
    a, b, c, d = fadelu.a.item(), fadelu.b.item(), fadelu.c.item(), fadelu.d.item()

    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Plot DELU and target activation
    ax1.plot(x.numpy(), y_fadelu.detach().numpy(), label='DELU')
    ax1.plot(x.numpy(), y_target.detach().numpy(), label=target_activation)
    ax1.set_xlabel('x')
    ax1.set_ylabel('Activation')
    ax1.tick_params(axis='y')

    # Create a secondary y-axis for Squared Error
    ax2 = ax1.twinx()
    ax2.plot(x.numpy(), squared_error.detach().numpy(), label='Squared Error', color='r', linestyle='--')
    ax2.set_ylabel('Squared Error')
    ax2.set_ylim(0, 0.15)
    ax2.tick_params(axis='y')

    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

    plt.title(f'$DELU_{{{a:.2f},{b:.2f},{c:.2f},{d:.2f}}}$ vs {target_activation}\nMSE: {mse:.4f}')
    plt.grid(True)
    plt.savefig(f'delu_vs_{target_activation}.png')
    plt.close()

def main():
    args = parse_args()
    target_activation = args.activation
    
    print(f"Optimizing FADELU for {target_activation}")
    
    # Optimize FADELU
    optimized_fadelu = optimize_fadelu(target_activation)
    
    # Print optimized parameters
    print("Optimized FADELU parameters:")
    print(f"a: {optimized_fadelu.a.item():.4f}")
    print(f"b: {optimized_fadelu.b.item():.4f}")
    print(f"c: {optimized_fadelu.c.item():.4f}")
    print(f"d: {optimized_fadelu.d.item():.4f}")
    
    # Plot results
    plot_results(optimized_fadelu, target_activation)
    
    print(f"Results plotted and saved as 'fadelu_vs_{target_activation}.png'")

if __name__ == "__main__":
    main()
