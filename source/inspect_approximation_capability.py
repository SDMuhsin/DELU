import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from common.DELU import FADELU, ADELU
from common.utility import get_activation_by_name
import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate

def optimize_activation(target_activation, learnable_activation):
    # Initialize learnable activation
    act = learnable_activation()
    
    # Create target activation function
    target_func = get_activation_by_name(target_activation)
    
    # Define optimizer
    optimizer = optim.Adam(act.parameters(), lr=0.001)
    
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
        y_act = act(x)
        y_target = target_func(x)
        
        # Compute loss
        loss = loss_fn(y_act, y_target)
        
        # Check for NaN loss
        if torch.isnan(loss):
            print(f"NaN loss detected at epoch {epoch+1}. Resetting parameters.")
            act = learnable_activation()
            optimizer = optim.Adam(act.parameters(), lr=0.001)
            continue
        
        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        
        # Clip gradients to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(act.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # Save best parameters
        if loss.item() < best_loss:
            best_loss = loss.item()
            best_params = {name: param.clone().detach() for name, param in act.named_parameters()}
        
    # Load best parameters
    act.load_state_dict(best_params)
    
    return act, best_loss

def main():
    activations = ['ELU', 'GELU', 'LeakyReLU', 'Mish', 'RReLU', 'ReLU', 'SELU', 'Softplus', 'Softsign']
    results = []

    for activation in activations:
        print(f"Approximating {activation}...")
        
        fadelu, fadelu_loss = optimize_activation(activation, FADELU)
        adelu, adelu_loss = optimize_activation(activation, ADELU)
        
        results.append([activation, fadelu_loss, adelu_loss])

    # Print results in a table
    headers = ["Activation", "FADELU MSE", "ADELU MSE"]
    print(tabulate(results, headers=headers, floatfmt=".6f"))

if __name__ == "__main__":
    main()
