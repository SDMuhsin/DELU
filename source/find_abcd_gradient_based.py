import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate
from common.DELU import FADELU, ADELU
from common.utility import get_activation_by_name

class GELU(nn.Module):
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * torch.pow(x, 3))))

def compute_gradient(func, x):
    x.requires_grad_(True)
    y = func(x)
    grad = torch.autograd.grad(y.sum(), x, create_graph=True)[0]
    return grad

def evaluate_properties(func, x):
    grad = compute_gradient(func, x)
    second_grad = compute_gradient(lambda x: compute_gradient(func, x), x)
    
    smoothness = -torch.mean(torch.abs(second_grad))
    non_zero_grad = torch.mean(torch.abs(grad))
    bounded_grad = -torch.max(torch.abs(grad))
    identity_near_origin = -torch.mean(torch.abs(grad[:100] - 1))
    monotonicity = torch.min(grad)
    
    return smoothness, non_zero_grad, bounded_grad, identity_near_origin, monotonicity

def optimize_activation(learnable_activation, device):
    act = learnable_activation().to(device)
    gelu = GELU().to(device)
    
    optimizer = optim.Adam(act.parameters(), lr=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=500, factor=0.5, verbose=True)
    
    x = torch.linspace(-10, 10, 1000).unsqueeze(1).to(device)
    
    num_epochs = 4000
    best_score = float('-inf')
    best_loss = float('inf')
    best_params = None
    
    for epoch in range(num_epochs):
        act_props = evaluate_properties(act, x)
        gelu_props = evaluate_properties(gelu, x)
        
        score = sum([1 for a, g in zip(act_props, gelu_props) if a > g])
        loss = -sum(act_props)  # Minimize negative sum of properties
        
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"Invalid loss detected at epoch {epoch+1}. Resetting parameters.")
            act = learnable_activation().to(device)
            optimizer = optim.Adam(act.parameters(), lr=0.01)
            continue
        
        # Update best parameters based on loss
        if loss < best_loss:
            best_loss = loss.item()
            best_params = {name: param.clone().detach() for name, param in act.named_parameters()}
            best_score = score
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(act.parameters(), max_norm=1.0)
        optimizer.step()
        
        scheduler.step(loss)
        
        if epoch % 1000 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}, Score: {score}")
            for name, param in act.named_parameters():
                print(f"{name}: {param.item():.4f}")
            print()

    # Load the best parameters before returning
    act.load_state_dict(best_params)
    return act, best_score
def plot_activations(gelu, delu, fadelu, device):
    x = torch.linspace(-10, 10, 1000).unsqueeze(1).to(device)
    
    plt.figure(figsize=(12, 8))
    plt.plot(x.cpu().numpy(), gelu(x).detach().cpu().numpy(), label='GELU')
    plt.plot(x.cpu().numpy(), delu(x).detach().cpu().numpy(), label='DELU')
    plt.plot(x.cpu().numpy(), fadelu(x).detach().cpu().numpy(), label='FADELU')
    plt.legend()
    plt.title('Activation Functions Comparison')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid(True)
    plt.savefig('activation_comparison.png')
    plt.close()

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    gelu = GELU().to(device)
    delu, delu_score = optimize_activation(ADELU, device)
    fadelu, fadelu_score = optimize_activation(FADELU, device)

    x = torch.linspace(-10, 10, 1000).unsqueeze(1).to(device)
    gelu_props = evaluate_properties(gelu, x)
    delu_props = evaluate_properties(delu, x)
    fadelu_props = evaluate_properties(fadelu, x)

    properties = ['Smoothness', 'Non-zero gradients', 'Bounded gradients', 'Identity near origin', 'Monotonicity']
    results = [
        ['GELU'] + list(gelu_props),
        ['DELU'] + list(delu_props),
        ['FADELU'] + list(fadelu_props)
    ]

    headers = ['Activation'] + properties
    print(tabulate(results, headers=headers, floatfmt=".4f"))

    print(f"\nDELU better than GELU on {delu_score}/5 properties")
    print(f"FADELU better than GELU on {fadelu_score}/5 properties")

    plot_activations(gelu, delu, fadelu, device)

    print("\nDELU parameters:")
    for name, param in delu.named_parameters():
        print(f"{name}: {param.item():.4f}")

    print("\nFADELU parameters:")
    for name, param in fadelu.named_parameters():
        print(f"{name}: {param.item():.4f}")

if __name__ == "__main__":
    main()
