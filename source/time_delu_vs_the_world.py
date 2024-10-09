import torch
import torch.nn as nn
from common.DELU import DELU, FADELU
import time
import csv
import matplotlib.pyplot as plt

# Manually specified colors
colors = {
    'DELU': '#e41a1c',
    'FADELU': '#377eb8',
    'ELU': '#4daf4a',
    'GELU': '#984ea3',
    'LeakyReLU': '#ff7f00',
    'Mish': '#ffff33',
    'RReLU': '#a65628',
    'ReLU': '#f781bf',
    'SELU': '#999999',
    'Softplus': '#66c2a5',
    'Softsign': '#fc8d62'
}

def run_comparison(input_size, num_runs=1000):
    device = torch.device("cpu" if torch.cuda.is_available() else "cpu")
    
    activation_functions = {
        'DELU': DELU(),
        'FADELU': FADELU(),
        'ELU': nn.ELU(),
        'GELU': nn.GELU(),
        'LeakyReLU': nn.LeakyReLU(),
        'Mish': nn.Mish(),
        'RReLU': nn.RReLU(),
        'ReLU': nn.ReLU(),
        'SELU': nn.SELU(),
        'Softplus': nn.Softplus(),
        'Softsign': nn.Softsign()
    }
    
    input_tensor = torch.randn(input_size).to(device)
    input_tensor.requires_grad = True
    
    def measure_time(func):
        start = time.perf_counter()
        result = func()
        torch.cuda.synchronize()
        end = time.perf_counter()
        return result, (end - start) * 1000
    
    results = {}
    
    for name, func in activation_functions.items():
        func = func.to(device)
        forward_times = []
        backward_times = []
        
        for _ in range(num_runs):
            _, forward_time = measure_time(lambda: func(input_tensor))
            output = func(input_tensor)
            _, backward_time = measure_time(lambda: output.sum().backward(retain_graph=True))
            
            input_tensor.grad = None
            
            forward_times.append(forward_time)
            backward_times.append(backward_time)
        
        avg_forward = sum(forward_times) / num_runs
        avg_backward = sum(backward_times) / num_runs
        
        results[name] = [avg_forward, avg_backward]
    
    return results

def plot_results(all_results, pass_type):
    plt.figure(figsize=(12, 8))
    for name in all_results[100].keys():
        times = [all_results[size][name][0 if pass_type == 'forward' else 1] for size in input_sizes]
        plt.plot(input_sizes, times, marker='o', label=name, color=colors[name], linewidth=2)
    
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Input Size', fontsize=14)
    plt.ylabel(f'Average {pass_type.capitalize()} Time (ms)', fontsize=14)
    plt.title(f'{pass_type.capitalize()} Pass Performance Comparison', fontsize=16)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., fontsize=12)
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.tight_layout()
    plt.savefig(f'./saves/{pass_type}_comparison.png', dpi=300, bbox_inches='tight')

def main():
    global input_sizes
    input_sizes = [100, 1000, 10000,100000]
    all_results = {}
    
    for size in input_sizes:
        print(f"\nRunning comparison for input size: {size}")
        results = run_comparison((size,))
        all_results[size] = results
    
    # Save results to CSV
    with open('./saves/delu_vs_the_world.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Input Size', 'Activation', 'Avg Forward (ms)', 'Avg Backward (ms)'])
        for size, results in all_results.items():
            for name, (forward, backward) in results.items():
                writer.writerow([size, name, forward, backward])
    
    # Plot graphs
    plot_results(all_results, 'forward')
    plot_results(all_results, 'backward')

if __name__ == "__main__":
    main()
