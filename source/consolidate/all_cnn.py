import os
import pandas as pd
from tabulate import tabulate
import numpy as np

# Configuration dictionary
config = {
    'mnist': {'epochs': 10, 'batch_size': 64, 'learning_rate': 0.001},
    'fmnist': {'epochs': 10, 'batch_size': 64, 'learning_rate': 0.001},
    'cifar10': {'epochs': 50, 'batch_size': 128, 'learning_rate': 0.01},
    'cifar100': {'epochs': 100, 'batch_size': 128, 'learning_rate': 0.01},
    'svhn': {'epochs': 50, 'batch_size': 128, 'learning_rate': 0.01},
    'stl10': {'epochs': 100, 'batch_size': 64, 'learning_rate': 0.001},
    'emnist': {'epochs': 20, 'batch_size': 64, 'learning_rate': 0.001},
    'kmnist': {'epochs': 20, 'batch_size': 64, 'learning_rate': 0.001}
}

# Function to process a single dataset
def process_dataset(file_path, dataset_name):
    df = pd.read_csv(file_path)
    
    # Filter based on config
    df_filtered = df[
        (df['Total Epochs'] == config[dataset_name]['epochs']) &
        (df['Batch Size'] == config[dataset_name]['batch_size']) &
        (df['Learning Rate'] == config[dataset_name]['learning_rate'])
    ]
    
    # Check for missing seeds
    seeds = [41, 42, 43, 44, 45]
    missing_seeds = set(seeds) - set(df_filtered['Seed'])
    if missing_seeds:
        raise ValueError(f"Missing seeds {missing_seeds} for dataset {dataset_name}")
    
    # Group by Activation Function and calculate median
    result = df_filtered.groupby('Activation Function')['Top-1 Accuracy'].median().reset_index()
    result['Dataset'] = dataset_name
    return result

# Process all datasets
all_results = []
for file in os.listdir('saves'):
    if file.endswith('_results.txt'):
        dataset_name = file.split('_')[0]
        if dataset_name in config:
            file_path = os.path.join('saves', file)
            try:
                result = process_dataset(file_path, dataset_name)
                all_results.append(result)
            except ValueError as e:
                print(str(e))
                continue

# Combine all results
combined_results = pd.concat(all_results, ignore_index=True)

# Find common activation functions
activation_functions = set.intersection(*[set(df['Activation Function']) for df in all_results])

# Filter for common activation functions
combined_results = combined_results[combined_results['Activation Function'].isin(activation_functions)]

# Pivot the table
pivot_table = combined_results.pivot(index='Activation Function', columns='Dataset', values='Top-1 Accuracy')

# Convert to percentage
pivot_table = pivot_table * 100

# Generate LaTeX table
latex_table = tabulate(pivot_table, headers='keys', tablefmt='latex_raw', floatfmt='.2f')

# Print LaTeX table
print(latex_table)

# Optionally, save to a file
with open('consolidated_results.tex', 'w') as f:
    f.write(latex_table)
