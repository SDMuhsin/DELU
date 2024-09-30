import os
import pandas as pd
from tabulate import tabulate
import numpy as np

# Configuration dictionary
config = {
    'mnist': {'epochs': 10, 'batch_size': 64, 'learning_rate': 0.001},
    'fmnist': {'epochs': 10, 'batch_size': 64, 'learning_rate': 0.001},
    'cifar10': {'epochs': 10, 'batch_size': 128, 'learning_rate': 0.01},
    'cifar100': {'epochs': 50, 'batch_size': 128, 'learning_rate': 0.01},
    'svhn': {'epochs': 50, 'batch_size': 128, 'learning_rate': 0.01},
    'stl10': {'epochs': 50, 'batch_size': 64, 'learning_rate': 0.001},
    'emnist': {'epochs': 30, 'batch_size': 64, 'learning_rate': 0.001},
    'kmnist': {'epochs': 10, 'batch_size': 64, 'learning_rate': 0.001}
}

def process_dataset(file_path, dataset_name):
    df = pd.read_csv(file_path)
    
    # Filter based on config
    df_filtered = df[
        (df['Total Epochs'] == config[dataset_name]['epochs']) &
        (df['Batch Size'] == config[dataset_name]['batch_size']) &
        (df['Learning Rate'] == config[dataset_name]['learning_rate'])
    ]
    
    if df_filtered.empty:
        print(f"Warning: No data for dataset {dataset_name} with the specified configuration.")
        return pd.DataFrame(columns=['Activation Function', 'Top-1 Accuracy', 'Dataset'])

    seeds = [41, 42, 43, 44, 45]
    result = []
    for activation in df_filtered['Activation Function'].unique():
        df_act = df_filtered[df_filtered['Activation Function'] == activation]
        if len(df_act) < 5:
            missing_seeds = set(seeds) - set(df_act['Seed'])
            print(f"Warning: Missing seeds {missing_seeds} for dataset {dataset_name}, activation {activation}")
            # Fill missing seeds with dash
            for seed in missing_seeds:
                df_act = df_act.append({
                    'Activation Function': activation,
                    'Seed': seed,
                    'Top-1 Accuracy': '-'
                }, ignore_index=True)
        result.append(df_act)
    
    if not result:
        print(f"Warning: No valid data for dataset {dataset_name} after filtering.")
        return pd.DataFrame(columns=['Activation Function', 'Top-1 Accuracy', 'Dataset'])

    df_filtered = pd.concat(result, ignore_index=True)
    
    # Group by Activation Function and calculate median
    result = df_filtered.groupby('Activation Function')['Top-1 Accuracy'].agg(lambda x: np.median([float(i) for i in x if i != '-']) if '-' not in x else '-').reset_index()
    result['Dataset'] = dataset_name
    return result

# Process all datasets
all_results = []
for file in os.listdir('saves'):
    if file.endswith('_results.txt'):
        dataset_name = file.split('_')[0]
        if dataset_name in config:
            file_path = os.path.join('saves', file)
            result = process_dataset(file_path, dataset_name)
            if not result.empty:
                all_results.append(result)

if not all_results:
    print("Error: No valid data found for any dataset.")
    exit(1)

# Combine all results
combined_results = pd.concat(all_results, ignore_index=True)

# Find common activation functions
activation_functions = set.intersection(*[set(df['Activation Function']) for df in all_results])

if not activation_functions:
    print("Error: No common activation functions found across datasets.")
    exit(1)

# Filter for common activation functions
combined_results = combined_results[combined_results['Activation Function'].isin(activation_functions)]

# Pivot the table
pivot_table = combined_results.pivot(index='Activation Function', columns='Dataset', values='Top-1 Accuracy')

# Convert to percentage (only for numeric values)
pivot_table = pivot_table.applymap(lambda x: f'{float(x)*100:.2f}' if x != '-' else x)

# Generate LaTeX table
latex_table = tabulate(pivot_table, headers='keys', tablefmt='latex_raw')

# Print LaTeX table
print(latex_table)

# Optionally, save to a file
with open('consolidated_results.tex', 'w') as f:
    f.write(latex_table)
