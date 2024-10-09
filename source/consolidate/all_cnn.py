import argparse
import pandas as pd
import numpy as np
from tabulate import tabulate
import warnings

def filter_rows(df, args):
    conditions = [
        (args.model is None) | (df['Model'] == args.model),
        (args.epochs is None) | (df['Total Epochs'].astype(int) == args.epochs),
        (args.batch_size is None) | (df['Batch Size'].astype(int) == args.batch_size),
        (args.lr is None) | (df['Learning Rate'].astype(float) == args.lr),
        (args.seed is None) | (df['Seed'].astype(int) == args.seed),
        (args.activation is None) | (df['Activation Function'] == args.activation)
    ]
    return df[pd.concat(conditions, axis=1).all(axis=1)]

def get_median_of_5(df):
    grouped = df.groupby(['Model', 'Total Epochs', 'Learning Rate', 'Batch Size', 'Activation Function'])
    result = []
    for x, group in grouped:
        if len(group) != 5:
            warnings.warn(f"Exactly 5 seeds are required for each unique combination when --mo5 is 'y', not the case for {x}")
            median_row = group.sort_values('Top-1 Accuracy', ascending=False).iloc[len(group) // 2]
            median_row = median_row.fillna('-')
        else:
            median_row = group.sort_values('Top-1 Accuracy', ascending=False).iloc[2]
        result.append(median_row)
    return pd.DataFrame(result)

def process_dataset(task, args):
    df = pd.read_csv(f'./saves/{task}_results.txt')
    filtered_df = filter_rows(df, args)
    
    if args.mo5.lower() == 'y':
        if args.seed is not None:
            raise ValueError("Cannot specify a single seed when --mo5 is 'y'")
        filtered_df = get_median_of_5(filtered_df)
    
    return filtered_df

def aggregate_results(datasets, args):
    all_results = {}
    complete_cases = set()
    
    for dataset in datasets:
        results = process_dataset(dataset, args)
        all_results[dataset] = results
        
        if not results.empty:
            cases = set(zip(results['Model'], results['Total Epochs'], results['Learning Rate'], 
                            results['Batch Size'], results['Activation Function']))
            if not complete_cases:
                complete_cases = cases
            else:
                complete_cases = complete_cases.intersection(cases)
    
    aggregated_table = pd.DataFrame(index=pd.unique(pd.concat([df['Activation Function'] for df in all_results.values()])))
    
    for dataset, results in all_results.items():
        if not results.empty:
            for case in complete_cases:
                model, epochs, lr, batch_size, activation = case
                row = results[(results['Model'] == model) & 
                              (results['Total Epochs'] == epochs) & 
                              (results['Learning Rate'] == lr) & 
                              (results['Batch Size'] == batch_size) & 
                              (results['Activation Function'] == activation)]
                
                if not row.empty:
                    aggregated_table.loc[activation, f'{dataset}_Accuracy'] = row['Top-1 Accuracy'].values[0]
                    aggregated_table.loc[activation, f'{dataset}_F1'] = row['F1-score'].values[0]
                else:
                    warnings.warn(f"Incomplete data for {activation} in {dataset}")
    
    return aggregated_table

def main():
    parser = argparse.ArgumentParser(description='Filter and display results for multiple datasets.')
    parser.add_argument('--tasks', type=str, help='Comma-separated list of tasks/datasets')
    parser.add_argument('--model', type=str, default='resnet18', help='Model name')
    parser.add_argument('--epochs', type=int, help='Total epochs')
    parser.add_argument('--batch-size', type=int, help='Batch size')
    parser.add_argument('--lr', type=float, help='Learning rate')
    parser.add_argument('--seed', type=int, help='Seed')
    parser.add_argument('--activation', type=str, help='Activation function')
    parser.add_argument('--mo5', type=str, default='y', help='Median of 5 best results (y/n)')
    args = parser.parse_args()

    datasets = args.tasks.split(',')
    
    # Process each dataset separately
    for dataset in datasets:
        print(f"\nResults for {dataset}:")
        results = process_dataset(dataset, args)
        if not results.empty:
            print(tabulate(results, headers='keys', tablefmt='grid'))
        else:
            print("No matching rows found.")
    
    # Aggregate results
    print("\nAggregated Results:")
    aggregated_table = aggregate_results(datasets, args)
    print(tabulate(aggregated_table, headers='keys', tablefmt='grid'))

if __name__ == '__main__':
    main()
