
# csv_utils.py

import pandas as pd
import os

# Define the header as a constant
HEADER = ['Model', 'Total Epochs', 'Activation Function', 'Batch Size', 'Seed', 'Learning Rate',
          'Best Epoch', 'Top-1 Accuracy', 'Top-5 Accuracy', 'Precision', 'Recall', 'F1-score']

# Define the columns to use for duplicate detection
ID_COLUMNS = ['Model', 'Total Epochs', 'Activation Function', 'Batch Size', 'Seed', 'Learning Rate']

def initialize_csv(file_path):
    """
    Initialize the CSV file with the header if it doesn't exist.
    
    Args:
    file_path (str): Path to the CSV file.
    """
    if not os.path.isfile(file_path):
        df = pd.DataFrame(columns=HEADER)
        df.to_csv(file_path, index=False)
        print(f"Initialized CSV file at {file_path}")
    else:
        print(f"CSV file already exists at {file_path}")

def update_results(file_path, new_result):
    """
    Update the CSV file with new results, overwriting if a matching entry exists.
    
    Args:
    file_path (str): Path to the CSV file.
    new_result (dict): Dictionary containing the new result data.
    """
    # Ensure all required keys are in new_result
    if not all(key in new_result for key in HEADER):
        raise ValueError("new_result is missing some required fields")

    # Create a DataFrame from the new result
    new_df = pd.DataFrame([new_result])

    # Read existing results
    if os.path.isfile(file_path):
        df = pd.read_csv(file_path)
    else:
        df = pd.DataFrame(columns=HEADER)

    # Update existing row or append new row
    df = pd.concat([df, new_df]).drop_duplicates(subset=ID_COLUMNS, keep='last').reset_index(drop=True)

    # Write updated results back to file
    df.to_csv(file_path, index=False)
    print(f"Updated results in {file_path}")

def save_results(args, best_epoch, best_top1_accuracy, top5_accuracy, precision, recall, f1):
    import os

    os.makedirs('./saves', exist_ok=True)
    file_path = f'./saves/{args.task}_results.txt'
    initialize_csv(file_path)

    activation_name = args.activation
    if args.activation == 'DELU':
        activation_name += f"_a{args.a}_b{args.b}"

    new_result = {
        'Model': args.model,
        'Total Epochs': args.epochs,
        'Activation Function': activation_name,
        'Batch Size': args.batch_size,
        'Seed': args.seed,
        'Learning Rate': args.lr,
        'Best Epoch': best_epoch,
        'Top-1 Accuracy': best_top1_accuracy,
        'Top-5 Accuracy': top5_accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-score': f1
    }

    update_results(file_path, new_result)

import os
import csv

def configuration_exists(args):
    file_path = f'./saves/{args.task}_results.txt'
    
    # If the file doesn't exist, return False
    if not os.path.exists(file_path):
        return False
    
    # Construct the activation name
    activation_name = args.activation
    if args.activation == 'DELU':
        activation_name += f"_a{args.a}_b{args.b}"
    
    # Define the key configuration elements
    config_key = (
        args.model,
        args.task,  # Assuming the task is always MNIST
        args.batch_size,
        args.epochs,
        args.lr,
        activation_name,
        args.seed
    )
    
    # Read the CSV file and check for matching configurations
    with open(file_path, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            row_key = (
                row['Model'],
                args.task,  # Assuming the task is always MNIST
                int(row['Batch Size']),
                int(row['Total Epochs']),
                float(row['Learning Rate']),
                row['Activation Function'],
                int(row['Seed'])
            )
            if row_key == config_key:
                print(f"Configuration exists : ", row_key)
                return True
    
    return False
