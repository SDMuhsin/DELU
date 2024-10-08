
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
    if args.activation in ['DELU','ADELU']:
        activation_name += f"_a{args.a}_b{args.b}"
    elif args.activation == 'FADELU':
        activation_name += f"_a{args.a}_b{args.b}_c{args.c}_d{args.d}"
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
    if args.activation in ['DELU','ADELU']:
        activation_name += f"_a{args.a}_b{args.b}"
    elif args.activation == 'FADELU':
        activation_name += f"_a{args.a}_b{args.b}_c{args.c}_d{args.d}"

    # Read the CSV file into a DataFrame
    df = pd.read_csv(file_path)

    # Define the configuration to check
    config_to_check = pd.Series({
        'Model': args.model,
        'Batch Size': args.batch_size,
        'Total Epochs': args.epochs,
        'Learning Rate': args.lr,
        'Activation Function': activation_name,
        'Seed': args.seed
    })

    # Check if the configuration exists
    matching_configs = (df[config_to_check.index] == config_to_check).all(axis=1)
    
    if matching_configs.any():
        matching_row = df[matching_configs].iloc[0]
        print(f"Configuration exists: {matching_row.to_dict()}")
        return True

    return False
