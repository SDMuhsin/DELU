
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


