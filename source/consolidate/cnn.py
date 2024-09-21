import argparse
import pandas as pd
from tabulate import tabulate

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
    grouped = df.groupby(['Model', 'Total Epochs', 'Learning Rate', 'Batch Size'])
    result = []
    for x, group in grouped:
        if len(group) != 5:

            raise ValueError(f"Exactly 5 seeds are required for each unique combination when --mo5 is 'y', not the case for {x}, {group}")
        median_row = group.sort_values('Top-1 Accuracy', ascending=False).iloc[2]
        result.append(median_row)
    return pd.DataFrame(result)

def main():
    parser = argparse.ArgumentParser(description='Filter and display MNIST results.')
    parser.add_argument('--task', type=str, help='mnist,fmnist,cifar10,cifar100,svhn')
    parser.add_argument('--model', type=str, help='Model name')
    parser.add_argument('--epochs', type=int, help='Total epochs')
    parser.add_argument('--batch-size', type=int, help='Batch size')
    parser.add_argument('--lr', type=float, help='Learning rate')
    parser.add_argument('--seed', type=int, help='Seed')
    parser.add_argument('--activation', type=str, help='Activation function')
    parser.add_argument('--mo5', type=str, default='n', help='Median of 5 best results (y/n)')
    args = parser.parse_args()

    # Read the CSV file using pandas
    df = pd.read_csv(f'./saves/{args.task}_results.txt')

    # Apply filters
    filtered_df = filter_rows(df, args)

    if args.mo5.lower() == 'y':
        if args.seed is not None:
            raise ValueError("Cannot specify a single seed when --mo5 is 'y'")
        try:
            filtered_df = get_median_of_5(filtered_df)
        except ValueError as e:
            print(f"Error: {str(e)}")
            return

    if not filtered_df.empty:
        # Convert DataFrame to a list of lists for tabulate
        table = filtered_df.values.tolist()
        headers = filtered_df.columns.tolist()
        print(tabulate(table, headers=headers, tablefmt='grid'))
    else:
        print("No matching rows found.")

if __name__ == '__main__':
    main()
