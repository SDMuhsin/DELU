import argparse
import csv
from tabulate import tabulate

def filter_rows(row, args):
    conditions = [
        args.model is None or row['Model'] == args.model,
        args.epochs is None or int(row['Total Epochs']) == args.epochs,
        args.batch_size is None or int(row['Batch Size']) == args.batch_size,
        args.lr is None or float(row['Learning Rate']) == args.lr,
        args.seed is None or int(row['Seed']) == args.seed,
        args.activation is None or row['Activation Function'] == args.activation
    ]
    return all(conditions)

def main():
    parser = argparse.ArgumentParser(description='Filter and display MNIST results.')
    parser.add_argument('--task',type=str, help='mnist,fmnist,cifar10,cifar100,svhn')
    parser.add_argument('--model', type=str, help='Model name')
    parser.add_argument('--epochs', type=int, help='Total epochs')
    parser.add_argument('--batch-size', type=int, help='Batch size')
    parser.add_argument('--lr', type=float, help='Learning rate')
    parser.add_argument('--seed', type=int, help='Seed')
    parser.add_argument('--activation', type=str, help='Activation function')
    args = parser.parse_args()

    with open(f'./saves/{args.task}_results.txt', 'r') as f:
        reader = csv.DictReader(f)
        data = [row for row in reader if filter_rows(row, args)]

    if data:
        headers = data[0].keys()
        table = [[row[col] for col in headers] for row in data]
        print(tabulate(table, headers=headers, tablefmt='grid'))
    else:
        print("No matching rows found.")

if __name__ == '__main__':
    main()
