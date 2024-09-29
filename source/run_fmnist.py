import argparse
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from common.DELU import DELU
from common.utility import replace_activations
from common.utility import get_activation_by_name
import csv
from common.cnn_csv_utils import * 
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_data(data_dir, batch_size):
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.Grayscale(3),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_dataset = datasets.FashionMNIST(data_dir, train=True, download=True, transform=transform)
    test_dataset = datasets.FashionMNIST(data_dir, train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

def get_model(model_name):
    if model_name == 'resnet18':
        model = models.resnet18(pretrained=False)
    elif model_name == 'vgg16':
        model = models.vgg16(pretrained=False)
    elif model_name == 'densenet121':
        model = models.densenet121(pretrained=False)
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    # Modify the last layer for Fashion MNIST (10 classes)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 10)

    return model

def train(model, train_loader, optimizer, criterion, device):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

def evaluate(model, test_loader, device):
    model.eval()
    all_preds = []
    all_targets = []
    all_top5_preds = []

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            _, top5_predicted = torch.topk(output.data, 5, dim=1)
            all_preds.extend(predicted.cpu().numpy())
            all_top5_preds.extend(top5_predicted.cpu().numpy())
            all_targets.extend(target.cpu().numpy())

    top1_accuracy = accuracy_score(all_targets, all_preds)
    top5_accuracy = sum(target in pred for target, pred in zip(all_targets, all_top5_preds)) / len(all_targets)
    precision = precision_score(all_targets, all_preds, average='weighted')
    recall = recall_score(all_targets, all_preds, average='weighted')
    f1 = f1_score(all_targets, all_preds, average='weighted')

    return top1_accuracy, top5_accuracy, precision, recall, f1

def main(args):

    if(configuration_exists(args)):
        exit()
    
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, test_loader = load_data(args.data_dir, args.batch_size)
    model = get_model(args.model).to(device)

    activation = get_activation_by_name(args.activation,float(args.a),float(args.b),args.c,args.d)

    replace_activations(model, nn.ReLU, activation)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    best_top1_accuracy = 0
    best_results = None
    best_epoch = 0

    for epoch in range(1, args.epochs + 1):
        train(model, train_loader, optimizer, criterion, device)
        top1_accuracy, top5_accuracy, precision, recall, f1 = evaluate(model, test_loader, device)

        print(f"Epoch {epoch}/{args.epochs}")
        print(f"Top-1 Accuracy: {top1_accuracy:.4f}")
        print(f"Top-5 Accuracy: {top5_accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-score: {f1:.4f}")
        print()

        if top1_accuracy > best_top1_accuracy:
            best_top1_accuracy = top1_accuracy
            best_results = [args.model, args.epochs, args.activation, args.batch_size, args.seed, args.lr, 
                            epoch, top1_accuracy, top5_accuracy, precision, recall, f1]
            best_epoch = epoch

    # Save results
    os.makedirs('./saves', exist_ok=True)
    file_path = './saves/fmnist_results.txt'

    save_results(args, best_epoch, best_top1_accuracy, top5_accuracy, precision, recall, f1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Fashion MNIST CNN Training')
    parser.add_argument('--model', type=str, default='resnet18', choices=['resnet18', 'vgg16', 'densenet121'],
                        help='model architecture')
    parser.add_argument('--data-dir', type=str, default='./data', help='data directory')
    parser.add_argument('--batch-size', type=int, default=64, help='input batch size for training')
    parser.add_argument('--epochs', type=int, default=10, help='number of epochs to train')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--seed', type=int, default=41, help='random seed')
    parser.add_argument('--activation', type=str, default='ReLU',choices=['FADELU','ADELU','ReLU', 'LeakyReLU', 'ELU', 'SELU', 'GELU', 'Tanh', 'Sigmoid','Hardswish', 'Mish', 'SiLU', 'Softplus', 'Softsign', 'Hardshrink','Softshrink', 'Tanhshrink', 'PReLU', 'RReLU', 'CELU', 'Hardtanh','DELU'],help='Activation function to use in the model')
    parser.add_argument('--task',type=str)  
    parser.add_argument('--a',type=float,default=1)
    parser.add_argument('--b',type=float,default=1)
    parser.add_argument('--c',type=float,default=1)
    parser.add_argument('--d',type=float,default=1)
    args = parser.parse_args()
    
    main(args)
