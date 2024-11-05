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
from common.utility import get_model
from common.cnn_csv_utils import *
from torch.utils.data import Subset
import csv
import time

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
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(data_dir, train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


class SmallNet(nn.Module):
    def __init__(self, activation_func='ReLU'):
        super(SmallNet, self).__init__()

        # Dictionary of available activation functions (unchanged)
        self.activation_functions = {
            'ReLU': nn.ReLU(),
            'LeakyReLU': nn.LeakyReLU(),
            'ELU': nn.ELU(),
            'SELU': nn.SELU(),
            'GELU': nn.GELU(),
            'Tanh': nn.Tanh(),
            'Sigmoid': nn.Sigmoid(),
            'Hardswish': nn.Hardswish(),
            'Mish': nn.Mish(),
            'SiLU': nn.SiLU(),  # Also known as Swish
            'Softplus': nn.Softplus(),
            'Softsign': nn.Softsign(),
            'Hardshrink': nn.Hardshrink(),
            'Softshrink': nn.Softshrink(),
            'Tanhshrink': nn.Tanhshrink(),
            'PReLU': nn.PReLU(),
            'RReLU': nn.RReLU(),
            'CELU': nn.CELU(),
            'Hardtanh': nn.Hardtanh(),
        }

        # Select the activation function
        self.activation = self.activation_functions[activation_func]

        # Modified first layer to accept 3 input channels
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            self.activation,
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Adjust the second layer to handle the larger input size
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            self.activation,
            nn.MaxPool2d(2)
        )

        # Adjust the fully connected layers for the new input size
        self.fc1 = nn.Linear(in_features=64*56*56, out_features=256)
        self.drop = nn.Dropout2d(0.25)
        self.fc2 = nn.Linear(in_features=256, out_features=64)
        self.classifier = nn.Linear(in_features=64, out_features=10)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.drop(out)
        out = self.fc2(out)
        out = self.classifier(out)
        return out  # Make sure to return the output




def train(model, train_loader, optimizer, criterion, device, epoch, total_epochs):
    model.train()
    start_time = time.time()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        elapsed_time = time.time() - start_time
        progress = (batch_idx + 1) / len(train_loader)
        estimated_total_time = elapsed_time / progress
        remaining_time = estimated_total_time - elapsed_time
        
        print(f"\rEpoch {epoch}/{total_epochs} - Batch {batch_idx+1}/{len(train_loader)} - "
              f"Est. time remaining: {remaining_time:.2f}s", end="")

    print()
def train_old(model, train_loader, optimizer, criterion, device):
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

    #if(configuration_exists(args)):
        #exit()
    #    pass

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, test_loader = load_data(args.data_dir, args.batch_size)
    #train_loader = torch.utils.data.DataLoader(list(train_loader)[:100], batch_size=train_loader.batch_size, shuffle=True)

    model = get_model(args.model,args.task).to(device)
    activation = get_activation_by_name(args.activation,float(args.a),float(args.b),float(args.c),float(args.d))
    replace_activations(model, nn.ReLU, activation)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    best_top1_accuracy = 0
    best_results = None
    best_epoch = 0

    for epoch in range(1, args.epochs + 1):
        train(model, train_loader, optimizer, criterion, device, epoch, args.epochs)
        #train(model, train_loader, optimizer, criterion, device)
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

    ''' Evaluate Model with GELU'''
    original_results = evaluate(model, test_loader, device)
    print("Before any approximation",original_results)
    
    # Analytical GELU approximation
    agelua = get_activation_by_name("AGELUA",float(args.a),float(args.b))
    replace_activations(model,type(activation),agelua)
    agelua_results = evaluate(model, test_loader, device)
    print("After analytical approximation",agelua_results)


    hgelua = get_activation_by_name("HGELUA",float(args.a),float(args.b))
    replace_activations(model,type(agelua),hgelua)
    hgelua_results = evaluate(model, test_loader, device)
    print("After hardware approximation",hgelua_results)

    exit()
    # Save results
    #save_results(args, best_epoch, best_top1_accuracy, top5_accuracy, precision, recall, f1)

    import copy
    import json
 
    trained_model_copy = copy.deepcopy(model)

    # PWL stuff
    print(f"Before PWL")
    original_results = evaluate(model, test_loader, device)
    print(original_results)

    # Step 2: Create PWL approximation and replace activations
    activation_func = get_activation_by_name(args.activation, args.a, args.b, args.c, args.d)
    pwl_approximation = PWLApproximation(args.pwl_segments, activation_func)
    replace_with_pwl(model, pwl_approximation)

    print(f"After PWL")
    pwl_results = evaluate(model, test_loader, device)
    print(pwl_results)

    # Step 3: Fine-tune the model with PWL approximation
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    #for epoch in range(1, 11):  # 10 epochs for fine-tuning
    for epoch in range(1, args.progressive_epochs):
        train(model, train_loader, optimizer, criterion, device, epoch, args.progressive_epochs)
    finetuned_results = evaluate(model, test_loader, device)
    print(f"After fine tuning")
    print(finetuned_results)

    # Step 4: Progressive replacement experiment
    model = trained_model_copy#get_model(args.model, args.task).to(device)  # Reset the model
    replace_activations(model, nn.ReLU, activation_func)  # Reset to original activation
    optimizer = optim.Adam(model.parameters(), lr=args.lr)  # Reset optimizer
    progressive_results = progressive_replacement_experiment(model, train_loader, test_loader, optimizer, criterion, device, args, pwl_approximation,train,evaluate)
    print(f"After progressive fine tuning")
    print(progressive_results)
    # Save results to JSON
    save_results_to_json(args, original_results, pwl_results, finetuned_results, progressive_results)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MNIST CNN Training')
    parser.add_argument('--model', type=str, default='resnet18', 
                        help='model architecture')
    parser.add_argument('--data-dir', type=str, default='./data', help='data directory')
    parser.add_argument('--batch-size', type=int, default=64, help='input batch size for training')
    parser.add_argument('--epochs', type=int, default=10, help='number of epochs to train')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--seed', type=int, default=41, help='random seed')
    parser.add_argument('--activation', type=str, default='ReLU')
    parser.add_argument('--a',type=float,default=1)
    parser.add_argument('--b',type=float,default=1)
    parser.add_argument('--c',type=float,default=1)
    parser.add_argument('--d',type=float,default=1)
    parser.add_argument('--task',type=str)
    from common.approx import *
    parser.add_argument('--pwl_segments', type=int, default=4, help='number of segments for PWL approximation')
    parser.add_argument('--progressive_epochs', type=int, default=4, help='number of epochs for progressive replacement')
    args = parser.parse_args()
    main(args)
