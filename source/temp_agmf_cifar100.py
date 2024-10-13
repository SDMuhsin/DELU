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
from common.utility import get_model
from common.utility import get_activation_by_name
from common.cnn_csv_utils import * 
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
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    ])

    train_dataset = datasets.CIFAR100(data_dir, train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR100(data_dir, train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader
import time
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

class AGMFOptimizer(optim.Optimizer):
    def __init__(self, params, lr=0.001, beta=0.9, window_size=5, freeze_ratio=0.05):
        defaults = dict(lr=lr, beta=beta, window_size=window_size)
        super(AGMFOptimizer, self).__init__(params, defaults)
        
        self.gradient_momentum = {}
        self.momentum_history = {}
        self.freeze_ratio = freeze_ratio
        for group in self.param_groups:
            for p in group['params']:
                self.gradient_momentum[p] = 0
                self.momentum_history[p] = []

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad.data
                state = self.state[p]

                # Update gradient momentum
                if 'momentum_buffer' not in state:
                    buf = state['momentum_buffer'] = torch.zeros_like(p.data)
                else:
                    buf = state['momentum_buffer']
                buf.mul_(group['beta']).add_(grad, alpha=1 - group['beta'])
                
                # Update momentum history
                self.gradient_momentum[p] = buf.norm().item()
                self.momentum_history[p].append(self.gradient_momentum[p])
                if len(self.momentum_history[p]) > group['window_size']:
                    self.momentum_history[p].pop(0)

                # Apply gradients
                p.data.add_(grad, alpha=-group['lr'])

        return loss

    def get_momentum_stability(self, param):
        history = self.momentum_history[param]
        if len(history) < 2:
            return float('inf')
        return np.var(history)

    def freeze_params(self):
        all_params = list(self.param_groups[0]['params'])
        num_params_to_freeze = int(len(all_params) * self.freeze_ratio)
        
        # Sort parameters by momentum stability
        sorted_params = sorted(all_params, key=lambda p: self.get_momentum_stability(p))
        
        # Freeze the most stable parameters
        for param in sorted_params[:num_params_to_freeze]:
            param.requires_grad = False

def train(model, train_loader, optimizer, criterion, device, epoch, total_epochs):
    model.train()
    start_time = time.time()
    
    # Freeze a portion of the model at the start of each epoch
    optimizer.freeze_params()
    
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

def main(args):
    #if(configuration_exists(args)):
        #exit()
    #    pass
    set_seed(args.seed)
    print(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, test_loader = load_data(args.data_dir, args.batch_size)
    model = get_model(args.model,args.task).to(device)

    activation = get_activation_by_name(args.activation, float(args.a) , float(args.b),args.c,args.d)

    replace_activations(model, nn.ReLU, activation)

    optimizer = AGMFOptimizer(model.parameters(), lr=args.lr, beta=0.9, window_size=5, freeze_ratio=args.freeze_ratio)
    criterion = nn.CrossEntropyLoss()

    best_top1_accuracy = 0
    best_results = None
    best_epoch = 0
    
    st = time.time()
    for epoch in range(1, args.epochs + 1):
        train(model, train_loader, optimizer, criterion, device, epoch, args.epochs)
        top1_accuracy, top5_accuracy, precision, recall, f1 = evaluate(model, test_loader, device)

        print(f"\nEpoch {epoch}/{args.epochs}")
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
    et = time.time()
    tt_hr  = (et - st) / 60 
    print(f"That took {tt_hr} minutes to train")

    # Save results
    #save_results(args, best_epoch, best_top1_accuracy, top5_accuracy, precision, recall, f1)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CIFAR-100 CNN Training')
    parser.add_argument('--model', type=str, default='resnet18', choices=['resnet18', 'vgg16', 'densenet121','shufflenet','mobilenetv2','resnet34','resnet50'],
                        help='model architecture')
    parser.add_argument('--data-dir', type=str, default='./data', help='data directory')
    parser.add_argument('--batch-size', type=int, default=64, help='input batch size for training')
    parser.add_argument('--epochs', type=int, default=10, help='number of epochs to train')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--seed', type=int, default=41, help='random seed')
    parser.add_argument('--activation', type=str, default='ReLU',choices=['FADELU','ADELU','ReLU', 'LeakyReLU', 'ELU', 'SELU', 'GELU', 'Tanh', 'Sigmoid','Hardswish', 'Mish', 'SiLU', 'Softplus', 'Softsign', 'Hardshrink','Softshrink', 'Tanhshrink', 'PReLU', 'RReLU', 'CELU', 'Hardtanh','DELU'],help='Activation function to use in the model')
    parser.add_argument('--a',type=float,default=1)
    parser.add_argument('--b',type=float,default=1)
    parser.add_argument('--c',type=float,default=1)
    parser.add_argument('--d',type=float,default=1)
    parser.add_argument('--task',type=str)
    parser.add_argument('--freeze-ratio',type=float,default=0.03) 
    args = parser.parse_args()
    main(args)