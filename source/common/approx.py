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
from common.utility import replace_activations, get_activation_by_name, get_model
from common.cnn_csv_utils import *
import json
import time

class PWLApproximation(nn.Module):
    def __init__(self, segments, activation_func):
        super(PWLApproximation, self).__init__()
        self.segments = segments
        self.activation_func = activation_func
        self.breakpoints = torch.linspace(-4, 4, segments + 1)
        self.slopes = nn.Parameter(torch.randn(segments), requires_grad=False)
        self.intercepts = nn.Parameter(torch.randn(segments), requires_grad=False)
        self.initialize_parameters()

    def initialize_parameters(self):
        with torch.no_grad():
            x = torch.linspace(-4, 4, 1000)
            y = self.activation_func(x)
            for i in range(self.segments):
                mask = (x >= self.breakpoints[i]) & (x < self.breakpoints[i+1])
                x_segment = x[mask]
                y_segment = y[mask]
                if len(x_segment) > 1:
                    slope, intercept = np.polyfit(x_segment.numpy(), y_segment.numpy(), 1)
                    self.slopes[i] = slope
                    self.intercepts[i] = intercept

    def forward(self, x):
        output = torch.zeros_like(x)
        for i in range(self.segments):
            mask = (x >= self.breakpoints[i]) & (x < self.breakpoints[i+1])
            output[mask] = self.slopes[i] * x[mask] + self.intercepts[i]
        output[x < self.breakpoints[0]] = self.slopes[0] * x[x < self.breakpoints[0]] + self.intercepts[0]
        output[x >= self.breakpoints[-1]] = self.slopes[-1] * x[x >= self.breakpoints[-1]] + self.intercepts[-1]
        return output
def replace_with_pwl(model, pwl_approximation):
    for name, module in model.named_children():
        if isinstance(module, (nn.ReLU, nn.LeakyReLU, nn.ELU, nn.SELU, nn.GELU, DELU)):
            setattr(model, name, pwl_approximation)
        else:
            replace_with_pwl(module, pwl_approximation)
def count_activations(model):
    count = 0
    for module in model.modules():
        if isinstance(module, (nn.ReLU, nn.LeakyReLU, nn.ELU, nn.SELU, nn.GELU, DELU)):
            count += 1
    return count
def replace_activations_progressively(model, original_activation, new_activation, percentage):
    count = 0
    total = count_activations(model)
    target = int(total * percentage)

    for name, module in model.named_modules():
        if isinstance(module, original_activation):
            if count < target:
                setattr(model, name.split('.')[-1], new_activation)
                count += 1
            else:
                break

def progressive_replacement_experiment(model, train_loader, test_loader, optimizer, criterion, device, args, pwl_approximation,train,evaluate):
    total_activations = count_activations(model)
    original_activation = type(get_activation_by_name(args.activation, args.a, args.b, args.c, args.d))
    
    progressive_results = []
    
    for epoch in range(1, args.progressive_epochs + 1):
        percentage = epoch / args.progressive_epochs
        replace_activations_progressively(model, original_activation, pwl_approximation, percentage)
        
        train(model, train_loader, optimizer, criterion, device, epoch, args.progressive_epochs)
        results = evaluate(model, test_loader, device)
        
        progressive_results.append({
            "epoch": epoch,
            "percentage_replaced": percentage * 100,
            "results": results
        })
        
        print(f"Progressive Replacement - Epoch {epoch}/{args.progressive_epochs}")
        print(f"Percentage of activations replaced: {percentage * 100:.2f}%")
        print(f"Top-1 Accuracy: {results[0]:.4f}")
        print(f"Top-5 Accuracy: {results[1]:.4f}")
        print()

    return progressive_results

def save_results_to_json(args, original_results, pwl_results, finetuned_results, progressive_results):
    results = {
        "model": args.model,
        "activation": args.activation,
        "pwl_segments": args.pwl_segments,
        "original_results": original_results,
        "pwl_results": pwl_results,
        "finetuned_results": finetuned_results,
        "progressive_results": progressive_results
    }
    
    filename = f"./saves/{args.model}_{args.activation}_{args.pwl_segments}_segments.json"
    with open(filename, 'w') as f:
        json.dump(results, f, indent=4)
