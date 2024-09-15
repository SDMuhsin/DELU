import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random

import torch
import torch.nn as nn
from torch.autograd import Variable

import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix
import argparse
import json
import os

seed = 41
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

np.random.seed(42)
random.seed(42)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

g = torch.Generator()
g.manual_seed(seed)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

train_set = torchvision.datasets.FashionMNIST("./data", download=True, transform=transforms.Compose([transforms.ToTensor()]))
test_set = torchvision.datasets.FashionMNIST("./data", download=True, train=False, transform=transforms.Compose([transforms.ToTensor()])) 

train_loader = torch.utils.data.DataLoader(train_set, batch_size=100)
test_loader = torch.utils.data.DataLoader(test_set,batch_size=100)

def output_label(label):
    output_mapping = {
                 0: "T-shirt/Top",
                 1: "Trouser",
                 2: "Pullover",
                 3: "Dress",
                 4: "Coat", 
                 5: "Sandal", 
                 6: "Shirt",
                 7: "Sneaker",
                 8: "Bag",
                 9: "Ankle Boot"
                 }
    input = (label.item() if type(label) == torch.Tensor else label)
    return output_mapping[input]

parser = argparse.ArgumentParser(description='FashionCNN with customizable activation function')
parser.add_argument('--activation', type=str, default='ReLU',choices=['ReLU', 'LeakyReLU', 'ELU', 'SELU', 'GELU', 'Tanh', 'Sigmoid','Hardswish', 'Mish', 'SiLU', 'Softplus', 'Softsign', 'Hardshrink','Softshrink', 'Tanhshrink', 'PReLU', 'RReLU', 'CELU', 'Hardtanh'],help='Activation function to use in the model')

args = parser.parse_args()

class FashionCNN(nn.Module):
    def __init__(self, activation_func='ReLU'):
        super(FashionCNN, self).__init__()
        
        # Dictionary of available activation functions
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

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            self.activation,
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
            nn.BatchNorm2d(64),
            self.activation,
            nn.MaxPool2d(2)
        )

        self.fc1 = nn.Linear(in_features=64*6*6, out_features=600)
        self.drop = nn.Dropout2d(0.25)
        self.fc2 = nn.Linear(in_features=600, out_features=120)
        self.fc3 = nn.Linear(in_features=120, out_features=10)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.drop(out)
        out = self.fc2(out)
        out = self.fc3(out)

        return out
model = FashionCNN(activation_func=args.activation)
model.to(device)

error = nn.CrossEntropyLoss()

learning_rate = 0.001
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

num_epochs = 5
count = 0
# Lists for visualization of loss and accuracy 
loss_list = []
iteration_list = []
accuracy_list = []

# Lists for knowing classwise accuracy
predictions_list = []
labels_list = []

best_accuracy = 0
for epoch in range(num_epochs):
    for images, labels in train_loader:
        # Transfering images and labels to GPU if available
        images, labels = images.to(device), labels.to(device)
    
        train = Variable(images.view(100, 1, 28, 28))
        labels = Variable(labels)
        
        # Forward pass 
        outputs = model(train)
        loss = error(outputs, labels)
        
        # Initializing a gradient as 0 so there is no mixing of gradient among the batches
        optimizer.zero_grad()
        
        #Propagating the error backward
        loss.backward()
        
        # Optimizing the parameters
        optimizer.step()
    
        count += 1
    
    # Testing the model
    
        if not (count % 50):    # It's same as "if count % 50 == 0"
            total = 0
            correct = 0
        
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                labels_list.append(labels)
            
                test = Variable(images.view(100, 1, 28, 28))
            
                outputs = model(test)
            
                predictions = torch.max(outputs, 1)[1].to(device)
                predictions_list.append(predictions)
                correct += (predictions == labels).sum()
            
                total += len(labels)
            
            accuracy = correct * 100 / total
            loss_list.append(loss.data)
            iteration_list.append(count)
            accuracy_list.append(accuracy)
        
        if not (count % 500):
            print("Iteration: {}, Loss: {}, Accuracy: {}%".format(count, loss.data, accuracy))
best_accuracy = max(accuracy_list)

# Save data
save_file_name = f"./saves/baseline_fashionmnist.json"
# Ensure the directory exists
os.makedirs(os.path.dirname(save_file_name), exist_ok=True)

# Load existing data or create an empty dictionary
if os.path.exists(save_file_name):
    with open(save_file_name, 'r') as f:
        data = json.load(f)
else:
    data = {}

data[args.activation] = best_accuracy.item()
print(data)
print(best_accuracy)
print(args.activation)
with open(save_file_name, 'w') as f:
    json.dump(data, f, indent=4)

print(f"Data saved to {save_file_name}")
