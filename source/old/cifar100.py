import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18
from torch.utils.data import DataLoader
import numpy as np
import random

seed = 41
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
num_epochs = 10
learning_rate = 0.001
batch_size = 128

# Data preprocessing and augmentation
transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
])

# CIFAR-100 dataset
train_dataset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
test_dataset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# Load pre-trained ResNet-18 and modify for 100 classes
class SmoothExponentialLinear(nn.Module):
    def forward(self, x):
        return x * torch.exp(-torch.exp(-x))


def replace_relu_with_sel(module):
    """
    Recursively replaces all ReLU activations in the model with SmoothExponentialLinear (SEL).
    Args:
        module (nn.Module): The PyTorch model or module where replacements are needed.
    """
    for name, child in module.named_children():
        if isinstance(child, nn.ReLU):
            # If the child is ReLU, replace it with SmoothExponentialLinear
            setattr(module, name, SmoothExponentialLinear())
        else:
            # Recursively apply to child modules
            replace_relu_with_sel(child)

model = resnet18(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, 100)
model = model.to(device)
replace_relu_with_sel(model)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training function
def train(model, loader, criterion, optimizer, epoch):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for i, (inputs, targets) in enumerate(loader):
        inputs, targets = inputs.to(device), targets.to(device)
        
        # Zero the parameter gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        if (i + 1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(loader)}], Loss: {running_loss/100:.4f}, Accuracy: {100.*correct/total:.2f}%')
            running_loss = 0.0

# Test/Validation function
def test(model, loader, criterion):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item()
            
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    print(f'Test Loss: {test_loss/len(loader):.4f}, Accuracy: {100.*correct/total:.2f}%')

# Training the model
for epoch in range(num_epochs):
    train(model, train_loader, criterion, optimizer, epoch)
    test(model, test_loader, criterion)

# Saving the trained model
torch.save(model.state_dict(), 'resnet18_cifar100.pth')
print('Model saved as resnet18_cifar100.pth')

