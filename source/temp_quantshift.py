import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import time
import numpy as np

# QuantShift module (unchanged)
class QuantShift(nn.Module):
    def __init__(self, num_bits=8):
        super(QuantShift, self).__init__()
        self.num_bits = num_bits
        self.shift = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        x_freq = torch.fft.fft2(x)
        x_shifted = torch.fft.ifft2(x_freq * torch.exp(1j * self.shift))
        x = x_shifted.real
        scale = torch.max(torch.abs(x))
        x = x / scale
        x = torch.round(x * (2**self.num_bits - 1)) / (2**self.num_bits - 1)
        return x * scale

# QuantShiftCNN (unchanged)
class QuantShiftCNN(nn.Module):
    def __init__(self):
        super(QuantShiftCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.quant1 = QuantShift(num_bits=4)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.quant2 = QuantShift(num_bits=4)
        self.fc1 = nn.Linear(9216, 128)
        self.quant3 = QuantShift(num_bits=4)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.quant1(x)
        x = nn.functional.relu(x)
        x = self.conv2(x)
        x = self.quant2(x)
        x = nn.functional.relu(x)
        x = nn.functional.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.quant3(x)
        x = nn.functional.relu(x)
        x = self.fc2(x)
        return nn.functional.log_softmax(x, dim=1)

# Baseline CNN
class BaselineCNN(nn.Module):
    def __init__(self):
        super(BaselineCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = nn.functional.relu(x)
        x = self.conv2(x)
        x = nn.functional.relu(x)
        x = nn.functional.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = nn.functional.relu(x)
        x = self.fc2(x)
        return nn.functional.log_softmax(x, dim=1)

# Data loading (unchanged)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('./data', train=True, download=True, transform=transform),
    batch_size=64, shuffle=True)

test_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('./data', train=False, transform=transform),
    batch_size=1000, shuffle=True)

# Initialize models, loss function, and optimizers
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
quant_model = QuantShiftCNN().to(device)
baseline_model = BaselineCNN().to(device)
criterion = nn.CrossEntropyLoss()
quant_optimizer = optim.Adam(quant_model.parameters(), lr=0.001)
baseline_optimizer = optim.Adam(baseline_model.parameters(), lr=0.001)

# Training function
def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')

# Testing function
def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    print(f'\nTest set: Average loss: {test_loss:.4f}, '
          f'Accuracy: {correct}/{len(test_loader.dataset)} '
          f'({accuracy:.2f}%)\n')
    return accuracy

# Function to measure inference time
def measure_inference_time(model, device, test_loader):
    model.eval()
    start_time = time.time()
    with torch.no_grad():
        for data, _ in test_loader:
            data = data.to(device)
            _ = model(data)
    end_time = time.time()
    return end_time - start_time

# Function to measure memory usage
def measure_memory_usage(model, device, test_loader):
    model.eval()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    with torch.no_grad():
        for data, _ in test_loader:
            data = data.to(device)
            _ = model(data)
    return torch.cuda.max_memory_allocated() / 1024 / 1024  # Convert to MB

# Main training loop
num_epochs = 10
for epoch in range(1, num_epochs + 1):
    print("Training QuantShift model:")
    train(quant_model, device, train_loader, quant_optimizer, epoch)
    print("Training Baseline model:")
    train(baseline_model, device, train_loader, baseline_optimizer, epoch)

print("Training complete!")

# Evaluation and comparison
print("\nFinal Evaluation:")

quant_accuracy = test(quant_model, device, test_loader)
baseline_accuracy = test(baseline_model, device, test_loader)

quant_size = sum(p.numel() for p in quant_model.parameters() if p.requires_grad)
baseline_size = sum(p.numel() for p in baseline_model.parameters() if p.requires_grad)

quant_inference_time = measure_inference_time(quant_model, device, test_loader)
baseline_inference_time = measure_inference_time(baseline_model, device, test_loader)

quant_memory = measure_memory_usage(quant_model, device, test_loader)
baseline_memory = measure_memory_usage(baseline_model, device, test_loader)

print("\nComparison:")
print(f"Model Size (parameters): QuantShift: {quant_size}, Baseline: {baseline_size}")
print(f"Size Reduction: {(1 - quant_size/baseline_size)*100:.2f}%")

print(f"\nInference Time: QuantShift: {quant_inference_time:.4f}s, Baseline: {baseline_inference_time:.4f}s")
print(f"Speed Improvement: {(1 - quant_inference_time/baseline_inference_time)*100:.2f}%")

print(f"\nAccuracy: QuantShift: {quant_accuracy:.2f}%, Baseline: {baseline_accuracy:.2f}%")
print(f"Accuracy Difference: {quant_accuracy - baseline_accuracy:.2f}%")

print(f"\nMemory Usage: QuantShift: {quant_memory:.2f}MB, Baseline: {baseline_memory:.2f}MB")
print(f"Memory Reduction: {(1 - quant_memory/baseline_memory)*100:.2f}%")
