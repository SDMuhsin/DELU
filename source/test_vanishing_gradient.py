import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

# Set random seed for reproducibility
torch.manual_seed(42)

# Generate toy dataset
def generate_toy_dataset(num_samples=1000, input_size=32):
    X = torch.randn(num_samples, 1, input_size, input_size)
    y = torch.zeros(num_samples)
    y[X.mean(dim=(1, 2, 3)) > 0] = 1
    return X, y

# Define the CNN model with GELU activation
class ToyCNN(nn.Module):
    def __init__(self):
        super(ToyCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.gelu1 = nn.GELU()
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.gelu2 = nn.GELU()
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.gelu3 = nn.GELU()
        self.fc = nn.Linear(64 * 32 * 32, 1)
        
    def forward(self, x):
        x = self.gelu1(self.conv1(x))
        x = self.gelu2(self.conv2(x))
        x = self.gelu3(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x.squeeze()

# Generate dataset
X_train, y_train = generate_toy_dataset()
X_test, y_test = generate_toy_dataset(num_samples=200)

# Create DataLoaders
train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Initialize model, loss function, and optimizer
model = ToyCNN()
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 50
train_losses = []
grad_norms = []

for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    epoch_grad_norm = 0
    
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        
        # Calculate gradient norm
        total_norm = 0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        
        optimizer.step()
        
        epoch_loss += loss.item()
        epoch_grad_norm += total_norm
    
    avg_loss = epoch_loss / len(train_loader)
    avg_grad_norm = epoch_grad_norm / len(train_loader)
    train_losses.append(avg_loss)
    grad_norms.append(avg_grad_norm)
    
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, Grad Norm: {avg_grad_norm:.4f}")

# Plot training loss and gradient norm
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(train_losses)
plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.subplot(1, 2, 2)
plt.plot(grad_norms)
plt.title("Gradient Norm")
plt.xlabel("Epoch")
plt.ylabel("Norm")

plt.tight_layout()
plt.show()

# Evaluate the model
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for batch_X, batch_y in test_loader:
        outputs = model(batch_X)
        predicted = (outputs > 0).float()
        total += batch_y.size(0)
        correct += (predicted == batch_y).sum().item()

accuracy = 100 * correct / total
print(f"Test Accuracy: {accuracy:.2f}%")
