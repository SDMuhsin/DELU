import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from common.DELU import DELU

# Generate a toy dataset prone to vanishing/exploding gradients
np.random.seed(0)
X = np.random.normal(size=(1000, 1)) * 5
Y = np.tanh(X)

# Define a simple model class
class ToyModel:
    def __init__(self, activation_fn):
        self.weights = [np.random.randn(1, 512) * 0.01]
        self.biases = [np.zeros(512)]
        for _ in range(5):
            self.weights.append(np.random.randn(512, 512) * 0.01)
            self.biases.append(np.zeros(512))
        self.weights.append(np.random.randn(512, 1) * 0.01)
        self.biases.append(np.zeros(1))
        self.activation_fn = activation_fn

    def forward(self, x):
        for w, b in zip(self.weights[:-1], self.biases[:-1]):
            x = self.activation_fn(np.dot(x, w) + b)
        return np.dot(x, self.weights[-1]) + self.biases[-1]

# Implement optimizers
class Optimizer:
    def __init__(self, model, learning_rate=0.001):
        self.model = model
        self.learning_rate = learning_rate

    def step(self, x, y):
        # Forward pass
        activations = [x]
        for w, b in zip(self.model.weights, self.model.biases):
            z = np.dot(activations[-1], w) + b
            activations.append(self.model.activation_fn(z))
        
        # Backward pass
        delta = activations[-1] - y
        for i in range(len(self.model.weights) - 1, -1, -1):
            self.model.weights[i] -= self.learning_rate * np.dot(activations[i].T, delta)
            self.model.biases[i] -= self.learning_rate * np.sum(delta, axis=0)
            if i > 0:
                delta = np.dot(delta, self.model.weights[i].T) * self.model.activation_fn(activations[i]) * (1 - self.model.activation_fn(activations[i]))

def train_and_evaluate(model, optimizer, X, Y, epochs=100):
    losses = []
    for _ in range(epochs):
        optimizer.step(X, Y)
        loss = np.mean((model.forward(X) - Y) ** 2)
        losses.append(loss)
    return losses

# Compare different activation functions
activation_functions = {
    'ReLU': lambda x: np.maximum(0, x),
    'Tanh': np.tanh,
    'DELU': DELU
}

results = {}

for name, activation in activation_functions.items():
    model = ToyModel(activation)
    optimizer = Optimizer(model)
    losses = train_and_evaluate(model, optimizer, X, Y)
    results[name] = losses

# Plot results
plt.figure(figsize=(10, 6))
for name, losses in results.items():
    plt.plot(losses, label=name)
plt.xlabel('Epochs')
plt.ylabel('Mean Squared Error')
plt.title('Comparison of Activation Functions on Toy Dataset')
plt.legend()
plt.yscale('log')
plt.show()
