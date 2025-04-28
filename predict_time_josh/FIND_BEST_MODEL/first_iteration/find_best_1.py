import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import time
import os
import uuid

start_time = time.time()

input_dimensions = 20  # Number of PCA features

# Load and preprocess data
df = pd.read_csv("../PCA_20_flame_frame_data.csv", index_col=0)

X = df.iloc[:, :-1].values
y = df['distance_to_flame_frame'].values

# Split and standardize data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

# Create DataLoader
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Define neural network with variable architecture
class FlameFramePredictor(nn.Module):
    def __init__(self, input_dim, layer_sizes):
        super(FlameFramePredictor, self).__init__()
        layers = []
        prev_dim = input_dim
        for size in layer_sizes:
            layers.append(nn.Linear(prev_dim, size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.3))
            prev_dim = size
        layers.append(nn.Linear(prev_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

# Function to compute accuracy
def compute_accuracy(predictions, targets, tolerance=50):
    errors = torch.abs(predictions - targets)
    accurate = (errors <= tolerance).float()
    accuracy = accurate.mean().item() * 100
    return accuracy

# Architectures to test
architectures = [
    [64, 32],              # Small
    [128, 64, 32],         # Medium
    [256, 128, 64],        # Medium-Large
    [512, 256, 64],        # Large
    [1024, 512, 256, 64],  # Extra Large
]

# Training function
def train_model(model, train_loader, test_loader, num_epochs=100):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.003)
    train_losses = []
    test_losses = []
    train_accuracies = []
    test_accuracies = []

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        train_correct = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.detach().item() * X_batch.size(0)
            train_correct += compute_accuracy(outputs, y_batch, tolerance=100) * X_batch.size(0)
        train_loss /= len(train_loader.dataset)
        train_accuracy = train_correct / len(train_loader.dataset)
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)

        model.eval()
        test_loss = 0
        test_correct = 0
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                test_loss += loss.detach().item() * X_batch.size(0)
                test_correct += compute_accuracy(outputs, y_batch, tolerance=100) * X_batch.size(0)
        test_loss /= len(test_loader.dataset)
        test_accuracy = test_correct / len(test_loader.dataset)
        test_losses.append(test_loss)
        test_accuracies.append(test_accuracy)

    return train_losses, test_losses, train_accuracies, test_accuracies

# Test architectures and track results
results = []
for arch in architectures:
    print(f"\nTesting architecture: {arch}")
    model = FlameFramePredictor(input_dimensions, arch)
    train_losses, test_losses, train_accuracies, test_accuracies = train_model(model, train_loader, test_loader)
    
    # Save model
    model_id = str(uuid.uuid4())
    torch.save(model.state_dict(), f"flame_frame_predictor_{model_id}.pth")
    
    # Store results
    results.append({
        'architecture': arch,
        'model_id': model_id,
        'final_test_accuracy': test_accuracies[-1],
        'final_test_loss': test_losses[-1],
        'train_losses': train_losses,
        'test_losses': test_losses,
        'train_accuracies': train_accuracies,
        'test_accuracies': test_accuracies
    })

# Find best model
best_model = max(results, key=lambda x: x['final_test_accuracy'])
print("\nBest Model:")
print(f"Architecture: {best_model['architecture']}")
print(f"Test Accuracy: {best_model['final_test_accuracy']:.2f}%")
print(f"Test Loss: {best_model['final_test_loss']:.4f}")
print(f"Model saved as: flame_frame_predictor_{best_model['model_id']}.pth")

# Plot results for best model
best_architecture = best_model['architecture']

plt.figure(figsize=(10, 5))
plt.plot(range(1, 101), best_model['train_losses'], label='Train Loss')
plt.plot(range(1, 101), best_model['test_losses'], label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss (MSE)')
plt.title(f'Best Model Loss (Architecture: {best_architecture})')
plt.legend()
plt.grid(True)
plt.savefig('best_model_loss_plot.png')
plt.close()

plt.figure(figsize=(10, 5))
plt.plot(range(1, 101), best_model['train_accuracies'], label='Train Accuracy')
plt.plot(range(1, 101), best_model['test_accuracies'], label='Test Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.title(f'Best Model Accuracy (Architecture: {best_architecture})')
plt.legend()
plt.grid(True)
plt.savefig('best_model_accuracy_plot.png')
plt.close()

# Save summary of all results
plt.figure(figsize=(12, 6))
for result in results:
    plt.plot(range(1, 101), result['test_accuracies'], label=f"Arch: {result['architecture']}")
plt.xlabel('Epoch')
plt.ylabel('Test Accuracy (%)')
plt.title('Test Accuracy for All Architectures')
plt.legend()
plt.grid(True)
plt.savefig('all_architectures_accuracy_plot.png')
plt.close()

end_time = time.time()
print(f"\nTotal time taken: {end_time - start_time:.2f} seconds or {(end_time - start_time)/60:.2f} minutes")
print("Architecture testing complete.")