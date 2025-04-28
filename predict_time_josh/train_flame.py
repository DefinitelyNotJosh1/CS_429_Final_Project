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

start_time = time.time()

input_dimensions = 20  # Number of PCA features

# Load data
df = pd.read_csv("FIND_BEST_MODEL/PCA_20_flame_frame_data.csv", index_col=0)

# Features and labels
X = df.iloc[:, :-1].values  # PCA features (columns 0 to 99)
y = df['distance_to_flame_frame'].values  # Distance to flame frame

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
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

# Define neural network
class FlameFramePredictor(nn.Module):
    def __init__(self, input_dim=input_dimensions):
        super(FlameFramePredictor, self).__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 64)
        self.fc4 = nn.Linear(64, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        return x

# Initialize model, loss, and optimizer
model = FlameFramePredictor()
model.load_state_dict(torch.load("non-normal_flame_frame_predictor.pth"))
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.000000001)

# Function to compute accuracy (within 10 seconds)
def compute_accuracy(predictions, targets, tolerance=50):
    errors = torch.abs(predictions - targets)
    accurate = (errors <= tolerance).float()
    accuracy = accurate.mean().item() * 100  # Percentage
    return accuracy

# Training loop
num_epochs = 1000
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
        train_loss += loss.item() * X_batch.size(0)
        train_correct += compute_accuracy(outputs, y_batch, tolerance=100) * X_batch.size(0)
    train_loss /= len(train_loader.dataset)
    train_accuracy = train_correct / len(train_loader.dataset)
    train_losses.append(train_loss)
    train_accuracies.append(train_accuracy)

    # Validation
    model.eval()
    test_loss = 0
    test_correct = 0
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            test_loss += loss.item() * X_batch.size(0)
            test_correct += compute_accuracy(outputs, y_batch, tolerance=100) * X_batch.size(0)
    test_loss /= len(test_loader.dataset)
    test_accuracy = test_correct / len(test_loader.dataset)
    test_losses.append(test_loss)
    test_accuracies.append(test_accuracy)
    print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%, Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%")

# Save model
torch.save(model.state_dict(), "non-normal_flame_frame_predictor.pth")
print("Model saved to non-normal_flame_frame_predictor.pth")

# Plot and save loss graph
plt.figure(figsize=(10, 5))
plt.plot(range(1, num_epochs + 1), train_losses, label='Train Loss')
plt.plot(range(1, num_epochs + 1), test_losses, label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss (MSE)')
plt.title('Training and Test Loss Over Epochs')
plt.legend()
plt.grid(True)
plt.savefig('video1_non-normal_nn_loss_plot.png')
plt.close()

# Plot and save accuracy graph
plt.figure(figsize=(10, 5))
plt.plot(range(1, num_epochs + 1), train_accuracies, label='Train Accuracy')
plt.plot(range(1, num_epochs + 1), test_accuracies, label='Test Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.title('Training and Test Accuracy Over Epochs')
plt.legend()
plt.grid(True)
plt.savefig('video1_non-normal_nn_accuracy_plot.png')
plt.close()


end_time = time.time()
print(f"Time taken: {end_time - start_time:.2f} seconds or {(end_time - start_time)/60:.2f} minutes")
print("Training complete.")