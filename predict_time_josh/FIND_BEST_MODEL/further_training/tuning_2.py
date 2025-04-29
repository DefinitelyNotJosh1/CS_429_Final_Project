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
EPOCHS = 200  # short bursts for tuning
ACCURACY_TOLERANCE = 5  # Tolerance for accuracy calculation (1 seconds)

# Load and preprocess data
df = pd.read_csv("../PCA_20_flame_frame_data.csv", index_col=0)
model_path = "tuned_flame_frame_predictor_EXPERIMENTAL.pth"  # Path to pre-trained model

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

# Define neural network - decided on best performing. Practically speaking,
# smaller models wouldn't make a difference in prediction time (2ms vs 0.2ms isn't much)
class FlameFramePredictor(nn.Module):
    def __init__(self, input_dim, layer_sizes=[2048, 1024, 512], dropout_rate=0):
        super(FlameFramePredictor, self).__init__()
        layers = []
        prev_dim = input_dim
        for size in layer_sizes:
            layers.append(nn.Linear(prev_dim, size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = size
        layers.append(nn.Linear(prev_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

# Function to compute accuracy
def compute_accuracy(predictions, targets, tolerance=ACCURACY_TOLERANCE):
    errors = torch.abs(predictions - targets)
    accurate = (errors <= tolerance).float()
    accuracy = accurate.mean().item() * 100
    return accuracy

# Training function
def train_model(model, train_loader, test_loader, learning_rate, num_epochs=EPOCHS):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
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
            train_correct += compute_accuracy(outputs, y_batch, tolerance=ACCURACY_TOLERANCE) * X_batch.size(0)
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
                test_correct += compute_accuracy(outputs, y_batch, tolerance=ACCURACY_TOLERANCE) * X_batch.size(0)
        test_loss /= len(test_loader.dataset)
        test_accuracy = test_correct / len(test_loader.dataset)
        test_losses.append(test_loss)
        test_accuracies.append(test_accuracy)

        # Save model checkpoint after each epoch
        checkpoint_path = f"experimental/tuned_flame_frame_predictor_EXPERIMENTAL_epoch_{epoch+1}.pth"
        torch.save(model.state_dict(), checkpoint_path)

        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%, "
              f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%, Saved: {checkpoint_path}")

    return train_losses, test_losses, train_accuracies, test_accuracies

# Load the pre-trained model
model = FlameFramePredictor(input_dimensions)
if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path))
    print(f"Loaded model from {model_path}")
else:
    print(f"Error: Model file {model_path} not found. Please provide a valid path.")
    exit()

results = []
lr = 0.0001

# Create a new model instance to avoid overwriting previous weights
tuned_model = FlameFramePredictor(input_dimensions)
tuned_model.load_state_dict(torch.load(model_path))  # Reload initial weights
train_losses, test_losses, train_accuracies, test_accuracies = train_model(
    tuned_model, train_loader, test_loader, learning_rate=lr
)

# Save tuned model
model_id = "EXPERIMENTAL"
save_path = f"tuned_flame_frame_predictor_{model_id}.pth"
torch.save(tuned_model.state_dict(), save_path)

# Store results
result = {
    'learning_rate': lr,
    'model_id': model_id,
    'final_test_accuracy': test_accuracies[-1],
    'final_test_loss': test_losses[-1],
    'train_losses': train_losses,
    'test_losses': test_losses,
    'train_accuracies': train_accuracies,
    'test_accuracies': test_accuracies
}
results.append(result)

# Print results for this learning rate
print(f"Results for Learning Rate: {lr}")
print(f"Test Accuracy: {result['final_test_accuracy']:.2f}%")
print(f"Test Loss: {result['final_test_loss']:.4f}")
print(f"Model saved as: {save_path}")

end_time = time.time()
print(f"\nTotal time taken: {end_time - start_time:.2f} seconds or {(end_time - start_time)/60:.2f} minutes")
print("Model tuning complete.")