# Testing the trained NN

import pandas as pd
import numpy as np
import torch
import torch.nn as nn

num_classes = 10

# Define a simple neural network
class SimpleNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Load CSV without using first column as index
df = pd.read_csv("single_test_frame.csv", header=None)  # No header in the file

# Extract just the numeric features (everything except column 0)
features = df.iloc[:, 1:].values.astype(np.float32)

# Convert to tensor
X_single = torch.tensor(features, dtype=torch.float32)

# Load model
model = SimpleNN(input_dim=features.shape[1], hidden_dim=128, output_dim=num_classes)
model.load_state_dict(torch.load("simple_nn.pth"))
model.eval()

# Make prediction
with torch.no_grad():
    output = model(X_single)
    predicted_class = torch.argmax(output, dim=1).item()

print(f"Predicted burn stage cluster: {predicted_class}")