# Tests the NN with 12 features, finds the video and the time at which it lights on fire

import sys
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import image_extractor

num_classes = 12

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



# Load the frame features
df = pd.read_csv("image_output.csv", header=None)
features = df.values.astype(np.float32)
X_single = torch.tensor(features, dtype=torch.float32)

# Load the trained model
model = SimpleNN(input_dim=features.shape[1], hidden_dim=128, output_dim=num_classes)
model.load_state_dict(torch.load("simple_nn.pth"))
model.eval()

no_flame = [0, 1, 3]
before_flame = [4, 5, 7, 10, 11]
after_flame = [8, 9]
fire_now = [2, 6]

# Map cluster to video
cluster_to_video = {
    11: 1,
    4: 2,
    10: 4,
    5: 5,
    7: 7
}

# When flame first appears in each video (frames)
flame_times = [0, 4754, 5278, 0, 5656, 6101, 0, 7208, 0, 0, 0]
            #| 0|  1  |  2  | 3|  4  |  5  | 6|  7  | 8| 9| 10

# Helper function to convert frame to min:sec
def frame_to_min_sec(frame, fps):
    total_seconds = frame / fps
    minutes = int(total_seconds // 60)
    seconds = int(total_seconds % 60)
    return minutes, seconds

# Make predictions
with torch.no_grad():
    outputs = model(X_single)
    predicted_classes = torch.argmax(outputs, dim=1)

# Loop over frames
fps = 5

for i, pred_class in enumerate(predicted_classes):
    pred = pred_class.item()
    print(f"\nFrame {i}: Predicted burn stage cluster {pred}")

    if pred in no_flame:
        print("   ➡ Block will not catch fire.")
    elif pred in before_flame:
        print("   ➡ Danger! Flame imminent...")
    elif pred in after_flame:
        print("   ➡ Fire out, danger passed.")
    elif pred in fire_now:
        print("   ➡ Block currently on fire!")
    else:
        print("   ➡ Unknown state.")

    # If you want to predict fire start time (optional)
    if pred in cluster_to_video:
        video = cluster_to_video[pred]
        flame_frame = flame_times[video]
        if flame_frame != 0:
            minutes, seconds = frame_to_min_sec(flame_frame, fps)
            print(f"   🔥 Fire starts at {minutes}:{seconds:02d} in video {video}")
        else:
            print("   🔥 No flame in this video.")