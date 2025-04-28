import torch
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import StandardScaler
import pandas as pd

input_dimensions = 20  # Number of PCA features

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

# Load scaler (fit on training data)
df = pd.read_csv("non-normal_flame_frame_data.csv", index_col=0)
X_train = df.iloc[:, :-1].values  # PCA features
scaler = StandardScaler()
scaler.fit(X_train)

# Load model
model = FlameFramePredictor()
model.load_state_dict(torch.load("flame_frame_predictor.pth"))
model.eval()

# Function to predict and convert to time
def predict_flame_frame(features):
    # Input validation
    if len(features) != input_dimensions:
        print(f"Input feature vector must have {input_dimensions} dimensions")
        exit()
    
    # Convert to tensor and standardize
    features = np.array(features).reshape(1, -1)
    features_scaled = scaler.transform(features)
    features_tensor = torch.tensor(features_scaled, dtype=torch.float32)

    # Predict
    with torch.no_grad():
        prediction = model(features_tensor).item()  # Distance to flame frame in frames
    
    # Convert to seconds and minutes (5 FPS)
    seconds = prediction / 5
    minutes = seconds / 60

    # Interpret prediction
    if prediction >= 0:
        print(f"Predicted time until burning: {seconds:.2f} seconds ({minutes:.2f} minutes)")
    else:
        print(f"Predicted time since burning started: {abs(seconds):.2f} seconds ({abs(minutes):.2f} minutes)")
    
    return prediction, seconds, minutes


# Feed in the feature vector
pred_data = pd.read_csv("parsed_image_data.csv", index_col=0)
print(pred_data.shape)
prediction_features = pred_data.values
prediction_features = prediction_features[0]
print(prediction_features.shape)
predict_flame_frame(prediction_features)