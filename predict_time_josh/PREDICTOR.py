# Combined all the other scripts into one predictor script
# Predicts how long an image will take to burn/has been burning
# Load an image to the "image" directory

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import joblib
import os
import time

start_time = time.time()

# Parameters
input_dimensions = 20                             # Number of PCA features
image_dir = "image"                               # Directory containing the input image
pca_model_path = "pca_model.pkl"                  # Path to pre-trained PCA model
model_path = "FIND_BEST_MODEL/further_training/tuned_flame_frame_predictor_EXPERIMENTAL.pth"  # Path to pre-trained model
output_file = "parsed_image_data_pca.csv"         # Output file for PCA features
frame_rate = 5                                    # Frames per second for time conversion

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define neural network (using best model from training)
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

# Image preprocessing pipeline
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Function to extract ResNet50 features from an image
def extract_features(image_path):
    try:
        image = Image.open(image_path).convert("RGB")
        image_tensor = preprocess(image).unsqueeze(0).to(device)
        with torch.no_grad():
            # Load ResNet50
            model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
            model = torch.nn.Sequential(*list(model.children())[:-1]).to(device)
            model.eval()
            features = model(image_tensor)
        return features.squeeze().cpu().numpy()
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None

# Function to load PCA model
def load_pca_model(model_path):
    if os.path.exists(model_path):
        pca = joblib.load(model_path)
        print("PCA model loaded successfully.")
        return pca
    else:
        print(f"PCA model file {model_path} not found")
        return None

# Function to extract top PCA components
def extract_top_pca_components(feature_vector, pca_model, num_components=20):
    feature_vector = np.array(feature_vector).reshape(1, -1)
    if feature_vector.shape[1] != pca_model.n_features_in_:
        print(f"Input feature vector dimension ({feature_vector.shape[1]}) "
              f"does not match PCA model's expected dimension ({pca_model.n_features_in_}).")
        return None
    transformed_vector = pca_model.transform(feature_vector)
    return transformed_vector[:, :num_components]

# Function to predict flame frame and convert to time
def predict_flame_frame(features, model, scaler):
    if len(features) != input_dimensions:
        print(f"Input feature vector must have {input_dimensions} dimensions, got {len(features)}")
        return None, None, None
    features = np.array(features).reshape(1, -1)
    features_scaled = scaler.transform(features)
    features_tensor = torch.tensor(features_scaled, dtype=torch.float32).to(device)
    with torch.no_grad():
        prediction = model(features_tensor).item()  # Distance to flame frame in frames
    seconds = prediction / frame_rate
    minutes = seconds // 60  # Integer minutes
    seconds_remainder = seconds % 60  # Remaining seconds
    # Format minutes:seconds with padded seconds
    time_formatted = f"{int(minutes):02d}:{int(seconds_remainder):02d}"
    if prediction >= 0:
        print(f"Predicted time until burning: {seconds:.2f} seconds ({time_formatted})")
    else:
        abs_seconds = abs(seconds)
        abs_minutes = abs_seconds // 60
        abs_seconds_remainder = abs_seconds % 60
        abs_time_formatted = f"{int(abs_minutes):02d}:{int(abs_seconds_remainder):02d}"
        print(f"Predicted time since burning started: {abs_seconds:.2f} seconds ({abs_time_formatted})")
    return prediction, seconds, minutes

#Load PCA model
pca_model = load_pca_model(pca_model_path)
if pca_model is None:
    exit()

# Process image and extract PCA features
image_files = [f for f in os.listdir(image_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
if not image_files:
    print(f"No valid images found in {image_dir}. Please add an image.")
    exit()

# Process only the first image
img_name = image_files[0]
img_path = os.path.join(image_dir, img_name)
print(f"Processing {img_name}...")

features = extract_features(img_path)
features = np.append(features, [0, 0])  # Append two zeros to the end of the feature vector (PCA expects 2050 features)
if features is None:
    print(f"Failed to extract features for {img_name}. Check the image file.")
    exit()

# Apply PCA
pca_components = extract_top_pca_components(features, pca_model, input_dimensions)
if pca_components is None:
    print("Failed to extract PCA components.")
    exit()
print(f"Extracted PCA components shape: {pca_components.shape}")

# Fit scaler on training data
train_df = pd.read_csv("FIND_BEST_MODEL/PCA_20_flame_frame_data.csv", index_col=0)
X_train = train_df.iloc[:, :-1].values
scaler = StandardScaler()
scaler.fit(X_train)

# Load model
model = FlameFramePredictor(input_dimensions)
if os.path.exists(model_path):
    print(f"Model file {model_path} exists. Proceeding to load the model.")
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()
    print(f"Loaded model from {model_path}")
else:
    print(f"Error: Model file {model_path} not found. Please provide a valid path.")
    exit()

# Step 4: Predict directly using PCA components
prediction, seconds, minutes = predict_flame_frame(pca_components[0], model, scaler)
if prediction is None:
    print("Prediction failed.")
    exit()

print("Processing complete!")
print(f"Total time: {time.time() - start_time:.2f} seconds")