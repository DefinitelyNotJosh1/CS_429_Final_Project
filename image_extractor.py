# turns an image into a feature vector and outputs to the test_frames.csv

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from sklearn.decomposition import PCA
import joblib
import os
import time

start_time = time.time()

# Parameters
input_dimensions = 20                             # Number of PCA features
image_dir = "image_simple"                               # Directory containing the input image
pca_model_path = "pca_model.pkl"                  # Path to pre-trained PCA model
output_file = "image_output.csv"         # Output file for PCA features

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


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

#Load PCA model
pca_model = load_pca_model(pca_model_path)
if pca_model is None:
    exit()

# Process image and extract PCA features
image_files = [f for f in os.listdir(image_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
if not image_files:
    print(f"No valid images found in {image_dir}. Please add an image.")
    exit()

# Process the first image
img_name = image_files[0]
img_path = os.path.join(image_dir, img_name)
print(f"Processing {img_name}...")

features = extract_features(img_path)
features = np.append(features, [0, 0])  # Append two zeros to the end of the feature vector (PCA expects 2050 features, issue on my end (Josh))
if features is None:
    print(f"Failed to extract features for {img_name}. Check the image file.")
    exit()

# Apply PCA
pca_components = extract_top_pca_components(features, pca_model, input_dimensions)
if pca_components is None:
    print("Failed to extract PCA components.")
    exit()
print(f"Extracted PCA components shape: {pca_components.shape}")

# save PCA components to CSV
pca_df = pd.DataFrame(pca_components)
pca_df.to_csv(output_file, index=False, header=False)


print("Processing complete!")
print(f"Total time: {time.time() - start_time:.2f} seconds")