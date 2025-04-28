import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import pandas as pd
import os
import time
from sklearn.decomposition import PCA
import joblib

start_time = time.time()

# Parameters
image_dir = "image"  # Input directory
output_file = "parsed_image_data_pca.csv"
num_components = 20
pca_model_path = "pca_model.pkl"

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load ResNet50
model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
model = torch.nn.Sequential(*list(model.children())[:-1]).to(device)
model.eval()

# Image preprocessing pipeline
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Function to extract features from an image
def extract_features(image_path):
    try:
        image = Image.open(image_path).convert("RGB")
        image_tensor = preprocess(image).unsqueeze(0).to(device)
        with torch.no_grad():
            features = model(image_tensor)
        return features.squeeze().cpu().numpy()
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None

# Function to load PCA model
def load_pca_model(model_path):
    try:
        pca = joblib.load(model_path)
        print("PCA model loaded successfully.")
        return pca
    except FileNotFoundError:
        raise FileNotFoundError(f"PCA model file {model_path} not found. Please train and save the PCA model first.")

# Function to extract top PCA components
def extract_top_pca_components(feature_vector, pca_model, num_components=20):
    # Convert input to numpy array and ensure correct shape (1, n_features)
    feature_vector = np.array(feature_vector).reshape(1, -1)
    
    # Check if input dimension matches PCA model's expected input
    if feature_vector.shape[1] != pca_model.n_features_in_:
        raise ValueError(f"Input feature vector dimension ({feature_vector.shape[1]}) "
                        f"does not match PCA model's expected dimension ({pca_model.n_features_in_}).")
    
    # Transform the feature vector using PCA
    transformed_vector = pca_model.transform(feature_vector)
    
    # Ensure we return only the specified number of components
    return transformed_vector[:, :num_components]

# Load PCA model
pca_model = load_pca_model(pca_model_path)

# Collect feature vectors and identifiers
all_pca_vectors = []
all_identifiers = []

# Get the first valid image in the directory
image_files = [f for f in os.listdir(image_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
if not image_files:
    print(f"No valid images found in {image_dir}. Please add an image.")
    exit()

# Process only the first image
img_name = image_files[0]
img_path = os.path.join(image_dir, img_name)
print(f"Processing {img_name}...")

# Extract ResNet50 features
features = extract_features(img_path)
if features is None:
    print(f"Failed to extract features for {img_name}. Check the image file.")
    exit()

# Apply PCA to get top 20 components
try:
    pca_components = extract_top_pca_components(features, pca_model, num_components)
    all_pca_vectors.append(pca_components[0])  # Extract 1D array
    all_identifiers.append(img_name)
    print(f"Successfully extracted {num_components} PCA components for {img_name}")
except ValueError as e:
    print(f"PCA Error: {e}")
    exit()

# Save PCA components to CSV
if all_pca_vectors:
    # Convert to array
    pca_array = np.array(all_pca_vectors)
    print(f"PCA components array shape: {pca_array.shape}")
    
    # Save PCA components to CSV
    pca_df = pd.DataFrame(pca_array, index=all_identifiers, 
                          columns=[f"PCA_{i+1}" for i in range(num_components)])
    pca_df.to_csv(output_file)
    print(f"PCA components saved to {output_file}")
else:
    print("No PCA components extracted.")

print("Processing complete!")
print(f"Total time: {time.time() - start_time:.2f} seconds")