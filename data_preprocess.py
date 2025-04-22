import torch
import torchvision.models as models
import torchvision.transforms as transforms
from sklearn.decomposition import PCA
from PIL import Image
import numpy as np
import pandas as pd
import os

classifying_info = [0.999, 0.999] # Size of block[0.0001 for small, 0.5 for medium, 0.999 for large], wind speed[0.0001 for 0.5 m/s, 0.5 for 1.0 m/s, 0.999 for 1.5 m/s]
video_name = "No.1_L_1.5_200"
image_dir = "../Videos/extracted/" + video_name # Directory containing images
num_components = 10 # Number of PCA components to keep

if not os.path.exists(image_dir):
    print(f"Directory {image_dir} does not exist. Creating it.")
    os.makedirs(image_dir)  # Create directory if it doesn't exist


# Check GPU and PyTorch setup
print("Checking PyTorch and GPU setup...")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"Number of GPUs: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"GPU name: {torch.cuda.get_device_name(0)}")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load ResNet50 model
print("Loading ResNet50 model...")
model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
model.eval()
model = torch.nn.Sequential(*list(model.children())[:-1])
model = model.to(device)  # Move model to GPU

# Define image preprocessing
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
    
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

# Process images
print(f"Checking directory: {os.path.abspath(image_dir)}")
# print(f"Files in directory: {os.listdir(image_dir)}") # Uncomment to debug

feature_vectors = []
image_names = []

for img_name in os.listdir(image_dir):
    if img_name.endswith(".png"):
        print(f"Processing {img_name}...")
        img_path = os.path.join(image_dir, img_name)
        features = extract_features(img_path)
        if features is not None:
            feature_vectors.append(features)
            feature_vectors.append(classifying_info)
            image_names.append(img_name)
        else:
            print(f"Skipping {img_name} due to error")

# Save to CSV
if feature_vectors:
    print("Saving feature vectors to CSV...")
    feature_vectors = np.array(feature_vectors)
    df = pd.DataFrame(feature_vectors, index=image_names)
    df.to_csv(video_name = "_feature_vectors.csv")
    print("Feature vectors saved to " + video_name + "_feature_vectors.csv")
    print("Done!")
else:
    print("No features extracted. Check directory or image files.")

print("Feature extraction completed.")


print("Running PCA on " + num_components + " components...")
# Load feature vectors from CSV
df = pd.read_csv(video_name + "_feature_vectors.csv", index_col=0)
feature_vectors = df.values

print(feature_vectors.shape)


# Perform PCA
pca = PCA(n_components=num_components)
pca.fit(feature_vectors)
explained_variance = pca.explained_variance_ratio_
print("Explained variance ratio:", explained_variance)
print("Total variance explained:", sum(explained_variance))
print("PCA completed.")


# apend PCA components to CSV
pca_components = pca.transform(feature_vectors)
pca_df = pd.DataFrame(pca_components, index=df.index)
if not os.path.exists("pca_components.csv"): # if the file does not exist, create it
    pca_df.to_csv("pca_components.csv")
    print("PCA components saved to pca_components.csv")
else: # if the file exists, append to it
    pca_df.to_csv("pca_components.csv", mode='a', header=False)
    print("PCA components appended to pca_components.csv")



