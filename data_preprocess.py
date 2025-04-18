import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import pandas as pd
import os

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
image_dir = "../Videos/extracted"  # Try absolute path if this fails
print(f"Checking directory: {os.path.abspath(image_dir)}")
print(f"Files in directory: {os.listdir(image_dir)}")
feature_vectors = []
image_names = []

for img_name in os.listdir(image_dir):
    if img_name.endswith(".png"):
        print(f"Processing {img_name}...")
        img_path = os.path.join(image_dir, img_name)
        features = extract_features(img_path)
        if features is not None:
            feature_vectors.append(features)
            image_names.append(img_name)
        else:
            print(f"Skipping {img_name} due to error")

# Save to CSV
if feature_vectors:
    print("Saving feature vectors to CSV...")
    feature_vectors = np.array(feature_vectors)
    df = pd.DataFrame(feature_vectors, index=image_names)
    df.to_csv("feature_vectors.csv")
    print("Done!")
else:
    print("No features extracted. Check directory or image files.")