import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import pandas as pd
import os

# Using the pretrained ResNet50 model for feature extraction, it's a CNN that tends to work well with image extraction
model = models.resnet50(pretrained=True)
model.eval()  # Set to evaluation mode
model = torch.nn.Sequential(*list(model.children())[:-1])  # Remove final classification layer

# Define image preprocessing
preprocess = transforms.Compose([
    # Resize to 224x224, as required by ResNet50
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def extract_features(image_path):
    # Load and preprocess image
    image = Image.open(image_path).convert("RGB")
    image_tensor = preprocess(image).unsqueeze(0)  # Add batch dimension
    
    # Extract features
    with torch.no_grad():
        features = model(image_tensor)
    return features.squeeze().numpy()  # Flatten to 1D array

# Process all images and save to CSV
image_dir = "../Videos/extracted"
feature_vectors = []
image_names = []

for img_name in os.listdir(image_dir):
    if img_name.endswith(".png"):
        img_path = os.path.join(image_dir, img_name)
        features = extract_features(img_path)
        feature_vectors.append(features)
        image_names.append(img_name)

# Save to CSV
feature_vectors = np.array(feature_vectors)
df = pd.DataFrame(feature_vectors, index=image_names)
df.to_csv("feature_vectors.csv")