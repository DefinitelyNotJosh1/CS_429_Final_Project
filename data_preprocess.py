# THIS IS FOR VIDEO EXTRACTION AND FEATURE EXTRACTION - DO NOT RUN THIS UNLESS YOU HAVE THE DATASET/MACHINE SETUP

import torch
import torchvision.models as models
import torchvision.transforms as transforms
from sklearn.decomposition import PCA
from PIL import Image
import numpy as np
import pandas as pd
import os
import ffmpeg
import time
import tqdm # tqdm is a progress bar library - I just wanted to use it

start_time = time.time()

# Input data
num_components = 10 # for PCA

  
# Classiying info - how "heavy" the changes in block size and wind speed are
# Block info
small_block = -5 # Size of small block
medium_block = 0.0001 # Size of medium block
large_block = 5 # Size of large block
# Wind speed
low_wind = -5 # Wind speed of low wind
medium_wind = 0.0001 # Wind speed of medium wind
high_wind = 5 # Wind speed of high wind

classifying_info = [[large_block, high_wind], [large_block, high_wind], [small_block, high_wind], 
                    [medium_block, high_wind], [large_block, medium_wind], [medium_block, high_wind],
                    [small_block, medium_wind], [medium_block, low_wind]] 
# ^ Size of block  (-15 for small,   0.0001 for medium,  15 for large), 
#   wind speed     (-15 for 0.5 m/s, 0.0001 for 1.0 m/s, 15 for 1.5 m/s)
video_names_with_end = ["No. 1_L_1.5 _200 _02-18-2025.MOV", "No. 2_L_1.5 _200 _02-19-2025.MOV", "No. 3_S_1.5 _200 _04-03-2025.MOV", 
                        "No. 4_M_1.5 _200 _02-27-2025.MOV", "No. 5_L_1.0 _200 _01-31-2025.MOV", "No. 6_M_1.0 _200 _02-13-2025.MOV",
                        "No. 7_S_1.0 _200 _03-27-2025.MOV", "No. 10_M_0.5 _200 _03-12-2025.MOV"]

video_names = [name[:-4] for name in video_names_with_end]

# Device setup
print("Checking PyTorch and GPU setup...")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"Number of GPUs: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"GPU name: {torch.cuda.get_device_name(0)}")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Ask if user wants to run FFMPEG - if not, skip the video extraction step
ffmpeg_answer = input("Do you want to run FFMPEG to extract frames from videos? (y/n): ")
if ffmpeg_answer.lower() != 'y':
    print("Skipping FFMPEG extraction step.")

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

# Collect all feature vectors
all_feature_vectors = []
all_identifiers = []
video_idx = 0 # track video ID

for video_name in video_names:
    output_dir = "../Videos/extracted/" + video_name + "/" # kept out of if statement so image processing works

    if ffmpeg_answer.lower() == 'y':
        video_path = "../Videos/" + video_names_with_end[video_idx]

        if not os.path.exists(video_path):
            print(f"Video {video_path} does not exist. Skipping.")
            continue
        
       
        os.makedirs(output_dir, exist_ok=True)
        print(f"Extracting frames from {video_path}...")
        # ffmpeg has a python library - not even surprised lol - but great, makes my job easier
        ffmpeg.input(video_path).output(f"{output_dir}/%04d.png", format="image2", vcodec="png", r=5).run(overwrite_output=True)

    # Process images
    for img_name in tqdm.tqdm(os.listdir(output_dir), desc=f"Processing images for {video_name}"):
        if img_name.endswith(".png"):
            img_path = os.path.join(output_dir, img_name)
            features = extract_features(img_path)
            if features is None:  # Probably won't be an issue, but I don't want to have to rerun the whole thing if I can avoid it
                print(f"Skipping {img_name} due to extraction error.")
                continue
            # Combine features with classifying info
            feature_vector = np.concatenate([features, classifying_info[video_idx]])
            all_feature_vectors.append(feature_vector)
        all_identifiers.append(f"{video_name}/{img_name}")  # Unique identifier for each image - why not, better to add now then want them later and not have them
    
    video_idx += 1
    print(f"Processed video {video_name}.\n")


# Convert to array and apply PCA - doing this after all videos are processed, running PCA on each video separately was a bad idea in hindsight
if all_feature_vectors:
    # Save all feature vectors to a CSV file - it'll be a big one, but I don't care - I only want to run this script once
    print("Saving feature vectors to CSV...")
    feature_df = pd.DataFrame(all_feature_vectors, index=all_identifiers)
    feature_df.to_csv("feature_vectors.csv")
    print("Feature vectors saved to feature_vectors.csv")

    # apply PCA to all feature vectors
    print("Applying PCA to all feature vectors...")
    feature_array = np.array(all_feature_vectors)
    pca = PCA(n_components=num_components)
    pca.fit(feature_array)
    pca_components = pca.transform(feature_array)
    
    print(f"Explained variance ratio: {pca.explained_variance_ratio_}")
    print(f"Total variance explained: {sum(pca.explained_variance_ratio_)}")

    # Save PCA components
    pca_df = pd.DataFrame(pca_components, index=all_identifiers)
    pca_df.to_csv("pca_components.csv")
    print("PCA components saved to pca_components.csv")
else:
    print("No features extracted.")

print("All videos processed!")
print(f"Total time: {time.time() - start_time:.2f} seconds or {(time.time() - start_time) / 60:.2f} minutes")

# Write total time it took to a file just because
with open("total_time.txt", "w") as f:
    f.write(f"Total time: {time.time() - start_time:.2f} seconds or {(time.time() - start_time) / 60:.2f} minutes")
