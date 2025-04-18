import os

# this is just hear to test if the images are in the right spot

# Process all images and save to CSV
image_dir = "../Videos/extracted"
feature_vectors = []
image_names = []

for img_name in os.listdir(image_dir):
    if img_name.endswith(".png"):
        print(f"Processing {img_name}...")

