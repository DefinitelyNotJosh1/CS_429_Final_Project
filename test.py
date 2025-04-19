import os

# this is just hear to test if the images are in the right spot

# Process all images and save to CSV
image_dir = "../Videos/extracted"
feature_vectors = []
image_names = []

for img_name in os.listdir(image_dir):
    if img_name.endswith(".png"):
        image_names.append(img_name)
        break

if image_names.__len__() < 0:
    print("Images not in correct directory. Images should be in ../Videos/extracted (Videos/extracted in the folder outside of this folder).")
else:
    print("Images are in the correct directory.")