# THIS SCRIPT IS FOR RUNNING EVERYTHING - DO NOT RUN IT UNLESS YOU HAVE THE DATASET/MACHINE SETUP
import os

cancel = input("This script is for running all the preprocessing and data analysis. You will get to choose what to run - Do you want to continue? (y/n): ")
if cancel.lower() != 'y':
    print("Exiting script.")
    exit()

image_dir = "../Videos/extracted"
feature_vectors = []
image_names = []

for img_name in os.listdir(image_dir):
    if img_name.endswith(".png"):
        image_names.append(img_name)
    
num_images = len(image_names)
allow_preprocess = True

if num_images < 0:
    print("Images not in correct directory. Images should be in ../Videos/extracted (Videos/extracted in the folder outside of this folder).")
    print("Cannot run data preprocessing.")
    allow_preprocess = False
else:
    print("Images are in the correct directory.")
    print(f"{num_images} are available...")
    print("Data preprocessing is allowed.")


# Ask which steps to run
if allow_preprocess:
    preprocess_answer = input("Do you want to run data preprocessing? (y/n): ")
else:
    preprocess_answer = 'n'

pca_answer = input("Do you want to run PCA analysis? (y/n): ")
kmeans_answer = input("Do you want to run K-Means clustering? (y/n): ")

if preprocess_answer.lower() == 'y':
    print("-------------Running data preprocessing-------------")
    os.system("python data_preprocess.py")
    print("-------------Data preprocessing Done-------------\n")

if pca_answer.lower() == 'y':
    print("-------------Running PCA analysis-------------")
    os.system("python data_pca.py")
    print("-------------PCA analysis Done-------------\n")

if kmeans_answer.lower() == 'y':
    print("-------------Running K-Means clustering-------------")
    os.system("python k_means.py")
    print("-------------K-Means clustering Done-------------")
    print("(Don't worry about the errors in the K-means clustering, the output is still correct)\n")


print("-------------Done-------------")