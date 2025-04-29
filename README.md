Data processing stuff for a bunch of fire videos we got
HOW TO USE
We have two interactive options:

Simple neural network: uses k-means clustering to give the time in a video a block is expected to catch fire, or if it will catch fire at all.

Time prediction neural network: predicts how long it will be until a block catches fire.

Simple neural network
Put an image in the image_simple directory.

Run image_extractor.py.

Run test_nn.py.

Time prediction neural network
Put an image in the predict_time_josh/image directory.

Run Predictor.py.

FILE EXPLANATION
<details> <summary><strong>Directory Tree</strong></summary>
plaintext
Copy
Edit
Potential_Deliverables/
    (Charts from data analysis varying wind speed and block size weight)

predict_time_josh/
├── FIND_BEST_MODEL/
│   (Scripts, models, and text outputs from 7 iterations)
│
├── all_archs.txt (All architectures tested)
├── PCA_20_flame_frame_data.csv (Training data)
│
├── image/
│   (Image to be predicted)
│
├── old_data/
│   (Old data not used but kept)
│
├── ####_iteration directories
│   (Scripts, models, and text outputs per iteration)
│
├── further_training/
│   (Scripts and models for fine-tuning)
│
├── PREDICTOR.py (Final model prediction script)
├── separate_features.py (Separates video data into features/labels)
└── pca_model.pkl (PCA model for predictions)

image_simple/
    (Directory for images for simple neural network)

Miscellaneous files (root level):
├── image_extractor.py (Converts image to feature vector CSV)
├── data_preprocess.py (Main preprocessing script)
├── change_class_info.py (Script to alter wind speed and block size weight - unused)
├── data_pca.py (Deprecated PCA script)
├── k_means.py (Runs k-means clustering)
├── convenience.py (Convenience script for data collection)
├── simple_nn.py (Simple neural network for classification)
└── test_nn.py (Predicts fire timing and status)
</details>