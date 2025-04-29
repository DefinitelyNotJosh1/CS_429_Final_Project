# Data processing stuff for a bunch of fire videos we got

## HOW TO USE:
We have two interactive options - the simple neural network that uses k-means clustering to give the time in a video a block is expected to catch fire/if it will catch on fire at all, and the network that predicts how long it will be until a block catches fire.

#### Simple neural network:
1. Put an image in the image_simple directory
2. Run "image_extractor.py"
3. Run "test_nn.py"

#### Time prediction neural network:
1. Put an image in the predict_time_josh/image directory
2. Run Predictor.py


## File explanation

### Potential_Deliverables directory
Some charts collected from data analysis by changing the 'weight' of the wind speed and blocks size

### predict_time_josh directory
Josh's efforts with a burn prediction network
#### FIND_BEST_MODEL directory
Efforts to find the best model - 7 iterations/generations
#### misc.
all_archs.txt - All the architectures tested

PCA_20_flame_frame_data.csv - data used for training

#### image directory
Image that's being predicted
#### old_data directory
Data gathered that I didn't have use of but didn't want to totally scrap it. It's, for all intents and purposes, junk that might be useful later.
#### ####_iteration directories
Contains the scripts associated with their iterations, as well as the models and the text output
#### further_training directory
Various scripts/models created during the fine-tuning process

#### misc. 
PREDICTOR.py - Predicts the time until/how long an image has been burning using the final model chosen

separate_features.py - separates video data into separate features and adds label for training (distance from "flame frame")

pca_model.pkl - PCA model used for prediction of an image

### image_simple
Directory to put an image in for the simple neural network prediction

### misc.
image_extractor.py - converts an image in the image_simple directory into a feature vector CSV

data_preprocess.py - main preprocessing script. 

change_class_info.py - used to alter the 'weight' of wind speed and block size - didn't end up being useful unfortunately

data_pca.py - deprecated, now ran in data_preprocess.py automatically. Ran PCA on the full feature vector dataset.

k_means.py - runs k-means clustering on the PCA'd dataset

convenience.py - convenience script I ran during data collection

simple_nn.py - a simple neural network for classifying the data

test_nn.py - predicts if the picture is on fire, when it will catch fire in the video, etc