# Data processing stuff for a bunch of fire videos we got

### Potential_Deliverables directory
Some charts collected from data analysis by changing the 'weight' of the wind speed and blocks size

### predict_time_josh directory
Josh's efforts with training a good neural network
#### FIND_BEST_MODEL directory
Efforts to find the best model - 7 iterations
#### image directory
image that's being predicted
#### old_data directory
The aggregation of a lot of data gathered that I didn't want to use but didn't want to totally scrap
#### misc. 
all_archs.txt - All the architectures tested
transform_image.py - transforms the image within the 'image' directory
train_flame.py - The final model chosen
train_predict.py - Prediction function from the final model chosen
separate_features.py - separates video data into separate features and adds label for training (distance from "flame frame")
flame_frame_data.csv - processed flame frame data (from 20 PCA)
pca_model.pkl - PCA model used for prediction of an image
parsed_image_data.csv - parsed image data for prediction

### misc.
change_class_info.py - deprecated, used to alter the 'weight' of wind speed and block size - didn't end up being useful unfortunately
data_pca.py - runs PCA on the full feature vector dataset
k_means.py - runs k-means clustering on the PCA'd dataset
convenience.py - convenience script I ran during data collection
simple_nn.py - a simple neural network for classigyun
test_nn.py - predicts if the picture is on fire, when it will catch fire in the video, etc