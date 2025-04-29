# K-Means clustering script - runs on 

import datetime
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib.colors as mcolors


num_clusters = 12 # Number of clusters for K-Means

# Load feature vectors
df = pd.read_csv("pca_components.csv", index_col=0)
features = df.values

# Apply K-Means clustering
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
clusters = kmeans.fit_predict(features)

# Add cluster labels to DataFrame
df["cluster"] = clusters
df.to_csv("clustered_features.csv")
print("Cluster assignments saved to clustered_features.csv")


# Visualize the clusters in 3D
pca = PCA(n_components=3)
features_3d = pca.fit_transform(features)
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(features_3d[:, 0], features_3d[:, 1], features_3d[:, 2], c=clusters, cmap='viridis')
ax.set_title("K-Means Clustering (PCA-reduced to 3D)")
ax.set_xlabel("PCA Component 1")
ax.set_ylabel("PCA Component 2")
ax.set_zlabel("PCA Component 3")
plt.colorbar(scatter, label='Cluster')
plt.show()

# Reduce dimensionality to 2D
pca = PCA(n_components=2)
features_2d = pca.fit_transform(features)

# Create a scatter plot of the clusters
plt.figure(figsize=(10, 6))
scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1], c=clusters, cmap='viridis', s=50, alpha=0.6)

# Add cluster centroids
centroids_2d = pca.transform(kmeans.cluster_centers_)
plt.scatter(centroids_2d[:, 0], centroids_2d[:, 1], c='red', marker='x', s=200, linewidths=3, label='Centroids')

# Add labels and title
plt.title("K-Means Clustering (PCA-reduced to 2D)")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.colorbar(scatter, label='Cluster')
plt.legend()

# Create a an array of numbers of increasing increment for the number of clusters
tick_count = np.arange(0, num_clusters)
tick_count = tick_count.tolist()

###
#    Indiana added a graph to display the different labels through the video
#    (currently with 5 clusters)
###

# Extract the last column as labels
labels = df.iloc[:, -1]

# Convert labels to numeric values
numeric_labels = labels.astype(int).to_numpy().reshape(1, -1)  # shape (1, N)

# Set up colormap and normalization
cmap = mcolors.ListedColormap(['blue', 'orange', 'yellow', 'red', 'green', 'purple', 'pink', 'brown', 'gray', 'cyan', 'magenta', 'lime', 'teal', 'navy', 'olive', 'maroon', 'coral'])
bounds = np.arange(0, num_clusters + 1)  # bounds for the colormap
bounds = bounds.tolist()
norm = mcolors.BoundaryNorm(bounds, cmap.N)

# Plot using imshow
plt.figure(figsize=(12, 2))
plt.imshow(numeric_labels, aspect='auto', cmap=cmap, norm=norm)
plt.title("Frame Labels Over Time")
plt.xlabel("Frame Index (Time)")
plt.yticks([])  # hide y-axis
plt.colorbar(ticks=tick_count, label='Label')
plt.tight_layout()

# Show the plot
plt.show()

# Find cluster classifications for each video
df['video_id'] = df.index.to_series().apply(lambda x: x.split('/')[0]) # split the index to get video ID
video_groups = df.groupby('video_id')
num_videos = len(video_groups)
plt.figure(figsize=(20, 2 * num_videos))

first_flame_instances = {}

for idx, (video_id, group) in enumerate(video_groups):
    video_labels = group['cluster'].astype(int).to_numpy().reshape(1, -1)

    # Find the first occurrence of cluster 6 (flame) in the labels
    flame_frames = (video_labels == 6).nonzero()[1]  # Index where cluster 6 occurs
    first_flame_frame = flame_frames[0] if len(flame_frames) > 0 else None

    # Store the first frame where cluster 6 is detected
    first_flame_instances[video_id] = first_flame_frame

    plt.subplot(num_videos, 1, idx + 1)
    plt.imshow(video_labels, aspect='auto', cmap=cmap, norm=norm)
    plt.title(f"Group Layout Over Time - {video_id}")
    plt.xlabel("Frame Index (Time)")
    plt.yticks([])
    plt.colorbar(ticks=tick_count, label='Label')
plt.tight_layout()
plt.savefig("resnet_group_layout_over_time_-1-1_range.png")
plt.show()

# Print the first instance of cluster 6 for each video
print("First instances of cluster 6 (flame) in each video:")
for video_id, frame in first_flame_instances.items():
    if frame is not None:
        # Convert frame number to timestamp
        secs = frame / 5  # 5 fps
        timestamp = str(datetime.timedelta(seconds=secs))
        print(f"{video_id}: Frame {frame} (Timestamp: {timestamp})")
    else:
        print(f"{video_id}: No flame detected")