from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib.colors as mcolors

# Load feature vectors
df = pd.read_csv("L_1.5_200_pca_components.csv", index_col=0)
features = df.values

# Apply K-Means clustering
kmeans = KMeans(n_clusters=5, random_state=42)
clusters = kmeans.fit_predict(features)

# Add cluster labels to DataFrame
df["cluster"] = clusters
df.to_csv("L_1.5_200_clustered_features.csv")
print("Cluster assignments saved to L_1.5_200_clustered_features.csv")


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

###
#    Indiana added a graph to display the different labels through the video
#    (currently with 5 clusters)
###

# Extract the last column as labels
labels = df.iloc[:, -1]

# Convert labels to numeric values
numeric_labels = labels.astype(int).to_numpy().reshape(1, -1)  # shape (1, N)

# Set up colormap and normalization
cmap = mcolors.ListedColormap(['blue', 'orange', 'yellow', 'red', 'green'])
bounds = [0, 1, 2, 3, 4, 5]
norm = mcolors.BoundaryNorm(bounds, cmap.N)

# Plot using imshow
plt.figure(figsize=(12, 2))
plt.imshow(numeric_labels, aspect='auto', cmap=cmap, norm=norm)
plt.title("Frame Labels Over Time")
plt.xlabel("Frame Index (Time)")
plt.yticks([])  # hide y-axis
plt.colorbar(ticks=[0, 1, 2, 3, 4], label='Label')
plt.tight_layout()

# Show the plot
plt.show()

print("Explained variance ratio:", pca.explained_variance_ratio_)
print("Total variance explained:", sum(pca.explained_variance_ratio_))