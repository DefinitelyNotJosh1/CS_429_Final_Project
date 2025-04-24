import numpy as np
import pandas as pd

# Load the feature vectors
df = pd.read_csv("feature_vectors.csv", index_col=0)

# Drop the last two columns (block size and wind speed)
df = df.iloc[:, :-2]

# Run PCA on the feature vectors
from sklearn.decomposition import PCA
pca = PCA(n_components=10)  # Adjust the number of components as needed
pca.fit(df)
pca_features = pca.transform(df)
print("PCA completed.")
print("Explained variance ratio:", pca.explained_variance_ratio_)
print("Total variance explained:", sum(pca.explained_variance_ratio_))

# Save the PCA features to a new CSV file
pca_df = pd.DataFrame(pca_features, index=df.index)
pca_df.to_csv("testing_pca_features.csv")
print("PCA features saved to testing_pca_features.csv")