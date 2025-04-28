# RUN PCA ON ALL THE FEATURE VECTORS


from sklearn.decomposition import PCA
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import joblib

num_components = 20 # Number of PCA components to keep

df = pd.read_csv("feature_vectors.csv", index_col=0)
feature_vectors = df.values
feature_names = df.columns

print(feature_vectors.shape)

# Graph PCA total variance explained over number of components
# i = 1
# explained_variance = []
# for i in range(1, 80):
#     pca = PCA(n_components=i)
#     pca.fit(feature_vectors)
#     explained_variance.append(sum(pca.explained_variance_ratio_))
#     print(f"Components: {i}, Explained Variance: {sum(pca.explained_variance_ratio_)}")

# plt.figure(figsize=(10, 6))
# plt.plot(range(1, len(explained_variance) + 1), explained_variance)
# plt.title("Cumulative Explained Variance by PCA Components")
# plt.xlabel("Number of Components")
# plt.ylabel("Cumulative Explained Variance")
# plt.savefig("resnet-explained_variance_-1-1_range.png")
# plt.show()

# Perform PCA
pca = PCA(n_components=num_components)
pca.fit(feature_vectors)
explained_variance = pca.explained_variance_ratio_
print("Explained variance ratio:", explained_variance)
print("Total variance explained:", sum(explained_variance))
print("PCA completed.")

# joblib.dump(pca, "pca_model.pkl")




# Save PCA components to CSV
# pca_components = pca.transform(feature_vectors)
# pca_df = pd.DataFrame(pca_components, index=df.index)
# pca_df.to_csv("pca_components.csv")
# print("PCA components saved to pca_components.csv")