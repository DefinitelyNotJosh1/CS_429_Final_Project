import numpy as np
import pandas as pd

# load pca_components.csv and see range of values in each column
df = pd.read_csv("pca_components.csv", index_col=0)
pca_components = df.values
print(pca_components.shape)
print("Range of values in each column:")
for i in range(pca_components.shape[1]):
    print(f"Column {i}: min={np.min(pca_components[:, i])}, max={np.max(pca_components[:, i])}")
print("Mean of each column:")