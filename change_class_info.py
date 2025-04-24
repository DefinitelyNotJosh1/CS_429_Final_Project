import pandas as pd
import os

# Classifying info - how "heavy" the changes in block size and wind speed are
small_block = -5  # Size of small block
medium_block = 0.0001  # Size of medium block
large_block = 5  # Size of large block
low_wind = -5  # Wind speed of low wind
medium_wind = 0.0001  # Wind speed of medium wind
high_wind = 5  # Wind speed of high wind

classifying_info = [
    [large_block, high_wind],
    [large_block, high_wind],
    [small_block, high_wind],
    [medium_block, high_wind],
    [large_block, medium_wind],
    [medium_block, high_wind],
    [small_block, medium_wind],
    [medium_block, low_wind]
]

video_names = [
    "No. 1_L_1.5 _200 _02-18-2025",
    "No. 2_L_1.5 _200 _02-19-2025",
    "No. 3_S_1.5 _200 _04-03-2025",
    "No. 4_M_1.5 _200 _02-27-2025",
    "No. 5_L_1.0 _200 _01-31-2025",
    "No. 6_M_1.0 _200 _02-13-2025",
    "No. 7_S_1.0 _200 _03-27-2025",
    "No. 10_M_0.5 _200 _03-12-2025"
]

# Load the CSV file
df = pd.read_csv("feature_vectors.csv", index_col=0)

# Extract video ID from index
df['video_id'] = df.index.to_series().apply(lambda x: x.split('/')[0])

# Get the column names of the last two columns (video ID column is added for convenience, hence the "last two" are [-2] and [-3])
columns = df.columns.tolist()
last_col = columns[-2]  # Last column
second_last_col = columns[-3]  # Second-to-last column



# Assign block size to second-to-last column and wind speed to last column
for i in range(len(video_names)):
    print("Changing columns in " + video_names[i] + f" ({i+1}/{len(video_names)})")
    video_name = video_names[i]
    block = classifying_info[i][0]  # Block size
    wind = classifying_info[i][1]   # Wind speed
    mask = df['video_id'] == video_name
    if mask.sum() > 0:  # Ensure there are matching rows
        df.loc[mask, second_last_col] = block
        df.loc[mask, last_col] = wind
    else:
        print(f"Warning: No rows found for video {video_name}")

# Print confirmation
print("Updated feature_vectors.csv with new values in the last two columns.")
# Group by video ID extracted from index
for name, group in df.groupby(df.index.to_series().apply(lambda x: x.split('/')[0])):
    print(f"Video ID: {name}, Number of frames: {len(group)}, "
          f"Second-to-last column ({second_last_col}): {group[second_last_col].iloc[0]}, "
          f"Last column ({last_col}): {group[last_col].iloc[0]}")
    
# Drop the temporary video_id column
df = df.drop('video_id', axis=1)

# Save the modified DataFrame back to CSV
print("Saving modified dataframe to feature_vectors.csv...")
df.to_csv("feature_vectors.csv")

# Run PCA and K-Means clustering python files
print("Running PCA...")
os.system("python data_pca.py")

print("Running K-Means clustering...")
os.system("python k_means.py")