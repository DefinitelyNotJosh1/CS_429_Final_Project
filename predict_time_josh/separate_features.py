# This script separates the PCA features from the original data and calculates the distance to the flame frame for each video.
# It also saves the processed data to a new CSV file.

import pandas as pd
import numpy as np
import os

# Load PCA features
df = pd.read_csv("../pca_components.csv", index_col=0)

# Extract video ID and frame number from the index
df['video_id'] = df.index.to_series().apply(lambda x: x.split('/')[0])
df['frame_number'] = df.index.to_series().apply(lambda x: int(x.split('/')[-1].replace('.png', '')))

# Group by video ID
video_groups = df.groupby('video_id')
num_videos = len(video_groups)

# Define flame frames for each video
flame_frames = {
    "No. 1_L_1.5 _200 _02-18-2025": 4680,
    "No. 2_L_1.5 _200 _02-19-2025": 5180,
    "No. 4_M_1.5 _200 _02-27-2025": 5610,
    "No. 5_L_1.0 _200 _01-31-2025": 6075,
    "No. 7_S_1.0 _200 _03-27-2025": 7185,
}

# Add flame frame and calculate distance to flame frame
for video_id, group in video_groups:
    flame_frame = flame_frames.get(video_id, None)
    if flame_frame is not None:
        df.loc[df['video_id'] == video_id, 'flame_frame'] = flame_frame
        # Calculate distance to flame frame: flame_frame - frame_number
        df.loc[df['video_id'] == video_id, 'distance_to_flame_frame'] = (
            flame_frame - df.loc[df['video_id'] == video_id, 'frame_number']
        )
    else:
        print(f"Warning: No flame frame found for video {video_id}")

# Filter out videos without flame frames
df_with_flame_frame = df[df['flame_frame'].notnull()].copy()

# Drop unnecessary columns
df_with_flame_frame = df_with_flame_frame.drop(columns=['video_id', 'frame_number', 'flame_frame'])

# Save to a new CSV
df_with_flame_frame.to_csv("flame_frame_data.csv")
print("Saved processed data to flame_frame_data.csv")