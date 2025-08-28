import pandas as pd
import matplotlib.pyplot as plt
import os
import cv2
import numpy as np

input_csv_path = "output/angle_analysis.csv"
output_image_path = "output/plots/elbow_angle_report.png"

if not os.path.exists(input_csv_path):
    print(f"Error: Input file not found at '{input_csv_path}'")
    print("Please run the 'video_analyzer_3d.py' script first to generate the data.")
    exit()

df = pd.read_csv(input_csv_path)

# --- NEW: Calculate Wrist Velocity to find Ball Release (with smoothing) ---
# Smooth the raw wrist coordinates using a 3-frame rolling average to reduce noise
window_size = 3
df['r_wrist_x_smooth'] = df['r_wrist_x'].rolling(window=window_size, center=True).mean()
df['r_wrist_y_smooth'] = df['r_wrist_y'].rolling(window=window_size, center=True).mean()
df['r_wrist_z_smooth'] = df['r_wrist_z'].rolling(window=window_size, center=True).mean()

# Calculate the difference in the SMOOTHED position for each axis
df['wrist_x_diff'] = df['r_wrist_x_smooth'].diff()
df['wrist_y_diff'] = df['r_wrist_y_smooth'].diff()
df['wrist_z_diff'] = df['r_wrist_z_smooth'].diff()

# Calculate the 3D distance traveled between frames (Euclidean distance)
# This is a direct measure of the wrist's speed.
df['wrist_velocity'] = np.sqrt(
    df['wrist_x_diff']**2 + 
    df['wrist_y_diff']**2 + 
    df['wrist_z_diff']**2
)

release_frame = df['wrist_velocity'].idxmax()
release_frame_number = df.at[release_frame, 'frame']
print(f"ALGORITHM RESULT: Ball release detected at Frame #{release_frame_number}")

# --- NEW: Snapshot the Release Frame ---
print(f"Extracting snapshot for Frame #{release_frame_number}...")
video_path = 'videos/bowling_2.mp4'  # Path to the original video
cap = cv2.VideoCapture(video_path)

if cap.isOpened():
    # Set the video to the specific release frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, release_frame_number)
    success, frame = cap.read()
    if success:
        # Save the frame as an image
        snapshot_path = 'output/release_frame_snapshot2.png'
        cv2.imwrite(snapshot_path, frame)
        print(f"SUCCESS: Snapshot saved to '{snapshot_path}'")
    else:
        print("Error: Could not read the specific frame from the video.")
    cap.release()
else:
    print(f"Error: Could not open video file at '{video_path}' to take snapshot.")

plt.style.use('seaborn-v0_8-whitegrid')

fig, ax = plt.subplots(figsize=(12, 7))

ax.plot(df['frame'], df['right_elbow_angle'], label='Right Elbow Angle', color='royalblue', linewidth=2)
ax.plot(df['frame'], df['left_elbow_angle'], label='Left Elbow Angle', color='firebrick', linewidth=2)

ax.set_title('Elbow Angle Analysis During Bowling Action', fontsize=16, fontweight='bold')
ax.axvline(x=release_frame_number, color='green', linestyle='--', linewidth=2, label=f'Ball Release (Frame {release_frame_number})')
ax.set_xlabel('Frame Number', fontsize=12)
ax.set_ylabel('Angle (Degrees)', fontsize=12)
ax.legend(fontsize=10)
ax.grid(True)

ax.set_ylim(0, 190)

print(f"Saving report to '{output_image_path}'...")
plt.savefig(output_image_path, dpi=300) # Save with high resolution
print("SUCCESS: Report generated successfully!")