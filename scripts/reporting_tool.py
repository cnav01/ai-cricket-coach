import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# --- Configuration ---
# Define the paths for the two CSV files you want to compare
user_csv_path = 'output/bowling_analysis.csv'
benchmark_csv_path = 'output/pro_bowler_analysis.csv' # This should be the pro's data file
# Define where to save the final comparison report
output_image_path = 'output/comparison_report_vs_pro.png'

# --- Reusable Analysis Function ---
def analyze_performance_data(csv_path):
    """
    Reads a performance CSV, finds the ball release frame, 
    and returns a normalized DataFrame centered around the release.
    """
    df = pd.read_csv(csv_path)
    
    # 1. Find Ball Release using Smoothed Wrist Velocity
    window_size = 3
    df['wrist_x_smooth'] = df['bowling_arm_wrist_x'].rolling(window=window_size, center=True).mean()
    df['wrist_y_smooth'] = df['bowling_arm_wrist_y'].rolling(window=window_size, center=True).mean()
    df['wrist_z_smooth'] = df['bowling_arm_wrist_z'].rolling(window=window_size, center=True).mean()

    df['wrist_x_diff'] = df['wrist_x_smooth'].diff()
    df['wrist_y_diff'] = df['wrist_y_smooth'].diff()
    df['wrist_z_diff'] = df['wrist_z_smooth'].diff()

    df['wrist_velocity'] = np.sqrt(
        df['wrist_x_diff']**2 + df['wrist_y_diff']**2 + df['wrist_z_diff']**2
    )
    
    if df['wrist_velocity'].isnull().all():
        print(f"Warning: Could not calculate wrist velocity for {os.path.basename(csv_path)}.")
        return None, -1 # Return None if data is bad
        
    release_frame_index = df['wrist_velocity'].idxmax()
    release_frame_number = int(df.loc[release_frame_index, 'frame'])

    # 2. Normalize the Timeline
    # Create a new column "Frames from Release" to align the data
    df['frames_from_release'] = df['frame'] - release_frame_number
    
    return df, release_frame_number

# --- Main Script ---
# 1. Process both the user's and the benchmark's data
print("Processing user's performance data...")
user_df, user_release_frame = analyze_performance_data(user_csv_path)

print("Processing benchmark performance data...")
benchmark_df, benchmark_release_frame = analyze_performance_data(benchmark_csv_path)

if user_df is None or benchmark_df is None:
    print("Could not process one or both of the data files. Exiting.")
    exit()

print(f"User release frame: {user_release_frame}, Benchmark release frame: {benchmark_release_frame}")

# 2. Create the Comparison Plot
print("Generating comparison plot...")
plt.style.use('seaborn-v0_8-whitegrid')
fig, ax = plt.subplots(figsize=(15, 8))

# Plot the user's performance
ax.plot(user_df['frames_from_release'], user_df['bowling_arm_elbow_angle'], 
        label='Your Elbow Angle', color='royalblue', linewidth=2.5)

# Plot the benchmark performance
ax.plot(benchmark_df['frames_from_release'], benchmark_df['bowling_arm_elbow_angle'], 
        label='Pro Bowler Elbow Angle', color='firebrick', linestyle='--', linewidth=2)

# Highlight the moment of release (Frame 0 on our new timeline)
ax.axvline(x=0, color='green', linestyle=':', linewidth=2, label='Ball Release')

# 3. Add professional labels and a title
ax.set_title('Performance Comparison vs. Professional Bowler', fontsize=18, fontweight='bold')
ax.set_xlabel('Frames From Ball Release', fontsize=12)
ax.set_ylabel('Angle (Degrees)', fontsize=12)
ax.legend(fontsize=12)
ax.grid(True)
ax.set_ylim(0, 200)

# 4. Save the final report
fig.savefig(output_image_path, dpi=300)
print(f"SUCCESS: Comparison report saved to '{output_image_path}'")
plt.close(fig)

