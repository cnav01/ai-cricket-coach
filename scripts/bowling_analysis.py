import pandas as pd
import matplotlib.pyplot as plt
import os

input_csv_path = "output/angle_analysis.csv"
output_image_path = "output/plots/elbow_angle_report.png"

if not os.path.exists(input_csv_path):
    print(f"Error: Input file not found at '{input_csv_path}'")
    print("Please run the 'video_analyzer_3d.py' script first to generate the data.")
    exit()

df = pd.read_csv(input_csv_path)

plt.style.use('seaborn-v0_8-whitegrid')

fig, ax = plt.subplots(figsize=(12, 7))

ax.plot(df['frame'], df['right_elbow_angle'], label='Right Elbow Angle', color='royalblue', linewidth=2)
ax.plot(df['frame'], df['left_elbow_angle'], label='Left Elbow Angle', color='firebrick', linewidth=2)

ax.set_title('Elbow Angle Analysis During Bowling Action', fontsize=16, fontweight='bold')
ax.set_xlabel('Frame Number', fontsize=12)
ax.set_ylabel('Angle (Degrees)', fontsize=12)
ax.legend(fontsize=10)
ax.grid(True)

ax.set_ylim(0, 190)

print(f"Saving report to '{output_image_path}'...")
plt.savefig(output_image_path, dpi=300) # Save with high resolution
print("SUCCESS: Report generated successfully!")