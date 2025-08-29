import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
import mediapipe as mp

# Import our angle calculator
from scripts.angle_calculator import calculate_angle

def process_video_to_csv(video_path, bowler_hand, output_csv_path):
    """
    Processes a video file to extract 3D pose data and saves it to a CSV.
    This function contains the logic from your 'video_analyzer_3d.py' script.
    """
    print(f"Starting analysis for {video_path}...")
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file: {video_path}")
        return False

    analysis_data = []
    frame_number = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_frame)

        if results.pose_world_landmarks:
            landmarks_3d = results.pose_world_landmarks.landmark
            
            if bowler_hand.lower() == "right":
                shoulder_lm, elbow_lm, wrist_lm, hip_lm = (mp_pose.PoseLandmark.RIGHT_SHOULDER, 
                                                            mp_pose.PoseLandmark.RIGHT_ELBOW, 
                                                            mp_pose.PoseLandmark.RIGHT_WRIST, 
                                                            mp_pose.PoseLandmark.RIGHT_HIP)
            else:
                shoulder_lm, elbow_lm, wrist_lm, hip_lm = (mp_pose.PoseLandmark.LEFT_SHOULDER, 
                                                            mp_pose.PoseLandmark.LEFT_ELBOW, 
                                                            mp_pose.PoseLandmark.LEFT_WRIST, 
                                                            mp_pose.PoseLandmark.LEFT_HIP)

            shoulder_coords = [landmarks_3d[shoulder_lm.value].x, landmarks_3d[shoulder_lm.value].y, landmarks_3d[shoulder_lm.value].z]
            elbow_coords = [landmarks_3d[elbow_lm.value].x, landmarks_3d[elbow_lm.value].y, landmarks_3d[elbow_lm.value].z]
            wrist_coords = [landmarks_3d[wrist_lm.value].x, landmarks_3d[wrist_lm.value].y, landmarks_3d[wrist_lm.value].z]
            hip_coords = [landmarks_3d[hip_lm.value].x, landmarks_3d[hip_lm.value].y, landmarks_3d[hip_lm.value].z]

            bowling_arm_elbow_angle = calculate_angle(shoulder_coords, elbow_coords, wrist_coords)
            bowling_arm_shoulder_angle = calculate_angle(hip_coords, shoulder_coords, elbow_coords)

            analysis_data.append({
                "frame": frame_number,
                "bowling_arm_elbow_angle": bowling_arm_elbow_angle,
                "bowling_arm_shoulder_angle": bowling_arm_shoulder_angle,
                "bowling_arm_wrist_x": wrist_coords[0],
                "bowling_arm_wrist_y": wrist_coords[1],
                "bowling_arm_wrist_z": wrist_coords[2]
            })
        frame_number += 1
        
    cap.release()
    pose.close()

    df = pd.DataFrame(analysis_data)
    df.to_csv(output_csv_path, index=False)
    print(f"SUCCESS: Analysis data saved to {output_csv_path}")
    return True

def generate_comparison_report(user_csv_path, benchmark_csv_path, output_image_path):
    """
    Generates a comparison plot from two performance CSV files.
    This function contains the logic from your 'reporting_tool.py' script.
    """
    def analyze_performance_data(csv_path):
        df = pd.read_csv(csv_path)
        window_size = 3
        df['wrist_x_smooth'] = df['bowling_arm_wrist_x'].rolling(window=window_size, center=True).mean()
        df['wrist_y_smooth'] = df['bowling_arm_wrist_y'].rolling(window=window_size, center=True).mean()
        df['wrist_z_smooth'] = df['bowling_arm_wrist_z'].rolling(window=window_size, center=True).mean()
        df['wrist_x_diff'] = df['wrist_x_smooth'].diff()
        df['wrist_y_diff'] = df['wrist_y_smooth'].diff()
        df['wrist_z_diff'] = df['wrist_z_smooth'].diff()
        df['wrist_velocity'] = np.sqrt(df['wrist_x_diff']**2 + df['wrist_y_diff']**2 + df['wrist_z_diff']**2)
        
        if df['wrist_velocity'].isnull().all(): return None, -1
            
        release_frame_index = df['wrist_velocity'].idxmax()
        release_frame_number = int(df.loc[release_frame_index, 'frame'])
        df['frames_from_release'] = df['frame'] - release_frame_number
        return df, release_frame_number

    user_df, _ = analyze_performance_data(user_csv_path)
    benchmark_df, _ = analyze_performance_data(benchmark_csv_path)

    if user_df is None or benchmark_df is None: return False

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(15, 8))

    ax.plot(user_df['frames_from_release'], user_df['bowling_arm_elbow_angle'], label='Your Elbow Angle', color='royalblue', linewidth=2.5)
    ax.plot(benchmark_df['frames_from_release'], benchmark_df['bowling_arm_elbow_angle'], label='Pro Bowler Elbow Angle', color='firebrick', linestyle='--', linewidth=2)
    ax.axvline(x=0, color='green', linestyle=':', linewidth=2, label='Ball Release')
    
    ax.set_title('Performance Comparison vs. Professional Bowler', fontsize=18, fontweight='bold')
    ax.set_xlabel('Frames From Ball Release', fontsize=12)
    ax.set_ylabel('Angle (Degrees)', fontsize=12)
    ax.legend(fontsize=12)
    ax.grid(True)
    ax.set_ylim(0, 200)
    
    fig.savefig(output_image_path, dpi=300)
    plt.close(fig)
    print(f"SUCCESS: Comparison report saved to {output_image_path}")
    return True

def generate_ai_feedback(user_csv_path):
    """
    Analyzes the user's data and generates a detailed coaching report.
    For this MVP, it uses a rule-based system.
    """
    df = pd.read_csv(user_csv_path)
    
    # Calculate Release Efficiency Score
    window_size = 3
    df['wrist_x_smooth'] = df['bowling_arm_wrist_x'].rolling(window=window_size, center=True).mean()
    df['wrist_y_smooth'] = df['bowling_arm_wrist_y'].rolling(window=window_size, center=True).mean()
    df['wrist_z_smooth'] = df['bowling_arm_wrist_z'].rolling(window=window_size, center=True).mean()
    df['wrist_x_diff'] = df['wrist_x_smooth'].diff()
    df['wrist_y_diff'] = df['wrist_y_smooth'].diff()
    df['wrist_z_diff'] = df['wrist_z_smooth'].diff()
    df['wrist_velocity'] = np.sqrt(df['wrist_x_diff']**2 + df['wrist_y_diff']**2 + df['wrist_z_diff']**2)
    
    release_frame_index = df['wrist_velocity'].idxmax()
    peak_velocity = df['wrist_velocity'].max()
    # Let's assume release is within 2 frames of peak for this simple model
    actual_release_velocity = df.loc[release_frame_index, 'wrist_velocity']
    
    # A simple way to handle potential NaN values in peak_velocity
    if pd.isna(peak_velocity) or peak_velocity == 0:
        release_efficiency = 0
    else:
        release_efficiency = (actual_release_velocity / peak_velocity) * 100

    # Get Key Angle at release
    elbow_angle_at_release = df.loc[release_frame_index, 'bowling_arm_elbow_angle']

    # Generate Report
    report = "## AI Biomechanical Analysis Report\n\n"
    report += "This report analyzes your bowling action based on 3D motion capture data.\n\n"
    report += f"### Key Performance Metrics:\n\n"
    report += f"- **Release Efficiency Score:** `{release_efficiency:.2f}%`\n"
    report += f"- **Elbow Angle at Release:** `{elbow_angle_at_release:.2f}°`\n\n"
    
    report += "### Coaching Insights & Suggestions:\n\n"
    
    # Release Efficiency Feedback
    if release_efficiency >= 95:
        report += "- **Timing:** Excellent! Your release is perfectly timed with your peak arm speed, maximizing the 'whip effect' for great pace.\n"
    elif release_efficiency >= 85:
        report += "- **Timing:** Good, but there's room for improvement. Your release is slightly before your peak arm speed. Try to focus on 'throwing the ball through the target' to ensure you accelerate all the way to the release point.\n"
    else:
        report += "- **Timing:** This is a key area for improvement. Your release is significantly early, which means you are 'pushing' the ball and losing a lot of potential pace. A common drill is to practice bowling from a standing position, focusing only on the final arm swing and feeling that 'snap' at the end.\n"

    # Elbow Angle Feedback
    if elbow_angle_at_release < 15:
        report += "- **Elbow Position:** Your arm is very straight at release, which is excellent for pace and legality. Keep this form.\n"
    elif elbow_angle_at_release < 30:
        report += "- **Elbow Position:** Your elbow has a slight bend at release. While legal, focusing on a fully extended arm can help generate more pace and consistency.\n"
    else:
        report += "- **Elbow Position:** Your elbow is significantly bent at release. This can limit your pace and, more importantly, may indicate a 'throwing' action. Focus on keeping your bowling arm as straight as possible throughout the delivery arc.\n"
        
    return report
