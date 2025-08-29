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
    Processes a video file to extract 3D pose data for the entire body and saves it to a CSV.
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
                shoulder_lm, elbow_lm, wrist_lm, hip_lm = (
                    mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_ELBOW,
                    mp_pose.PoseLandmark.RIGHT_WRIST, mp_pose.PoseLandmark.RIGHT_HIP
                )
                front_hip_lm, front_knee_lm, front_ankle_lm = (
                    mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.LEFT_KNEE,
                    mp_pose.PoseLandmark.LEFT_ANKLE
                )
            else: 
                shoulder_lm, elbow_lm, wrist_lm, hip_lm = (
                    mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_ELBOW,
                    mp_pose.PoseLandmark.LEFT_WRIST, mp_pose.PoseLandmark.LEFT_HIP
                )
                front_hip_lm, front_knee_lm, front_ankle_lm = (
                    mp_pose.PoseLandmark.RIGHT_HIP, mp_pose.PoseLandmark.RIGHT_KNEE,
                    mp_pose.PoseLandmark.RIGHT_ANKLE
                )

            def get_coords(landmark_enum):
                lm = landmarks_3d[landmark_enum.value]
                return [lm.x, lm.y, lm.z]

            shoulder_coords = get_coords(shoulder_lm)
            elbow_coords = get_coords(elbow_lm)
            wrist_coords = get_coords(wrist_lm)
            hip_coords = get_coords(hip_lm)
            front_hip_coords = get_coords(front_hip_lm)
            front_knee_coords = get_coords(front_knee_lm)
            front_ankle_coords = get_coords(front_ankle_lm)

            bowling_arm_elbow_angle = calculate_angle(shoulder_coords, elbow_coords, wrist_coords)
            bowling_arm_shoulder_angle = calculate_angle(hip_coords, shoulder_coords, elbow_coords)
            front_leg_brace_angle = calculate_angle(front_hip_coords, front_knee_coords, front_ankle_coords)

            analysis_data.append({
                "frame": frame_number,
                "bowling_arm_elbow_angle": bowling_arm_elbow_angle,
                "bowling_arm_shoulder_angle": bowling_arm_shoulder_angle,
                "front_leg_brace_angle": front_leg_brace_angle,
                "bowling_arm_wrist_x": wrist_coords[0],
                "bowling_arm_wrist_y": wrist_coords[1],
                "bowling_arm_wrist_z": wrist_coords[2]
            })
        frame_number += 1
        
    cap.release()
    pose.close()

    if analysis_data:
        df = pd.DataFrame(analysis_data)
        df.to_csv(output_csv_path, index=False)
        print(f"SUCCESS: Analysis data saved to {output_csv_path}")
        return True
    else:
        print(f"Warning: No pose landmarks detected in {video_path}. CSV not created.")
        return False

def generate_comparison_report(user_csv_path, benchmark_csv_path, output_image_path):
    def analyze_performance_data(csv_path):
        if not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0: return None
        df = pd.read_csv(csv_path)
        if df.empty: return None
            
        window_size = 3
        df['wrist_x_smooth'] = df['bowling_arm_wrist_x'].rolling(window=window_size, center=True).mean()
        df['wrist_y_smooth'] = df['bowling_arm_wrist_y'].rolling(window=window_size, center=True).mean()
        df['wrist_z_smooth'] = df['bowling_arm_wrist_z'].rolling(window=window_size, center=True).mean()
        df['wrist_x_diff'] = df['wrist_x_smooth'].diff()
        df['wrist_y_diff'] = df['wrist_y_smooth'].diff()
        df['wrist_z_diff'] = df['wrist_z_smooth'].diff()
        df['wrist_velocity'] = np.sqrt(df['wrist_x_diff']**2 + df['wrist_y_diff']**2 + df['wrist_z_diff']**2)
        
        if df['wrist_velocity'].isnull().all(): return None
            
        release_frame_index = df['wrist_velocity'].idxmax()
        release_frame_number = int(df.loc[release_frame_index, 'frame'])
        df['frames_from_release'] = df['frame'] - release_frame_number
        return df

    user_df = analyze_performance_data(user_csv_path)
    benchmark_df = analyze_performance_data(benchmark_csv_path)

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

def generate_ai_feedback(user_csv_path, benchmark_csv_path):
    """
    Generates a comparative AI coaching report by analyzing both user and benchmark data.
    """
    def get_metrics_from_csv(csv_path):
        if not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0: return None
        df = pd.read_csv(csv_path)
        if df.empty: return None

        window_size = 3
        df['wrist_x_smooth'] = df['bowling_arm_wrist_x'].rolling(window=window_size, center=True).mean()
        df['wrist_y_smooth'] = df['bowling_arm_wrist_y'].rolling(window=window_size, center=True).mean()
        df['wrist_z_smooth'] = df['bowling_arm_wrist_z'].rolling(window=window_size, center=True).mean()
        df['wrist_x_diff'] = df['wrist_x_smooth'].diff()
        df['wrist_y_diff'] = df['wrist_y_smooth'].diff()
        df['wrist_z_diff'] = df['wrist_z_smooth'].diff()
        df['wrist_velocity'] = np.sqrt(df['wrist_x_diff']**2 + df['wrist_y_diff']**2 + df['wrist_z_diff']**2)
        
        if df['wrist_velocity'].isnull().all(): return None

        release_frame_index = df['wrist_velocity'].idxmax()
        pre_release_df = df[df['frame'] <= release_frame_index]
        ffc_frame_index = pre_release_df['front_leg_brace_angle'].idxmin() if not pre_release_df.empty else release_frame_index

        metrics = {
            "elbow_angle_release": df.loc[release_frame_index, 'bowling_arm_elbow_angle'],
            "shoulder_angle_release": df.loc[release_frame_index, 'bowling_arm_shoulder_angle'],
            "brace_angle_ffc": df.loc[ffc_frame_index, 'front_leg_brace_angle']
        }
        return metrics

    user_metrics = get_metrics_from_csv(user_csv_path)
    benchmark_metrics = get_metrics_from_csv(benchmark_csv_path)

    if not user_metrics or not benchmark_metrics:
        return "Could not generate a comparative report. One or both analysis files are missing or empty."

    report = "## AI Comparative Analysis Report\n\n"
    report += "This report compares your key biomechanical markers against the professional benchmark.\n\n"
    report += "| Metric | Your Performance | Pro Benchmark |\n"
    report += "|:---|:---:|:---:|\n"
    report += f"| Elbow Angle at Release | `{user_metrics['elbow_angle_release']:.1f}°` | `{benchmark_metrics['elbow_angle_release']:.1f}°` |\n"
    report += f"| Shoulder Angle at Release | `{user_metrics['shoulder_angle_release']:.1f}°` | `{benchmark_metrics['shoulder_angle_release']:.1f}°` |\n"
    report += f"| Front Leg Brace at Landing | `{user_metrics['brace_angle_ffc']:.1f}°` | `{benchmark_metrics['brace_angle_ffc']:.1f}°` |\n\n"
    
    report += "### Coaching Insights & Suggestions:\n\n"

    # Elbow Angle Feedback
    elbow_diff = user_metrics['elbow_angle_release'] - benchmark_metrics['elbow_angle_release']
    if elbow_diff > 15:
        report += "- **Elbow Position:** Your arm is significantly more bent at release than the pro's. This is a key area to work on to improve pace and efficiency. Focus on fully extending your arm and 'snapping' your wrist at the release point.\n"
    elif elbow_diff > 5:
        report += "- **Elbow Position:** Your arm is slightly more bent than the pro's at release. While this can be a stylistic difference, striving for a straighter arm like the benchmark can often unlock more speed.\n"
    else:
        report += "- **Elbow Position:** Excellent! Your arm extension at release is very similar to the professional benchmark, indicating an efficient energy transfer.\n"

    # Front Leg Brace Feedback
    brace_diff = user_metrics['brace_angle_ffc'] - benchmark_metrics['brace_angle_ffc']
    if brace_diff < -15:
        report += "- **Front Leg Brace:** Your front knee is collapsing much more than the pro's upon landing. This is a major power leak and increases injury risk. Focus on drills that strengthen your glutes and quads to maintain a firm, braced front leg.\n"
    elif brace_diff < -5:
        report += "- **Front Leg Brace:** Your front leg is slightly less braced than the benchmark. Ensuring a firm plant will help you rotate your upper body faster and more powerfully.\n"
    else:
        report += "- **Front Leg Brace:** Great work! You have a strong and stable front leg, very similar to the professional. This provides a solid foundation for your delivery.\n"
        
    return report

def generate_annotated_video(video_path, csv_path, output_video_path, bowler_hand):
    """
    Generates a new video with the 2D pose skeleton and angles drawn on it.
    """
    if not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0: return False
    data_df = pd.read_csv(csv_path)
    if data_df.empty: return False
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened(): return False

    frame_width, frame_height = int(cap.get(3)), int(cap.get(4))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0 or fps > 60: fps = 30 
    
    # --- THE FIX IS HERE: Using a more compatible codec ---
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    # Add a check to ensure the VideoWriter was created successfully
    if not out.isOpened():
        print(f"Error: Failed to create VideoWriter for {output_video_path}")
        cap.release()
        return False

    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)

    frame_number = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_frame)

        if results.pose_landmarks:
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                      mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
                                      mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2))
            
            angle_data = data_df[data_df['frame'] == frame_number]
            if not angle_data.empty:
                angle = angle_data['bowling_arm_elbow_angle'].values[0]
                
                if bowler_hand.lower() == "right": elbow_lm = mp_pose.PoseLandmark.RIGHT_ELBOW
                else: elbow_lm = mp_pose.PoseLandmark.LEFT_ELBOW
                
                elbow_coords = (int(results.pose_landmarks.landmark[elbow_lm.value].x * frame_width),
                                int(results.pose_landmarks.landmark[elbow_lm.value].y * frame_height))

                cv2.putText(frame, f"Angle: {int(angle)}", (elbow_coords[0] + 10, elbow_coords[1]),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

        out.write(frame)
        frame_number += 1
        
    cap.release(), out.release(), pose.close()
    print(f"SUCCESS: 2D annotated video saved to {output_video_path}")
    return True

