import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
import mediapipe as mp
import open3d as o3d

# Import our angle calculator
from scripts.angle_calculator import calculate_angle

def process_video_to_csv(video_path, bowler_hand, output_csv_path):
    """
    Processes a video file to extract 3D pose data and saves it to a CSV.
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

    if not analysis_data:
        print(f"Warning: No pose landmarks detected in {video_path}. CSV will be empty.")
        # Create an empty dataframe with correct columns if no data was generated
        df = pd.DataFrame(columns=[
            "frame", "bowling_arm_elbow_angle", "bowling_arm_shoulder_angle",
            "bowling_arm_wrist_x", "bowling_arm_wrist_y", "bowling_arm_wrist_z"
        ])
    else:
        df = pd.DataFrame(analysis_data)

    df.to_csv(output_csv_path, index=False)
    print(f"SUCCESS: Analysis data saved to {output_csv_path}")
    return True

def generate_comparison_report(user_csv_path, benchmark_csv_path, output_image_path):
    """
    Generates a comparison plot from two performance CSV files.
    """
    def analyze_performance_data(csv_path):
        if not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0:
            print(f"Warning: CSV file is missing or empty: {csv_path}")
            return None, -1
        df = pd.read_csv(csv_path)
        if df.empty:
            print(f"Warning: CSV file is empty after reading: {csv_path}")
            return None, -1
            
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
    """
    if not os.path.exists(user_csv_path) or os.path.getsize(user_csv_path) == 0: return "Could not generate report: No analysis data found."
    df = pd.read_csv(user_csv_path)
    if df.empty: return "Could not generate report: Analysis data is empty."
    
    # Calculate Release Efficiency Score
    window_size = 3
    df['wrist_x_smooth'] = df['bowling_arm_wrist_x'].rolling(window=window_size, center=True).mean()
    df['wrist_y_smooth'] = df['bowling_arm_wrist_y'].rolling(window=window_size, center=True).mean()
    df['wrist_z_smooth'] = df['bowling_arm_wrist_z'].rolling(window=window_size, center=True).mean()
    df['wrist_x_diff'] = df['wrist_x_smooth'].diff()
    df['wrist_y_diff'] = df['wrist_y_smooth'].diff()
    df['wrist_z_diff'] = df['wrist_z_smooth'].diff()
    df['wrist_velocity'] = np.sqrt(df['wrist_x_diff']**2 + df['wrist_y_diff']**2 + df['wrist_z_diff']**2)
    
    if df['wrist_velocity'].isnull().all(): return "Could not calculate release efficiency."

    release_frame_index = df['wrist_velocity'].idxmax()
    peak_velocity = df['wrist_velocity'].max()
    actual_release_velocity = df.loc[release_frame_index, 'wrist_velocity']
    
    if pd.isna(peak_velocity) or peak_velocity == 0:
        release_efficiency = 0
    else:
        release_efficiency = (actual_release_velocity / peak_velocity) * 100

    elbow_angle_at_release = df.loc[release_frame_index, 'bowling_arm_elbow_angle']

    report = "## AI Biomechanical Analysis Report\n\n"
    report += "This report analyzes your bowling action based on 3D motion capture data.\n\n"
    report += f"### Key Performance Metrics:\n\n"
    report += f"- **Release Efficiency Score:** `{release_efficiency:.2f}%`\n"
    report += f"- **Elbow Angle at Release:** `{elbow_angle_at_release:.2f}°`\n\n"
    report += "### Coaching Insights & Suggestions:\n\n"
    
    if release_efficiency >= 95:
        report += "- **Timing:** Excellent! Your release is perfectly timed with your peak arm speed, maximizing the 'whip effect' for great pace.\n"
    elif release_efficiency >= 85:
        report += "- **Timing:** Good, but there's room for improvement. Your release is slightly before your peak arm speed.\n"
    else:
        report += "- **Timing:** This is a key area for improvement. Your release is significantly early, which means you are 'pushing' the ball and losing pace.\n"

    if elbow_angle_at_release < 15:
        report += "- **Elbow Position:** Your arm is very straight at release, which is excellent for pace and legality.\n"
    elif elbow_angle_at_release < 30:
        report += "- **Elbow Position:** Your elbow has a slight bend at release. While legal, focusing on a fully extended arm can help generate more pace.\n"
    else:
        report += "- **Elbow Position:** Your elbow is significantly bent at release. This can limit your pace and may indicate a 'throwing' action.\n"
        
    return report

def generate_annotated_video(video_path, csv_path, output_video_path, bowler_hand):
    """
    Generates a new video with the 2D pose skeleton and angles drawn on it.
    """
    print(f"Generating 2D annotated video for {video_path}...")
    if not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0: return False
    data_df = pd.read_csv(csv_path)
    if data_df.empty: return False
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened(): return False

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0: fps = 30 # Default fps if not available
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

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
                                      mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                                      mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))
            
            # Get the angle for the current frame from our data
            angle_data = data_df[data_df['frame'] == frame_number]
            if not angle_data.empty:
                angle = angle_data['bowling_arm_elbow_angle'].values[0]
                
                if bowler_hand.lower() == "right": elbow_lm = mp_pose.PoseLandmark.RIGHT_ELBOW
                else: elbow_lm = mp_pose.PoseLandmark.LEFT_ELBOW
                
                elbow_coords = (int(results.pose_landmarks.landmark[elbow_lm.value].x * frame_width),
                                int(results.pose_landmarks.landmark[elbow_lm.value].y * frame_height))

                cv2.putText(frame, f"Angle: {int(angle)}", 
                            (elbow_coords[0] + 10, elbow_coords[1]),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

        out.write(frame)
        frame_number += 1
        
    cap.release()
    out.release()
    pose.close()
    print(f"SUCCESS: 2D annotated video saved to {output_video_path}")
    return True

def generate_3d_skeleton_video(video_path, csv_path, output_video_path):
    """
    Generates a looping video of the 3D skeleton visualization.
    """
    print(f"Generating 3D skeleton video for {video_path}...")
    if not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0: return False
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened(): return False
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0: fps = 30
    width, height = 960, 720
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=width, height=height, visible=False)
    
    point_cloud = o3d.geometry.PointCloud()
    line_set = o3d.geometry.LineSet()
    
    vis.add_geometry(point_cloud)
    vis.add_geometry(line_set)

    ctr = vis.get_view_control()
    ctr.set_zoom(0.7)
    ctr.set_front([0, -0.2, -1])
    ctr.set_lookat([0, 0, 0])
    ctr.set_up([0, -1, 0])
    
    connections = mp.solutions.pose.POSE_CONNECTIONS
    pose = mp.solutions.pose.Pose(static_image_mode=True, min_detection_confidence=0.5)

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    for frame_number in range(frame_count):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = cap.read()
        if not ret: continue
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_frame)

        if results.pose_world_landmarks:
            landmarks_3d = results.pose_world_landmarks.landmark
            points_3d = np.array([[lm.x, -lm.y, -lm.z] for lm in landmarks_3d])
            visibilities = np.array([lm.visibility for lm in landmarks_3d])

            visible_points = points_3d[visibilities > 0.5] # Increased visibility threshold
            point_cloud.points = o3d.utility.Vector3dVector(visible_points)
            point_cloud.paint_uniform_color([1.0, 0.0, 1.0])

            lines = []
            for connection in connections:
                a, b = connection
                if visibilities[a] > 0.5 and visibilities[b] > 0.5:
                    lines.append([a, b])
            
            line_set.points = o3d.utility.Vector3dVector(points_3d)
            line_set.lines = o3d.utility.Vector2iVector(lines)
            line_set.paint_uniform_color([0.0, 1.0, 1.0])
            
            vis.update_geometry(point_cloud)
            vis.update_geometry(line_set)
            
            vis.poll_events()
            vis.update_renderer()

            img = vis.capture_screen_float_buffer(False)
            img_np = (np.asarray(img) * 255).astype(np.uint8)
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB_BGR)
            
            out.write(img_bgr)
            
    vis.destroy_window()
    out.release()
    cap.release()
    pose.close()
    print(f"SUCCESS: 3D skeleton video saved to {output_video_path}")
    return True

