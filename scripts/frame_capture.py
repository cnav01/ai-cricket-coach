import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
import mediapipe as mp
import google.generativeai as genai

# Import our angle calculator
from scripts.angle_calculator import calculate_angle

# --- Helper Functions from the second code snippet ---
# These are placed here as they are general utilities for angle calculations.

def calculate_angle_2d(a, b, c):
    """Calculates angle in degrees between three 2D points (a,b,c with b as vertex)."""
    a = np.array(a) # First point (e.g., shoulder)
    b = np.array(b) # Mid point (e.g., elbow/knee)
    c = np.array(c) # End point (e.g., wrist/ankle)
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle > 180.0:
        angle = 360 - angle
        
    return angle

def calculate_arm_vertical_angle(shoulder_3d, wrist_3d):
    """
    Calculates the angle of the arm vector (shoulder to wrist) with the vertical axis (Y-axis).
    Returns acute angle (0-90 deg), where 0 is perfectly vertical.
    Uses 3D coordinates.
    """
    shoulder_vec = np.array([shoulder_3d.x, shoulder_3d.y, shoulder_3d.z])
    wrist_vec = np.array([wrist_3d.x, wrist_3d.y, wrist_3d.z])
    
    arm_vector = wrist_vec - shoulder_vec
    vertical_axis = np.array([0, 1, 0]) # In MediaPipe Y increases downwards, so [0,1,0] is vertical down.
    
    magnitude_arm = np.linalg.norm(arm_vector)
    
    if magnitude_arm == 0:
        return 0 # Avoid division by zero
        
    dot_product = np.dot(arm_vector, vertical_axis)
    cosine_angle = np.clip(dot_product / magnitude_arm, -1.0, 1.0)
    
    angle_rad = np.arccos(cosine_angle)
    angle_deg = np.degrees(angle_rad)
    
    # Return the acute angle to the vertical line (0-90 degrees)
    return min(angle_deg, 180 - angle_deg)

def calculate_arm_horizontal_angle(shoulder_3d, wrist_3d):
    """
    Calculates the angle of the arm vector (shoulder to wrist) with the horizontal plane (XZ plane).
    Returns acute angle (0-90 deg), where 0 is perfectly horizontal.
    Uses 3D coordinates.
    """
    shoulder_vec = np.array([shoulder_3d.x, shoulder_3d.y, shoulder_3d.z])
    wrist_vec = np.array([wrist_3d.x, wrist_3d.y, wrist_3d.z])
    
    arm_vector = wrist_vec - shoulder_vec
    
    # Project arm vector onto the XZ plane to get horizontal component
    horizontal_component = np.array([arm_vector[0], 0, arm_vector[2]])
    
    magnitude_arm = np.linalg.norm(arm_vector)
    magnitude_horizontal = np.linalg.norm(horizontal_component)
    
    if magnitude_arm == 0:
        return 0 # Avoid division by zero
        
    # Angle between the arm vector and its horizontal projection
    if magnitude_horizontal == 0: # Arm is perfectly vertical
        return 90
    
    dot_product = np.dot(arm_vector, horizontal_component)
    cosine_angle = np.clip(dot_product / (magnitude_arm * magnitude_horizontal), -1.0, 1.0)
    
    angle_rad = np.arccos(cosine_angle)
    angle_deg = np.degrees(angle_rad)
    
    # This angle is the angle with the horizontal plane. 0 is horizontal, 90 is vertical.
    # We want to ensure it's always the acute angle if we're not caring about up/down.
    return angle_deg

# --- Frame Detection Functions ---

def find_arm_head_level_frame_A(df, bowler_hand):
    """
    Detects Frame A: when the bowling arm (wrist Y) and nose Y are most level.
    Adjusted for bowler_hand.
    """
    print("\n--- Running Arm-Head Level Frame (A) Detection ---")
    if df.empty: return 0

    df_copy = df.copy() 
    df_copy['frame'] = df_copy['frame'].astype(int)

    wrist_y_col = 'bowling_arm_wrist_y'
    shoulder_x_col = 'bowling_arm_shoulder_x'
    shoulder_y_col = 'bowling_arm_shoulder_y'
    shoulder_z_col = 'bowling_arm_shoulder_z'
    wrist_x_col = 'bowling_arm_wrist_x'
    wrist_z_col = 'bowling_arm_wrist_z'
    elbow_angle_col = 'bowling_arm_elbow_angle'

    if bowler_hand.lower() == 'right':
        nose_y_col = 'nose_y' # Assuming nose is generally consistent for both
    else: # Left hand bowler - need to ensure correct nose or other body part for "head level"
        # For simplicity, we'll still use 'nose_y' as it's a central head landmark
        nose_y_col = 'nose_y' 

    # Define a search window for Frame A
    # CRITICAL FIX: To find the "highest" point (physically), we look for the MINIMUM Y-coordinate in MediaPipe.
    overall_min_wrist_y_row_idx = df_copy[wrist_y_col].idxmin()
    
    # Frame A should be BEFORE this highest point.
    start_search_idx = max(0, overall_min_wrist_y_row_idx - 100) 
    end_search_idx = max(0, overall_min_wrist_y_row_idx - 30) # End search 30 frames before min_y

    if start_search_idx >= end_search_idx:
        end_search_idx = start_search_idx + 1 

    window_df = df_copy.iloc[start_search_idx:end_search_idx].copy()
    
    if window_df.empty:
        print("Warning: Search window for Frame A is empty. Falling back to first relevant frame.")
        return df_copy['frame'].iloc[0] if not df_copy.empty else 0

    # Rule: Minimize absolute difference between bowling arm wrist Y and nose Y
    window_df['y_diff_arm_head'] = np.abs(window_df[wrist_y_col] - window_df[nose_y_col])
    
    min_diff = window_df['y_diff_arm_head'].min()
    max_diff = window_df['y_diff_arm_head'].max()

    if max_diff > min_diff:
        window_df['score_arm_head_level'] = 1 - ((window_df['y_diff_arm_head'] - min_diff) / (max_diff - min_diff))
    else:
        window_df['score_arm_head_level'] = 0.5 

    # To calculate `right_arm_horizontal_angle_from_plane` we need raw 3D coords, not available directly in this df.
    # For now, we will skip this part, or ensure these are calculated and stored in `process_video_to_csv`.
    # Let's adjust `process_video_to_csv` to store these specific 3D angles for both arms.
    # Assuming these are now available as 'bowling_arm_horizontal_angle_from_plane'
    if 'bowling_arm_horizontal_angle_from_plane' in window_df.columns:
        window_df['score_arm_mid_angle'] = window_df['bowling_arm_horizontal_angle_from_plane'].apply(
            lambda x: 1 if 45 <= x <= 75 else 0.2 
        )
    else:
        window_df['score_arm_mid_angle'] = 0.5 # Neutral if data not available

    # Combine scores
    window_df['total_score'] = (
        window_df['score_arm_head_level'] * 2.0 + 
        window_df['score_arm_mid_angle'] * 1.0     
    )

    best_frame_idx_in_window = window_df['total_score'].idxmax()
    best_frame_number = int(window_df.loc[best_frame_idx_in_window, 'frame'])
    print(f"SUCCESS: Arm-Head Level Frame A detected at Frame #{best_frame_number} (Total Score: {window_df.loc[best_frame_idx_in_window, 'total_score']:.2f})")
    return best_frame_number


def find_strict_release_frame_B(df, bowler_hand):
    """
    Detects Frame B: The release point based on STRICTEST rules to match the image:
    1. Bowling arm elbow angle >= 170 degrees (very straight arm).
    2. Bowling arm is very close to vertical (e.g., <= 15 degrees from vertical).
    3. High bowling arm wrist Y-coordinate (normalized - adjusted for MediaPipe's Y-axis convention).
    4. Front knee angle >= 170 degrees (strong front leg brace).
    Adjusted for bowler_hand.
    """
    print("\n--- Running STRICT Release Frame (B) Detection ---")
    if df.empty: return 0

    df_copy = df.copy()
    df_copy['frame'] = df_copy['frame'].astype(int)

    wrist_y_col = 'bowling_arm_wrist_y'
    elbow_angle_col = 'bowling_arm_elbow_angle'
    arm_vertical_angle_col = 'bowling_arm_vertical_angle'
    front_knee_angle_col = 'front_leg_brace_angle' # This is already in the CSV

    # --- 1. Define a "Release Window" (around overall PHYSICAL highest wrist Y) ---
    # CRITICAL FIX: To find the "highest" point (physically), we look for the MINIMUM Y-coordinate in MediaPipe.
    overall_min_wrist_y_row_idx = df_copy[wrist_y_col].idxmin()
    
    # Define a window around this overall physically highest point (min_y)
    start_window_row_idx = max(0, overall_min_wrist_y_row_idx - 20) # Start 20 frames before
    end_window_row_idx = min(len(df_copy), overall_min_wrist_y_row_idx + 15) # End 15 frames after
    
    window_df = df_copy.iloc[start_window_row_idx:end_window_row_idx].copy()

    if window_df.empty:
        print("Warning: Search window for Frame B is empty. Falling back to last frame.")
        return df_copy['frame'].iloc[-1] if not df_copy.empty else 0

    # --- 2. Calculate Scores for Each Strict Rule ---

    # Rule 1: Bowling Elbow Angle Score (binary 0 or 1) - Very strict threshold
    elbow_extension_threshold = 170 
    window_df['score_elbow_angle'] = window_df[elbow_angle_col].apply(lambda x: 1 if x >= elbow_extension_threshold else 0)

    # Rule 2: Bowling Arm Verticality Score (binary 0 or 1) - ULTRA strict threshold
    arm_vertical_threshold = 15 # Degrees from perfect vertical (tightened from 20 to 15)
    window_df['score_arm_verticality'] = window_df[arm_vertical_angle_col].apply(lambda x: 1 if x <= arm_vertical_threshold else 0)
    
    # Rule 3: Bowling Arm Wrist Height Score (normalized 0-1) - Reinforcing
    # CRITICAL FIX: Invert scoring for Y-coordinate as smaller Y means higher in the image.
    min_wrist_y_in_window = window_df[wrist_y_col].min()
    max_wrist_y_in_window = window_df[wrist_y_col].max() # This is the lowest Y (bottom of image) in the window

    if max_wrist_y_in_window > min_wrist_y_in_window:
        # Score is higher if Y-coordinate is smaller (closer to top of image)
        window_df['score_wrist_y'] = 1 - ((window_df[wrist_y_col] - min_wrist_y_in_window) / (max_wrist_y_in_window - min_wrist_y_in_window)) 
    else:
        window_df['score_wrist_y'] = 0.5 # Neutral score
        
    # Rule 4: Front Knee Angle Score (binary 0 or 1) - Strong Front Leg Brace
    knee_extension_threshold = 170 
    window_df['score_front_knee_angle'] = window_df[front_knee_angle_col].apply(lambda x: 1 if x >= knee_extension_threshold else 0)

    # --- 3. Combine Scores and Find the Best Frame ---
    
    # Heavier weights for the absolutely critical rules to match the image
    weight_elbow_angle = 3.0 
    weight_arm_verticality = 4.0 # Heavily weighted for visual key
    weight_wrist_y = 1.0 # Reinforcing (now correctly inverted)
    weight_front_knee_angle = 2.0 # Strong contribution from front leg brace

    window_df['total_score'] = (
        window_df['score_elbow_angle'] * weight_elbow_angle +
        window_df['score_arm_verticality'] * weight_arm_verticality +
        window_df['score_wrist_y'] * weight_wrist_y +
        window_df['score_front_knee_angle'] * weight_front_knee_angle
    )

    best_frame_idx_in_window = window_df['total_score'].idxmax()
    best_frame_number = int(window_df.loc[best_frame_idx_in_window, 'frame'])

    print(f"SUCCESS: Strict Release Frame B detected at Frame #{best_frame_number} (Total Score: {window_df.loc[best_frame_idx_in_window, 'total_score']:.2f})")
    
    return best_frame_number

def process_video_to_csv(video_path, bowler_hand, output_csv_path):
    """
    Processes a video file to extract 3D pose data and additional angles for Frame A/B detection,
    and saves it to a CSV.
    """
    print(f"Starting analysis for {video_path}...")
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5) # Use False for better tracking across frames
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file: {video_path}")
        return False, None, None # Return False and None for detected frames

    analysis_data = []
    frame_number = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_frame)

        frame_dict = {'frame': frame_number}

        if results.pose_landmarks and results.pose_world_landmarks:
            landmarks_2d = results.pose_landmarks.landmark
            landmarks_3d = results.pose_world_landmarks.landmark
            
            # Determine bowling side landmarks
            if bowler_hand.lower() == "right":
                b_shoulder_lm_2d, b_elbow_lm_2d, b_wrist_lm_2d = (
                    mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_ELBOW, mp_pose.PoseLandmark.RIGHT_WRIST
                )
                b_shoulder_lm_3d, b_elbow_lm_3d, b_wrist_lm_3d = (
                    landmarks_3d[mp_pose.PoseLandmark.RIGHT_SHOULDER.value], 
                    landmarks_3d[mp_pose.PoseLandmark.RIGHT_ELBOW.value],
                    landmarks_3d[mp_pose.PoseLandmark.RIGHT_WRIST.value]
                )
                b_hip_lm_3d = landmarks_3d[mp_pose.PoseLandmark.RIGHT_HIP.value]
                front_hip_lm_3d = landmarks_3d[mp_pose.PoseLandmark.LEFT_HIP.value]
                front_knee_lm_3d = landmarks_3d[mp_pose.PoseLandmark.LEFT_KNEE.value]
                front_ankle_lm_3d = landmarks_3d[mp_pose.PoseLandmark.LEFT_ANKLE.value]
                
            else: # Left-hand bowler
                b_shoulder_lm_2d, b_elbow_lm_2d, b_wrist_lm_2d = (
                    mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_ELBOW, mp_pose.PoseLandmark.LEFT_WRIST
                )
                b_shoulder_lm_3d, b_elbow_lm_3d, b_wrist_lm_3d = (
                    landmarks_3d[mp_pose.PoseLandmark.LEFT_SHOULDER.value], 
                    landmarks_3d[mp_pose.PoseLandmark.LEFT_ELBOW.value],
                    landmarks_3d[mp_pose.PoseLandmark.LEFT_WRIST.value]
                )
                b_hip_lm_3d = landmarks_3d[mp_pose.PoseLandmark.LEFT_HIP.value]
                front_hip_lm_3d = landmarks_3d[mp_pose.PoseLandmark.RIGHT_HIP.value]
                front_knee_lm_3d = landmarks_3d[mp_pose.PoseLandmark.RIGHT_KNEE.value]
                front_ankle_lm_3d = landmarks_3d[mp_pose.PoseLandmark.RIGHT_ANKLE.value]

            # Universal landmarks
            nose_lm_2d = landmarks_2d[mp_pose.PoseLandmark.NOSE.value]
            
            # --- Extract 3D coords (world landmarks) for main angles ---
            b_shoulder_coords_3d = [b_shoulder_lm_3d.x, b_shoulder_lm_3d.y, b_shoulder_lm_3d.z]
            b_elbow_coords_3d = [b_elbow_lm_3d.x, b_elbow_lm_3d.y, b_elbow_lm_3d.z]
            b_wrist_coords_3d = [b_wrist_lm_3d.x, b_wrist_lm_3d.y, b_wrist_lm_3d.z]
            b_hip_coords_3d = [b_hip_lm_3d.x, b_hip_lm_3d.y, b_hip_lm_3d.z]
            front_hip_coords_3d = [front_hip_lm_3d.x, front_hip_lm_3d.y, front_hip_lm_3d.z]
            front_knee_coords_3d = [front_knee_lm_3d.x, front_knee_lm_3d.y, front_knee_lm_3d.z]
            front_ankle_coords_3d = [front_ankle_lm_3d.x, front_ankle_lm_3d.y, front_ankle_lm_3d.z]

            # Calculate main angles using the 'calculate_angle' function (which is 3D)
            bowling_arm_elbow_angle = calculate_angle(b_shoulder_coords_3d, b_elbow_coords_3d, b_wrist_coords_3d)
            bowling_arm_shoulder_angle = calculate_angle(b_hip_coords_3d, b_shoulder_coords_3d, b_elbow_coords_3d)
            front_leg_brace_angle = calculate_angle(front_hip_coords_3d, front_knee_coords_3d, front_ankle_coords_3d)

            # Store main angles and wrist 3D coordinates
            frame_dict.update({
                "bowling_arm_elbow_angle": bowling_arm_elbow_angle,
                "bowling_arm_shoulder_angle": bowling_arm_shoulder_angle,
                "front_leg_brace_angle": front_leg_brace_angle,
                "bowling_arm_wrist_x": b_wrist_lm_3d.x,
                "bowling_arm_wrist_y": b_wrist_lm_3d.y,
                "bowling_arm_wrist_z": b_wrist_lm_3d.z,
                "bowling_arm_shoulder_x": b_shoulder_lm_3d.x,
                "bowling_arm_shoulder_y": b_shoulder_lm_3d.y,
                "bowling_arm_shoulder_z": b_shoulder_lm_3d.z,
            })

            # --- Extract 2D coords (image landmarks) for Frame A/B detection helper functions ---
            # These are for calculations that need image-plane coordinates or specific 3D landmark objects
            # like calculate_arm_vertical_angle which expects landmark objects directly.
            
            # Store necessary Y-coordinates for Frame A detection
            frame_dict['nose_y'] = nose_lm_2d.y
            frame_dict['bowling_arm_wrist_y_2d'] = landmarks_2d[b_wrist_lm_2d.value].y # 2D Y for Frame A's head level check
            
            # Calculate and store bowling arm vertical angle (3D)
            arm_vertical_angle = calculate_arm_vertical_angle(b_shoulder_lm_3d, b_wrist_lm_3d)
            frame_dict['bowling_arm_vertical_angle'] = arm_vertical_angle
            
            # Store bowling arm horizontal angle (3D) for mid-angle check in Frame A
            arm_horizontal_angle_from_plane = calculate_arm_horizontal_angle(b_shoulder_lm_3d, b_wrist_lm_3d)
            frame_dict['bowling_arm_horizontal_angle_from_plane'] = arm_horizontal_angle_from_plane
            
            analysis_data.append(frame_dict)
        else:
            # Add NaNs for frames with no detection
            frame_dict.update({
                'bowling_arm_elbow_angle': np.nan, 'bowling_arm_shoulder_angle': np.nan,
                'front_leg_brace_angle': np.nan, 'bowling_arm_wrist_x': np.nan,
                'bowling_arm_wrist_y': np.nan, 'bowling_arm_wrist_z': np.nan,
                'bowling_arm_shoulder_x': np.nan, 'bowling_arm_shoulder_y': np.nan, 'bowling_arm_shoulder_z': np.nan,
                'nose_y': np.nan, 'bowling_arm_wrist_y_2d': np.nan,
                'bowling_arm_vertical_angle': np.nan, 'bowling_arm_horizontal_angle_from_plane': np.nan
            })
            analysis_data.append(frame_dict)

        frame_number += 1
        
    cap.release()
    pose.close()

    if analysis_data:
        df = pd.DataFrame(analysis_data)
        # Fill any missing angle or coordinate values for robustness
        df.fillna(method='ffill', inplace=True)
        df.fillna(method='bfill', inplace=True)
        df.fillna(0, inplace=True) # Final fill for any remaining NaNs

        df.to_csv(output_csv_path, index=False)
        print(f"SUCCESS: Analysis data saved to {output_csv_path}")

        # Now detect the specific frames
        detected_frame_A = find_arm_head_level_frame_A(df, bowler_hand)
        detected_frame_B = find_strict_release_frame_B(df, bowler_hand)

        return True, detected_frame_A, detected_frame_B
    else:
        print(f"Warning: No pose landmarks detected in {video_path}. CSV not created.")
        return False, None, None

def generate_performance_graph(csv_path, output_image_path):
    """
    Generates a performance graph from a single performance CSV file.
    """
    if not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0: return False
    df = pd.read_csv(csv_path)
    if df.empty: return False
        
    window_size = 3
    df['wrist_x_smooth'] = df['bowling_arm_wrist_x'].rolling(window=window_size, center=True).mean()
    df['wrist_y_smooth'] = df['bowling_arm_wrist_y'].rolling(window=window_size, center=True).mean()
    df['wrist_z_smooth'] = df['bowling_arm_wrist_z'].rolling(window=window_size, center=True).mean()
    df['wrist_x_diff'] = df['wrist_x_smooth'].diff()
    df['wrist_y_diff'] = df['wrist_y_smooth'].diff()
    df['wrist_z_diff'] = df['wrist_z_smooth'].diff()
    df['wrist_velocity'] = np.sqrt(df['wrist_x_diff']**2 + df['wrist_y_diff']**2 + df['wrist_z_diff']**2)
    
    if df['wrist_velocity'].isnull().all(): return False
        
    release_frame_index = df['wrist_velocity'].idxmax() # Default release detection
    release_frame_number = int(df.loc[release_frame_index, 'frame'])
    df['frames_from_release'] = df['frame'] - release_frame_number

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(15, 8))

    ax.plot(df['frames_from_release'], df['bowling_arm_elbow_angle'], label='Elbow Angle', color='royalblue', linewidth=2.5)
    ax.plot(df['frames_from_release'], df['bowling_arm_shoulder_angle'], label='Shoulder Angle', color='seagreen', linewidth=2.5)
    ax.axvline(x=0, color='green', linestyle=':', linewidth=2, label='Ball Release (Velocity Peak)') # Clarify this is velocity peak
    
    ax.set_title('Bowling Performance Analysis', fontsize=18, fontweight='bold')
    ax.set_xlabel('Frames From Ball Release', fontsize=12)
    ax.set_ylabel('Angle (Degrees)', fontsize=12)
    ax.legend(fontsize=12)
    ax.grid(True)
    ax.set_ylim(0, 200)
    
    fig.savefig(output_image_path, dpi=300)
    plt.close(fig)
    print(f"SUCCESS: Performance graph saved to {output_image_path}")
    return True

def generate_annotated_video(video_path, csv_path, output_video_path, bowler_hand, detected_frame_A=None, detected_frame_B=None):
    """
    Generates a new video with the 2D pose skeleton, angles drawn on it, and highlights detected key frames.
    """
    if not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0: return False
    data_df = pd.read_csv(csv_path)
    if data_df.empty: return False
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened(): return False

    frame_width, frame_height = int(cap.get(3)), int(cap.get(4))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0 or fps > 60: fps = 30 # Cap FPS for compatibility
    
    fourcc = cv2.VideoWriter_fourcc(*'avc1') # Use a more compatible codec
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    if not out.isOpened():
        print(f"Error: Failed to create VideoWriter for {output_video_path}")
        cap.release()
        return False

    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5) # Using static_image_mode=True for drawing is fine

    current_frame = 0
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
                elbow_angle = angle_data['bowling_arm_elbow_angle'].values[0]
                shoulder_angle = angle_data['bowling_arm_shoulder_angle'].values[0]
                
                if bowler_hand.lower() == "right": 
                    elbow_lm = mp_pose.PoseLandmark.RIGHT_ELBOW
                    shoulder_lm = mp_pose.PoseLandmark.RIGHT_SHOULDER
                else: 
                    elbow_lm = mp_pose.PoseLandmark.LEFT_ELBOW
                    shoulder_lm = mp_pose.PoseLandmark.LEFT_SHOULDER
                
                elbow_coords = (int(results.pose_landmarks.landmark[elbow_lm.value].x * frame_width),
                                int(results.pose_landmarks.landmark[elbow_lm.value].y * frame_height))
                shoulder_coords = (int(results.pose_landmarks.landmark[shoulder_lm.value].x * frame_width),
                                   int(results.pose_landmarks.landmark[shoulder_lm.value].y * frame_height))

                cv2.putText(frame, f"Elbow: {int(elbow_angle)} deg", (elbow_coords[0] + 10, elbow_coords[1]),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.putText(frame, f"Shoulder: {int(shoulder_angle)} deg", (shoulder_coords[0] + 10, shoulder_coords[1] - 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

        # Mark Frame A
        if detected_frame_A is not None and current_frame == detected_frame_A:
            cv2.putText(frame, f"Frame A: Arm-Head Level!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
            cv2.rectangle(frame, (0, 0), (frame.shape[1], frame.shape[0]), (255, 0, 0), 10) # Blue border
        
        # Mark Frame B
        if detected_frame_B is not None and current_frame == detected_frame_B:
            cv2.putText(frame, f"Frame B: STRICT RELEASE!", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.rectangle(frame, (0, 0), (frame.shape[1], frame.shape[0]), (0, 0, 255), 10) # Red border

        cv2.putText(frame, f"Frame: {current_frame}", (frame.shape[1] - 200, frame.shape[0] - 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        
        out.write(frame)
        current_frame += 1
        
    cap.release(), out.release(), pose.close()
    print(f"SUCCESS: 2D annotated video saved to {output_video_path}")
    return True

# --- Simplified Generative AI Feedback Function (No Comparison) ---
def generate_generative_ai_feedback(user_csv_path, api_key, detected_frame_A=None, detected_frame_B=None):
    """
    Generates an AI coaching report for a single performance using the Gemini API,
    utilizing precise detected frames for metric extraction.
    """
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-flash')
    except Exception as e:
        return f"Error configuring Generative AI model: {e}"

    if not os.path.exists(user_csv_path) or os.path.getsize(user_csv_path) == 0: 
        return "Could not generate report: No analysis data found."
    df = pd.read_csv(user_csv_path)
    if df.empty: 
        return "Could not generate report: Analysis data is empty."

    metrics = {}

    # Use detected frames if available, otherwise fall back to velocity peak for release
    if detected_frame_B is not None and 0 <= detected_frame_B < len(df):
        release_frame_index = df[df['frame'] == detected_frame_B].index[0]
        metrics["elbow_angle_release"] = df.loc[release_frame_index, 'bowling_arm_elbow_angle']
        metrics["shoulder_angle_release"] = df.loc[release_frame_index, 'bowling_arm_shoulder_angle']
    else:
        # Fallback to velocity peak if Frame B not detected or out of bounds
        window_size = 3
        df['wrist_x_smooth'] = df['bowling_arm_wrist_x'].rolling(window=window_size, center=True).mean()
        df['wrist_y_smooth'] = df['bowling_arm_wrist_y'].rolling(window=window_size, center=True).mean()
        df['wrist_z_smooth'] = df['bowling_arm_wrist_z'].rolling(window=window_size, center=True).mean()
        df['wrist_x_diff'] = df['wrist_x_smooth'].diff()
        df['wrist_y_diff'] = df['wrist_y_smooth'].diff()
        df['wrist_z_diff'] = df['wrist_z_smooth'].diff()
        df['wrist_velocity'] = np.sqrt(df['wrist_x_diff']**2 + df['wrist_y_diff']**2 + df['wrist_z_diff']**2)
        
        if df['wrist_velocity'].isnull().all(): return "Could not calculate release metrics (no velocity data)."
        
        release_frame_index = df['wrist_velocity'].idxmax()
        metrics["elbow_angle_release"] = df.loc[release_frame_index, 'bowling_arm_elbow_angle']
        metrics["shoulder_angle_release"] = df.loc[release_frame_index, 'bowling_arm_shoulder_angle']
        print(f"Warning: Frame B not provided or out of bounds. Using velocity peak at frame {df.loc[release_frame_index, 'frame']} for release metrics.")


    if detected_frame_A is not None and 0 <= detected_frame_A < len(df):
        ffc_frame_index = df[df['frame'] == detected_frame_A].index[0]
        metrics["brace_angle_ffc"] = df.loc[ffc_frame_index, 'front_leg_brace_angle']
    else:
        # Fallback to finding minimum brace angle before release if Frame A not detected or out of bounds
        pre_release_df = df[df['frame'] <= release_frame_index]
        if not pre_release_df.empty and 'front_leg_brace_angle' in pre_release_df.columns:
            ffc_frame_index = pre_release_df['front_leg_brace_angle'].idxmin()
            metrics["brace_angle_ffc"] = df.loc[ffc_frame_index, 'front_leg_brace_angle']
            print(f"Warning: Frame A not provided or out of bounds. Using min front leg brace angle before release at frame {df.loc[ffc_frame_index, 'frame']} for FFC metrics.")
        else:
            metrics["brace_angle_ffc"] = 0 # Default if no FFC can be found
            print("Warning: Could not determine FFC brace angle.")


    # --- Create the Prompt for the AI (Single Analysis) ---
    prompt = f"""
    You are an elite cricket biomechanics coach. Analyze the following data from a single bowling performance and provide a detailed, encouraging, and actionable coaching report.

    Here is the data, measured at critical moments in the bowling action:

    | Metric                                | Performance Value          | Ideal Benchmark Range |
    |---------------------------------------|----------------------------|-----------------------|
    | Elbow Angle at Ball Release (Frame {metrics.get('release_frame_number', 'B')})     | {metrics.get('elbow_angle_release', 0):.1f}°         | > 170° (Straighter)   |
    | Shoulder Angle at Ball Release (Frame {metrics.get('release_frame_number', 'B')}) | {metrics.get('shoulder_angle_release', 0):.1f}°         | Varies (Contextual)   |
    | Front Leg Brace Angle at Front Foot Contact (FFC) (Frame {metrics.get('ffc_frame_number', 'A')}) | {metrics.get('brace_angle_ffc', 0):.1f}°         | > 160° (Firmer)       |

    Based on this data, please provide the following in Markdown format:
    1.  **Overall Summary:** A brief, 1-2 sentence summary of the performance, specifically mentioning the identified FFC and Ball Release frames.
    2.  **Detailed Analysis:** A breakdown of each metric. Explain what the number means and how it compares to the ideal benchmark for an efficient action. Clearly reference "Front Foot Contact (FFC)" and "Ball Release" where appropriate.
    3.  **Top Priority for Improvement:** Identify the single most important area to work on based on this data.
    4.  **Suggested Drills:** Suggest one simple, actionable drill to help improve that top priority area.
    """

    try:
        print("Sending data to Generative AI for analysis...")
        response = model.generate_content(prompt)
        print("SUCCESS: Received AI-generated report.")
        return response.text
    except Exception as e:
        return f"Error communicating with the Generative AI model: {e}"