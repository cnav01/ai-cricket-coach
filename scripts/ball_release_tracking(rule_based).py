import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import collections
import os

# --- Configuration ---
VIDEO_PATH = 'videos/bowling_neww.mp4' # Make sure this path is correct 
SNAPSHOT_OUTPUT_DIR = 'output_snapshots' # Folder to save snapshots 
SNAPSHOT_FILENAME_A = 'detected_arm_head_level_frame_A.jpg'
SNAPSHOT_FILENAME_B = 'detected_strict_release_frame_B.jpg'

# --- Initialize MediaPipe Pose ---
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# --- Helper Function: Calculate Angle (2D) ---
from scripts.angle_calculator import calculate_angle


# --- Helper Function: Calculate Angle with Vertical (3D) ---
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

# --- Helper Function: Calculate Angle with Horizontal (3D) ---
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


# --- FRAME A: Arm and Head Level Detection ---
def find_arm_head_level_frame_A(df):
    """
    Detects Frame A: when the right arm (wrist Y) and nose Y are most level.
    """
    print("\n--- Running Arm-Head Level Frame (A) Detection ---")
    if df.empty: return 0

    df_copy = df.copy() 
    df_copy['frame'] = df_copy['frame'].astype(int)

    # Define a search window for Frame A
    # CRITICAL FIX: To find the "highest" point (physically), we look for the MINIMUM Y-coordinate in MediaPipe.
    overall_min_wrist_y_row_idx = df_copy['right_wrist_y'].idxmin()
    
    # Define a window relative to the physical highest point (which is min_y)
    # Frame A should be BEFORE this highest point.
    start_search_idx = max(0, overall_min_wrist_y_row_idx - 100) 
    end_search_idx = max(0, overall_min_wrist_y_row_idx - 30) # End search 30 frames before min_y

    if start_search_idx >= end_search_idx:
        end_search_idx = start_search_idx + 1 

    window_df = df_copy.iloc[start_search_idx:end_search_idx].copy()
    
    if window_df.empty:
        print("Warning: Search window for Frame A is empty. Falling back to first relevant frame.")
        return df_copy['frame'].iloc[0] if not df_copy.empty else 0

    # Rule: Minimize absolute difference between right wrist Y and nose Y
    window_df['y_diff_arm_head'] = np.abs(window_df['right_wrist_y'] - window_df['nose_y'])
    
    min_diff = window_df['y_diff_arm_head'].min()
    max_diff = window_df['y_diff_arm_head'].max()

    if max_diff > min_diff:
        window_df['score_arm_head_level'] = 1 - ((window_df['y_diff_arm_head'] - min_diff) / (max_diff - min_diff))
    else:
        window_df['score_arm_head_level'] = 0.5 

    # Also add a slight preference for arm not being completely vertical or horizontal, but in between
    window_df['score_arm_mid_angle'] = window_df['right_arm_horizontal_angle_from_plane'].apply(
        lambda x: 1 if 45 <= x <= 75 else 0.2 
    )

    # Combine scores
    window_df['total_score'] = (
        window_df['score_arm_head_level'] * 2.0 + 
        window_df['score_arm_mid_angle'] * 1.0    
    )

    best_frame_idx_in_window = window_df['total_score'].idxmax()
    best_frame_number = int(window_df.loc[best_frame_idx_in_window, 'frame'])
    print(f"SUCCESS: Arm-Head Level Frame A detected at Frame #{best_frame_number} (Total Score: {window_df.loc[best_frame_idx_in_window, 'total_score']:.2f})")
    return best_frame_number


# --- FRAME B: Strict Release Point Detection (ULTRA REFINED + Y-COORD FIX) ---
def find_strict_release_frame_B(df):
    """
    Detects Frame B: The release point based on STRICTEST rules to match the image:
    1. Right elbow angle >= 170 degrees (very straight arm).
    2. Right arm is very close to vertical (e.g., <= 15 degrees from vertical).
    3. High right wrist Y-coordinate (normalized - adjusted for MediaPipe's Y-axis convention).
    4. Left knee angle >= 170 degrees (strong front leg brace).
    """
    print("\n--- Running STRICT Release Frame (B) Detection ---")
    if df.empty: return 0

    df_copy = df.copy()
    df_copy['frame'] = df_copy['frame'].astype(int)

    # --- 1. Define a "Release Window" (around overall PHYSICAL highest wrist Y) ---
    # CRITICAL FIX: To find the "highest" point (physically), we look for the MINIMUM Y-coordinate in MediaPipe.
    overall_min_wrist_y_row_idx = df_copy['right_wrist_y'].idxmin()
    
    # Define a window around this overall physically highest point (min_y)
    start_window_row_idx = max(0, overall_min_wrist_y_row_idx - 20) # Start 20 frames before
    end_window_row_idx = min(len(df_copy), overall_min_wrist_y_row_idx + 15) # End 15 frames after
    
    window_df = df_copy.iloc[start_window_row_idx:end_window_row_idx].copy()

    if window_df.empty:
        print("Warning: Search window for Frame B is empty. Falling back to last frame.")
        return df_copy['frame'].iloc[-1] if not df_copy.empty else 0

    # --- 2. Calculate Scores for Each Strict Rule ---

    # Rule 1: Right Elbow Angle Score (binary 0 or 1) - Very strict threshold
    elbow_extension_threshold = 170 
    window_df['score_right_elbow_angle'] = window_df['right_elbow_angle'].apply(lambda x: 1 if x >= elbow_extension_threshold else 0)

    # Rule 2: Right Arm Verticality Score (binary 0 or 1) - ULTRA strict threshold
    arm_vertical_threshold = 15 # Degrees from perfect vertical (tightened from 20 to 15)
    window_df['score_arm_verticality'] = window_df['right_arm_vertical_angle'].apply(lambda x: 1 if x <= arm_vertical_threshold else 0)
    
    # Rule 3: Right Arm Wrist Height Score (normalized 0-1) - Reinforcing
    # CRITICAL FIX: Invert scoring for Y-coordinate as smaller Y means higher in the image.
    min_wrist_y_in_window = window_df['right_wrist_y'].min()
    max_wrist_y_in_window = window_df['right_wrist_y'].max() # This is the lowest Y (bottom of image) in the window

    if max_wrist_y_in_window > min_wrist_y_in_window:
        # Score is higher if Y-coordinate is smaller (closer to top of image)
        window_df['score_right_wrist_y'] = 1 - ((window_df['right_wrist_y'] - min_wrist_y_in_window) / (max_wrist_y_in_window - min_wrist_y_in_window)) 
    else:
        window_df['score_right_wrist_y'] = 0.5 # Neutral score
        
    # Rule 4: Left Knee Angle Score (binary 0 or 1) - Strong Front Leg Brace
    knee_extension_threshold = 170 
    window_df['score_left_knee_angle'] = window_df['left_knee_angle'].apply(lambda x: 1 if x >= knee_extension_threshold else 0)


    # --- 3. Combine Scores and Find the Best Frame ---
    
    # Heavier weights for the absolutely critical rules to match the image
    weight_right_elbow_angle = 3.0 
    weight_arm_verticality = 4.0 # Heavily weighted for visual key
    weight_right_wrist_y = 1.0 # Reinforcing (now correctly inverted)
    weight_left_knee_angle = 2.0 # Strong contribution from front leg brace

    window_df['total_score'] = (
        window_df['score_right_elbow_angle'] * weight_right_elbow_angle +
        window_df['score_arm_verticality'] * weight_arm_verticality +
        window_df['score_right_wrist_y'] * weight_right_wrist_y +
        window_df['score_left_knee_angle'] * weight_left_knee_angle
    )

    best_frame_idx_in_window = window_df['total_score'].idxmax()
    best_frame_number = int(window_df.loc[best_frame_idx_in_window, 'frame'])

    print(f"SUCCESS: Strict Release Frame B detected at Frame #{best_frame_number} (Total Score: {window_df.loc[best_frame_idx_in_window, 'total_score']:.2f})")
    
    # Optional: Print top 5 frames by score for debugging
    # print("\nTop 5 frames by score for Frame B:")
    # print(window_df[['frame', 'score_right_wrist_y', 'right_elbow_angle', 'right_arm_vertical_angle', 'left_knee_angle', 'total_score']].nlargest(5, 'total_score')))

    return best_frame_number


# --- Main script for standalone testing ---
def main():
    if not os.path.exists(SNAPSHOT_OUTPUT_DIR):
        os.makedirs(SNAPSHOT_OUTPUT_DIR)

    cap = cv2.VideoCapture(VIDEO_PATH)

    if not cap.isOpened():
        print(f"Error: Could not open video file: {VIDEO_PATH}")
        return

    frame_data_list = []
    frame_count = 0
    all_frames = [] # To store frames for saving snapshots later

    print(f"Processing video: {VIDEO_PATH} for pose landmarks and angles...")

    # First pass: Process video to extract landmarks and calculate angles
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        all_frames.append(frame.copy()) # Store a copy of the original frame

        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)

        frame_dict = {'frame': frame_count}

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            
            # Extract 3D landmark coordinates for specific body parts
            right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
            right_elbow = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW]
            right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST]
            nose = landmarks[mp_pose.PoseLandmark.NOSE] # For Frame A
            
            left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
            left_knee = landmarks[mp_pose.PoseLandmark.LEFT_KNEE]
            left_ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE]

            # Store necessary coordinates (normalized)
            frame_dict['right_wrist_x'] = right_wrist.x
            frame_dict['right_wrist_y'] = right_wrist.y
            frame_dict['right_wrist_z'] = right_wrist.z
            frame_dict['right_shoulder_x'] = right_shoulder.x
            frame_dict['right_shoulder_y'] = right_shoulder.y
            frame_dict['right_shoulder_z'] = right_shoulder.z
            frame_dict['nose_y'] = nose.y # For Frame A

            # Calculate and store RIGHT elbow angle (2D)
            elbow_angle = calculate_angle_2d(
                (right_shoulder.x, right_shoulder.y),
                (right_elbow.x, right_elbow.y),
                (right_wrist.x, right_wrist.y)
            )
            frame_dict['right_elbow_angle'] = elbow_angle

            # Calculate and store LEFT knee angle (2D) - now used in Frame B
            knee_angle = calculate_angle_2d(
                (left_hip.x, left_hip.y),
                (left_knee.x, left_knee.y),
                (left_ankle.x, left_ankle.y)
            )
            frame_dict['left_knee_angle'] = knee_angle
            
            # Calculate and store RIGHT arm vertical angle (3D)
            arm_vertical_angle = calculate_arm_vertical_angle(right_shoulder, right_wrist)
            frame_dict['right_arm_vertical_angle'] = arm_vertical_angle
            
            # Store right arm horizontal angle (3D) for mid-angle check in Frame A
            arm_horizontal_angle_from_plane = calculate_arm_horizontal_angle(right_shoulder, right_wrist)
            frame_dict['right_arm_horizontal_angle_from_plane'] = arm_horizontal_angle_from_plane
            
            frame_data_list.append(frame_dict)
        else:
            # Add NaNs for frames with no detection, will be filled
            frame_dict.update({
                'right_wrist_x': np.nan, 'right_wrist_y': np.nan, 'right_wrist_z': np.nan,
                'right_shoulder_x': np.nan, 'right_shoulder_y': np.nan, 'right_shoulder_z': np.nan,
                'nose_y': np.nan,
                'right_elbow_angle': np.nan, 'left_knee_angle': np.nan, 
                'right_arm_vertical_angle': np.nan, 'right_arm_horizontal_angle_from_plane': np.nan
            })
            frame_data_list.append(frame_dict)

        frame_count += 1
        if frame_count % 100 == 0:
            print(f"Processed {frame_count} frames...")

    cap.release()
    print(f"Video processing complete. Total frames: {frame_count}")

    # Convert list of dictionaries to a Pandas DataFrame
    df = pd.DataFrame(frame_data_list)
    
    # Fill any missing angle or coordinate values for robustness
    df.fillna(method='ffill', inplace=True)
    df.fillna(method='bfill', inplace=True)
    df.fillna(0, inplace=True) # Final fill for any remaining NaNs

    # --- Detect both frames ---
    detected_frame_A = find_arm_head_level_frame_A(df)
    detected_frame_B = find_strict_release_frame_B(df)

    # --- Save Snapshots of Detected Frames ---
    # Frame A
    if 0 <= detected_frame_A < len(all_frames):
        snapshot_frame_A = all_frames[detected_frame_A].copy()
        
        cv2.putText(snapshot_frame_A, f"Frame A: Arm-Head Level - {detected_frame_A}", (50, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3, cv2.LINE_AA)
        if detected_frame_A < len(df):
            frame_data = df.loc[df['frame'] == detected_frame_A].iloc[0]
            cv2.putText(snapshot_frame_A, f"Wrist Y: {frame_data['right_wrist_y']:.2f}", (50, 100), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(snapshot_frame_A, f"Nose Y: {frame_data['nose_y']:.2f}", (50, 140), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(snapshot_frame_A, f"Arm Vert: {frame_data['right_arm_vertical_angle']:.1f} deg", (50, 180), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2, cv2.LINE_AA)
        cv2.imwrite(os.path.join(SNAPSHOT_OUTPUT_DIR, SNAPSHOT_FILENAME_A), snapshot_frame_A)
        print(f"Snapshot of Frame A saved to: {os.path.join(SNAPSHOT_OUTPUT_DIR, SNAPSHOT_FILENAME_A)}")
    else:
        print(f"Warning: Detected Frame A ({detected_frame_A}) out of bounds. Cannot save snapshot.")

    # Frame B
    if 0 <= detected_frame_B < len(all_frames):
        snapshot_frame_B = all_frames[detected_frame_B].copy()
        
        cv2.putText(snapshot_frame_B, f"Frame B: STRICT RELEASE - {detected_frame_B}", (50, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3, cv2.LINE_AA)
        if detected_frame_B < len(df):
            frame_data = df.loc[df['frame'] == detected_frame_B].iloc[0]
            cv2.putText(snapshot_frame_B, f"Elbow: {frame_data['right_elbow_angle']:.1f} deg", (50, 100), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(snapshot_frame_B, f"Arm Vert: {frame_data['right_arm_vertical_angle']:.1f} deg", (50, 140), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(snapshot_frame_B, f"Wrist Y: {frame_data['right_wrist_y']:.2f}", (50, 180), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(snapshot_frame_B, f"Knee: {frame_data['left_knee_angle']:.1f} deg", (50, 220), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imwrite(os.path.join(SNAPSHOT_OUTPUT_DIR, SNAPSHOT_FILENAME_B), snapshot_frame_B)
        print(f"Snapshot of Frame B saved to: {os.path.join(SNAPSHOT_OUTPUT_DIR, SNAPSHOT_FILENAME_B)}")
    else:
        print(f"Warning: Detected Frame B ({detected_frame_B}) out of bounds. Cannot save snapshot.")


    # --- Second Pass: Display video with detected frames highlighted ---
    cap = cv2.VideoCapture(VIDEO_PATH)
    current_frame = 0

    print(f"\nDisplaying video. Detected Frame A: {detected_frame_A}, Detected Frame B: {detected_frame_B}")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Mark Frame A
        if current_frame == detected_frame_A:
            cv2.putText(frame, f"Frame A: Arm-Head Level!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
            cv2.rectangle(frame, (0, 0), (frame.shape[1], frame.shape[0]), (255, 0, 0), 10) # Blue border
        
        # Mark Frame B
        if current_frame == detected_frame_B:
            cv2.putText(frame, f"Frame B: STRICT RELEASE!", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.rectangle(frame, (0, 0), (frame.shape[1], frame.shape[0]), (0, 0, 255), 10) # Red border

        cv2.putText(frame, f"Frame: {current_frame}", (50, frame.shape[0] - 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        
        cv2.imshow('Cricket Phase Detection Test', frame)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

        current_frame += 1

    cap.release()
    cv2.destroyAllWindows()
    print("Video display complete.")

if __name__ == "__main__":
    main()