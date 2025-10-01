# scripts/cricket_analysis_pipeline.py

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import collections
import os
import matplotlib.pyplot as plt
import google.generativeai as genai
from PIL import Image


SNAPSHOT_FILENAME_A = 'detected_arm_head_level_frame_A.jpg'
SNAPSHOT_FILENAME_B = 'detected_strict_release_frame_B.jpg'
PERFORMANCE_GRAPH_FILENAME = 'bowling_arm_angles_graph.png'
ANNOTATED_VIDEO_FILENAME = 'annotated_bowling_action.mp4'
AI_REPORT_FILENAME = 'bowling_analysis_report.txt'
ANALYSIS_CSV_FILENAME = 'analysis_data.csv' # <-- Add a constant for just the filename


# --- Initialize MediaPipe Pose ---
mp_pose = mp.solutions.pose
# Keep the global 'pose' object as per your last instruction for the pipeline file
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# --- Helper Function: Calculate Angle (2D) ---
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
    overall_min_wrist_y_row_idx = df_copy['right_wrist_y'].idxmin()
    
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
    overall_min_wrist_y_row_idx = df_copy['right_wrist_y'].idxmin()
    
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
    
    return best_frame_number

# --- Function to Generate Performance Graph ---
# Now accepts graph_output_path as an argument
def generate_performance_graph(df, graph_output_path, detected_frame_A=None, detected_frame_B=None):
    """Generates a graph of key angles over time."""
    plt.figure(figsize=(12, 6))
    plt.plot(df['frame'], df['right_elbow_angle'], label='Right Elbow Angle', color='red')
    plt.plot(df['frame'], df['left_knee_angle'], label='Left Knee Angle', color='blue')
    plt.plot(df['frame'], df['right_arm_vertical_angle'], label='Right Arm Vertical Angle', color='green', linestyle='--')

    if detected_frame_A is not None:
        plt.axvline(x=detected_frame_A, color='orange', linestyle=':', label=f'Frame A (FFC): {detected_frame_A}')

        # Add text for Frame A angle values
        frame_A_data = df[df['frame'] == detected_frame_A]
        if not frame_A_data.empty:
            plt.text(detected_frame_A, plt.ylim()[1]*0.9,
                             f"Elbow:{frame_A_data['right_elbow_angle'].values[0]:.1f}\n"
                             f"Knee:{frame_A_data['left_knee_angle'].values[0]:.1f}\n"
                             f"ArmVert:{frame_A_data['right_arm_vertical_angle'].values[0]:.1f}",
                             color='orange', ha='right', va='top', fontsize=9, bbox=dict(facecolor='white', alpha=0.7))


    if detected_frame_B is not None:
        plt.axvline(x=detected_frame_B, color='purple', linestyle=':', label=f'Frame B (Release): {detected_frame_B}')

        # Add text for Frame B angle values
        frame_B_data = df[df['frame'] == detected_frame_B]
        if not frame_B_data.empty:
            plt.text(detected_frame_B, plt.ylim()[1]*0.8,
                             f"Elbow:{frame_B_data['right_elbow_angle'].values[0]:.1f}\n"
                             f"Knee:{frame_B_data['left_knee_angle'].values[0]:.1f}\n"
                             f"ArmVert:{frame_B_data['right_arm_vertical_angle'].values[0]:.1f}",
                             color='purple', ha='left', va='top', fontsize=9, bbox=dict(facecolor='white', alpha=0.7))


    plt.title('Bowling Arm and Leg Angles Over Time')
    plt.xlabel('Frame Number')
    plt.ylabel('Angle (degrees)')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(graph_output_path) # Uses the passed path
    plt.close()
    print(f"SUCCESS: Performance graph saved to {graph_output_path}")
    return graph_output_path

# --- Function to Generate Annotated Video ---
# Now accepts video_path and output_video_path as arguments
def generate_annotated_video(video_path, output_video_path, bowler_hand, detected_frame_A=None, detected_frame_B=None, analysis_df=None):
    """
    Generates a new video with the 2D pose skeleton, angles drawn on it, and highlights detected key frames.
    """
    cap = cv2.VideoCapture(video_path) # Uses the passed video_path
    if not cap.isOpened():
        print(f"Error: Could not open video file for annotation: {video_path}")
        return False

    frame_width, frame_height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0 or fps > 60: fps = 30 # Cap FPS for compatibility

    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Use mp4v for broader compatibility
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height)) # Uses the passed output_video_path

    if not out.isOpened():
        print(f"Error: Failed to create VideoWriter for {output_video_path}")
        cap.release()
        return False

    # These imports are fine here, or could be global if preferred
    mp_drawing = mp.solutions.drawing_utils
    mp_pose_model = mp.solutions.pose
    pose_model = mp_pose_model.Pose(static_image_mode=True, min_detection_confidence=0.5)

    current_frame = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose_model.process(rgb_frame)

        if results.pose_landmarks:
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose_model.POSE_CONNECTIONS,
                                     mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
                                     mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2))

            if analysis_df is not None and not analysis_df.empty:
                angle_data = analysis_df[analysis_df['frame'] == current_frame]
                if not angle_data.empty:
                    elbow_angle = angle_data['right_elbow_angle'].values[0]
                    shoulder_vertical_angle = angle_data['right_arm_vertical_angle'].values[0]
                    left_knee_angle = angle_data['left_knee_angle'].values[0]

                    # Assuming right-handed bowler for landmarks as per the detection logic
                    elbow_lm = mp_pose_model.PoseLandmark.RIGHT_ELBOW
                    shoulder_lm = mp_pose_model.PoseLandmark.RIGHT_SHOULDER
                    knee_lm = mp_pose_model.PoseLandmark.LEFT_KNEE

                    elbow_coords = (int(results.pose_landmarks.landmark[elbow_lm.value].x * frame_width),
                                    int(results.pose_landmarks.landmark[elbow_lm.value].y * frame_height))
                    shoulder_coords = (int(results.pose_landmarks.landmark[shoulder_lm.value].x * frame_width),
                                       int(results.pose_landmarks.landmark[shoulder_lm.value].y * frame_height))
                    knee_coords = (int(results.pose_landmarks.landmark[knee_lm.value].x * frame_width),
                                   int(results.pose_landmarks.landmark[knee_lm.value].y * frame_height))


                    cv2.putText(frame, f"Elbow: {int(elbow_angle)} deg", (elbow_coords[0] + 10, elbow_coords[1]),
                                 cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
                    cv2.putText(frame, f"Arm Vert: {int(shoulder_vertical_angle)} deg", (shoulder_coords[0] + 10, shoulder_coords[1] - 30),
                                 cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
                    cv2.putText(frame, f"L Knee: {int(left_knee_angle)} deg", (knee_coords[0] + 10, knee_coords[1] - 30),
                                 cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)


        # Mark Frame A
        if detected_frame_A is not None and current_frame == detected_frame_A:
            cv2.putText(frame, f"Frame A: FFC!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
            cv2.rectangle(frame, (0, 0), (frame.shape[1], frame.shape[0]), (255, 0, 0), 10) # Red border for A

        # Mark Frame B
        if detected_frame_B is not None and current_frame == detected_frame_B:
            cv2.putText(frame, f"Frame B: Release!", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.rectangle(frame, (0, 0), (frame.shape[1], frame.shape[0]), (0, 0, 255), 10) # Blue border for B

        cv2.putText(frame, f"Frame: {current_frame}", (frame.shape[1] - 200, frame.shape[0] - 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        out.write(frame)
        current_frame += 1

    cap.release(), out.release(), pose_model.close()
    print(f"SUCCESS: Annotated video saved to {output_video_path}")
    return output_video_path

# --- Function to Generate Gemini AI Feedback ---
def generate_generative_ai_feedback(bowler_hand, frame_A_data, frame_B_data, snapshot_A_path, snapshot_B_path, graph_path, output_report_path, api_key):
    """
    Generates a comprehensive analysis report using Gemini Pro Vision.
    """
    # Configure genai with the passed api_key
    try:
        genai.configure(api_key=api_key)
        # Use a more robust model if available, e.g., gemini-1.5-flash or gemini-1.0-pro-vision
        # 'gemini-2.5-flash' might be a typo for 'gemini-1.5-flash' or an older version.
        # Let's try 'gemini-1.5-flash' which is quite capable.
        model = genai.GenerativeModel('gemini-2.5-flash') 
        print("Gemini model configured successfully.")
    except Exception as e:
        print(f"ERROR: Failed to configure Generative AI model with provided key: {e}")
        return False # Indicate failure

    try:
        with Image.open(snapshot_A_path) as img_A:
            img_A_data = img_A.copy() 
        with Image.open(snapshot_B_path) as img_B:
            img_B_data = img_B.copy()
        with Image.open(graph_path) as img_graph:
            img_graph_data = img_graph.copy()
        print("Images loaded successfully for AI feedback.")
    except FileNotFoundError as e:
        print(f"ERROR: File not found for AI feedback: {e}. Skipping AI report.")
        return False
    except Exception as e:
        print(f"ERROR: An unexpected error occurred while loading images for AI feedback: {e}. Skipping AI report.")
        return False

    text_prompt = f"""
    Analyze the bowling action of a {bowler_hand}-handed bowler based on the provided key frames and angle data.
    
    ### Key Frame A (Front Foot Contact / Arm-Head Level):
    - Frame Number: {int(frame_A_data['frame'])}
    - Right Wrist Y (Normalized, lower is higher): {frame_A_data['right_wrist_y']:.2f}
    - Nose Y (Normalized, lower is higher): {frame_A_data['nose_y']:.2f}
    - Right Arm Vertical Angle (0=vertical): {frame_A_data['right_arm_vertical_angle']:.1f} degrees
    - Right Elbow Angle (2D): {frame_A_data['right_elbow_angle']:.1f} degrees
    - Left Knee Angle (2D, front leg): {frame_A_data['left_knee_angle']:.1f} degrees

    ### Key Frame B (Strict Release Point):
    - Frame Number: {int(frame_B_data['frame'])}
    - Right Wrist Y (Normalized, lower is higher): {frame_B_data['right_wrist_y']:.2f}
    - Right Elbow Angle (2D): {frame_B_data['right_elbow_angle']:.1f} degrees
    - Right Arm Vertical Angle (0=vertical): {frame_B_data['right_arm_vertical_angle']:.1f} degrees
    - Left Knee Angle (2D, front leg): {frame_B_data['left_knee_angle']:.1f} degrees

    ### Instructions for AI Feedback:
    1.  **Overall Assessment:** Provide a general overview of the bowler's technique based on the visual and numerical data.
    2.  **Frame A Analysis (FFC / Arm-Head Level):**
        - Comment on the alignment of the bowling arm (wrist Y) relative to the head (nose Y). Is it well-aligned?
        - Evaluate the right arm's vertical angle. Is it approaching a good position for delivery?
        - Discuss the right elbow angle.
        - Analyze the left knee angle (front leg). Is it braced effectively?
        3.  **Frame B Analysis (Strict Release):**
            - Evaluate the right elbow angle. Is the arm straight or nearly straight?
            - Comment on the right arm's vertical angle. Is it close to vertical for an optimal release?
            - Assess the left knee angle. Is the front leg braced for stability and power?
            - Compare the wrist height (Y) in Frame B to Frame A. Is the arm high at release?
        4.  **Angle Trend Analysis (from Graph):**
            - Look at the `bowling_arm_elbow_angle` and `right_arm_vertical_angle` trends.
            - Are there smooth transitions or any sudden drops/spikes indicating issues?
        5.  **Recommendations:** Offer specific, actionable advice for improvement. Focus on:
            - Arm alignment and extension.
            - Front leg bracing.
            - Body stability.
        6.  **Structure:** Use clear headings and bullet points. Start with a positive aspect, then areas for improvement.

    Be encouraging and focus on biomechanical efficiency for pace and control.
    """

    contents = [
        text_prompt,
        img_A_data,
        img_B_data,
        img_graph_data
    ]

    print("\n--- Generating AI Feedback with Gemini ---")
    try:
        response = model.generate_content(contents)
        report_content = response.text
        
        if not report_content.strip(): # Check if the content is empty or just whitespace
            print("WARNING: Gemini API returned an empty or whitespace-only response for the AI report.")
            # If the report is empty, let's explicitly write a placeholder to the file
            # so the Streamlit app can display *something* and indicate the issue.
            with open(output_report_path, "w") as f:
                f.write("## AI Coaching Feedback - No Content Generated\n\n"
                        "The AI model did not return any feedback. This could be due to:\n"
                        "- An issue with the Google Gemini API key or configuration.\n"
                        "- A temporary API service outage.\n"
                        "- The prompt or images potentially violating content policies (less likely for this use case).\n"
                        "- Rate limiting on the API calls.\n\n"
                        "Please check your `GOOGLE_API_KEY` in `.streamlit/secrets.toml` and try again later.")
            return False # Indicate failure as no meaningful content was generated
            
        with open(output_report_path, "w") as f:
            f.write(report_content)
        print(f"SUCCESS: AI analysis report saved to {output_report_path}")
        return True
    except Exception as e:
        print(f"ERROR: An error occurred while calling Gemini API: {e}")
        # Write an error message to the report file as well
        with open(output_report_path, "w") as f:
            f.write(f"## AI Coaching Feedback - Generation Failed\n\n"
                    f"An error occurred while attempting to generate AI feedback: {e}\n\n"
                    "Please check your internet connection, API key, and console logs for more details.")
        return False