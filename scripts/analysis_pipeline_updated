import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
import mediapipe as mp
import google.generativeai as genai

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
    
    landmarks_to_save_raw = [
        mp_pose.PoseLandmark.NOSE, mp_pose.PoseLandmark.LEFT_EYE_INNER, mp_pose.PoseLandmark.LEFT_EYE,
        mp_pose.PoseLandmark.LEFT_EYE_OUTER, mp_pose.PoseLandmark.RIGHT_EYE_INNER,
        mp_pose.PoseLandmark.RIGHT_EYE, mp_pose.PoseLandmark.RIGHT_EYE_OUTER,
        mp_pose.PoseLandmark.LEFT_EAR, mp_pose.PoseLandmark.RIGHT_EAR,
        mp_pose.PoseLandmark.MOUTH_LEFT, mp_pose.PoseLandmark.MOUTH_RIGHT,
        mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.RIGHT_SHOULDER,
        mp_pose.PoseLandmark.LEFT_ELBOW, mp_pose.PoseLandmark.RIGHT_ELBOW,
        mp_pose.PoseLandmark.LEFT_WRIST, mp_pose.PoseLandmark.RIGHT_WRIST,
        mp_pose.PoseLandmark.LEFT_PINKY, mp_pose.PoseLandmark.RIGHT_PINKY,
        mp_pose.PoseLandmark.LEFT_INDEX, mp_pose.PoseLandmark.RIGHT_INDEX,
        mp_pose.PoseLandmark.LEFT_THUMB, mp_pose.PoseLandmark.RIGHT_THUMB,
        mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.RIGHT_HIP,
        mp_pose.PoseLandmark.LEFT_KNEE, mp_pose.PoseLandmark.RIGHT_KNEE,
        mp_pose.PoseLandmark.LEFT_ANKLE, mp_pose.PoseLandmark.RIGHT_ANKLE,
        mp_pose.PoseLandmark.LEFT_HEEL, mp_pose.PoseLandmark.RIGHT_HEEL,
        mp_pose.PoseLandmark.LEFT_FOOT_INDEX, mp_pose.PoseLandmark.RIGHT_FOOT_INDEX
    ]
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_frame)

        if results.pose_world_landmarks:
            landmarks_3d = results.pose_world_landmarks.landmark
            
            frame_data = {'frame': frame_number}

            for lm in landmarks_to_save_raw:
                lm_coords = landmarks_3d[lm.value]
                lm_name = lm.name.lower()
                frame_data[f'{lm_name}_x'] = lm_coords.x
                frame_data[f'{lm_name}_y'] = lm_coords.y
                frame_data[f'{lm_name}_z'] = lm_coords.z

            if bowler_hand.lower() == "right":
                bowling_shoulder_lm, bowling_elbow_lm, bowling_wrist_lm, bowling_hip_lm = (
                    mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_ELBOW,
                    mp_pose.PoseLandmark.RIGHT_WRIST, mp_pose.PoseLandmark.RIGHT_HIP
                )
                front_hip_lm, front_knee_lm, front_ankle_lm = (
                    mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.LEFT_KNEE,
                    mp_pose.PoseLandmark.LEFT_ANKLE
                )
            else: 
                bowling_shoulder_lm, bowling_elbow_lm, bowling_wrist_lm, bowling_hip_lm = (
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

            shoulder_coords = get_coords(bowling_shoulder_lm)
            elbow_coords = get_coords(bowling_elbow_lm)
            wrist_coords = get_coords(bowling_wrist_lm)

            hip_coords = get_coords(bowling_hip_lm)

            front_hip_coords = get_coords(front_hip_lm)
            front_knee_coords = get_coords(front_knee_lm)
            front_ankle_coords = get_coords(front_ankle_lm)

            frame_data["bowling_arm_elbow_angle"] = calculate_angle(shoulder_coords, elbow_coords, wrist_coords)
            frame_data["bowling_arm_shoulder_angle"] = calculate_angle(hip_coords, shoulder_coords, elbow_coords)
            frame_data["front_leg_brace_angle"] = calculate_angle(front_hip_coords, front_knee_coords, front_ankle_coords)

            frame_data["bowling_arm_wrist_x"] = wrist_coords[0]
            frame_data["bowling_arm_wrist_y"] = wrist_coords[1]
            frame_data["bowling_arm_wrist_z"] = wrist_coords[2]

            analysis_data.append(frame_data)

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
    if fps == 0 or fps > 60: fps = 30 # Cap FPS for compatibility
    
    fourcc = cv2.VideoWriter_fourcc(*'avc1') # Use a more compatible codec
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

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

# --- NEW: Generative AI Feedback Function ---
def generate_generative_ai_feedback(user_csv_path, benchmark_csv_path, api_key):
    """
    Generates a comparative AI coaching report using the Gemini API.
    """
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-flash')
    except Exception as e:
        return f"Error configuring Generative AI model: {e}"

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

    # --- Create the Prompt for the AI ---
    prompt = f"""
    You are an elite cricket biomechanics coach. Analyze the following data comparing an amateur bowler to a professional benchmark and provide a detailed, encouraging, and actionable coaching report.

    Here is the data, measured at critical moments in the bowling action:

    | Metric                     | Amateur Bowler | Professional Benchmark |
    |----------------------------|----------------|------------------------|
    | Elbow Angle at Release     | {user_metrics['elbow_angle_release']:.1f}°      | {benchmark_metrics['elbow_angle_release']:.1f}°               |
    | Shoulder Angle at Release  | {user_metrics['shoulder_angle_release']:.1f}°      | {benchmark_metrics['shoulder_angle_release']:.1f}°               |
    | Front Leg Brace at Landing | {user_metrics['brace_angle_ffc']:.1f}°      | {benchmark_metrics['brace_angle_ffc']:.1f}°               |

    Based on this data, please provide the following in Markdown format:
    1.  **Overall Summary:** A brief, 1-2 sentence summary of the key differences.
    2.  **Detailed Analysis:** A breakdown of each metric. For each, explain what the number means and compare the amateur's performance to the professional's.
    3.  **Top Priority for Improvement:** Identify the single most important area the amateur should work on.
    4.  **Suggested Drills:** Suggest one simple, actionable drill to help improve that top priority area.
    """

    try:
        print("Sending data to Generative AI for analysis...")
        response = model.generate_content(prompt)
        print("SUCCESS: Received AI-generated report.")
        return response.text
    except Exception as e:
        return f"Error communicating with the Generative AI model: {e}"