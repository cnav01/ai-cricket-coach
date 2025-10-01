import streamlit as st
import os
import time
import cv2 # Keep cv2 import here for reading frames directly
import numpy as np # Keep numpy here for general array ops
import pandas as pd # Keep pandas here for DataFrame ops
from PIL import Image # Keep PIL here for image loading

# Import specific functions and constants from your new pipeline script
from scripts.cricket_analysis_pipeline import (
    get_mediapipe_pose_model, # Caching moved here
    calculate_angle_2d,
    calculate_arm_vertical_angle,
    calculate_arm_horizontal_angle,
    find_arm_head_level_frame_A,
    find_strict_release_frame_B,
    generate_performance_graph,
    generate_annotated_video,
    generate_generative_ai_feedback, # Modified to accept api_key
    mp_pose, # Need mp_pose object for landmarks
    mp_drawing, # Need mp_drawing object for drawing
    SNAPSHOT_FILENAME_A,
    SNAPSHOT_FILENAME_B,
    PERFORMANCE_GRAPH_FILENAME,
    ANNOTATED_VIDEO_FILENAME,
    AI_REPORT_FILENAME,
    ANALYSIS_CSV_PATH # Although we'll build the path dynamically, good to have filename
)

# Initialize MediaPipe Pose Model (cached by Streamlit)
pose = get_mediapipe_pose_model()

# --- Page Configuration ---
st.set_page_config(
    page_title="Spinvic AI Cricket Coach",
    page_icon="🏏",
    layout="wide"
)

# --- App Title ---
st.title("Spinvic AI Cricket Coach: Advanced Biomechanics 🏏")
st.write("Upload your bowling video to get an in-depth biomechanical analysis, including key phase detection, angle graphs, annotated video, and AI-powered coaching feedback.")

# --- File Uploader & Bowler Hand Selection ---
st.sidebar.header("Upload Video & Settings")
uploaded_file = st.sidebar.file_uploader("Upload your bowling video (MP4)", type=["mp4", "mov", "avi"])
bowler_hand_display = st.sidebar.selectbox("Bowler Hand", ["Right-handed", "Left-handed"], index=0)

# Map bowler_hand for internal logic (your pipeline currently assumes 'right', adjust if 'left' support is added)
# For the current pipeline, 'right' for all calculations is hardcoded for the bowling arm.
# If you want to switch arms based on `bowler_hand_display`, you need to modify your pipeline functions
# (e.g., `find_arm_head_level_frame_A`, `find_strict_release_frame_B`, and the landmark extraction loop)
# to dynamically select RIGHT_ELBOW/LEFT_ELBOW etc.
# For now, we'll proceed assuming the pipeline functions internally use RIGHT for bowling arm
# and LEFT for front leg brace. If you select "Left-handed" as the user, it mainly affects the AI prompt
# and a hypothetical future update where the landmark selection in the pipeline becomes dynamic.
internal_bowler_hand = "right" if bowler_hand_display == "Right-handed" else "left"


# --- Analysis Button ---
if st.button("Start Analysis", type="primary"):
    # --- Securely get the API Key from secrets.toml ---
    try:
        api_key = st.secrets["GOOGLE_API_KEY"]
    except KeyError:
        st.error("API Key not found. Please create a `.streamlit/secrets.toml` file with your `GOOGLE_API_KEY`.")
        st.stop()

    if uploaded_file is not None:
        st.video(uploaded_file, caption="Your uploaded video.")

        # Create a temporary directory for outputs
        with tempfile.TemporaryDirectory() as temp_dir:
            video_path = os.path.join(temp_dir, uploaded_file.name)
            with open(video_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            # Define output paths within the temporary folder
            output_dir = temp_dir # Use the temp_dir as the base output directory
            snapshot_output_dir = os.path.join(output_dir, 'snapshots')
            graph_output_dir = os.path.join(output_dir, 'graphs')
            annotated_video_output_dir = os.path.join(output_dir, 'annotated_videos')
            report_output_dir = os.path.join(output_dir, 'reports')
            analysis_csv_path = os.path.join(output_dir, 'analysis_data.csv') # Using the pipeline's constant filename

            # Ensure all sub-directories exist
            for d in [snapshot_output_dir, graph_output_dir, annotated_video_output_dir, report_output_dir]:
                os.makedirs(d, exist_ok=True)

            st.subheader("Processing...")
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Estimate total steps for the progress bar
            # 1 for video processing, 2 for detection, 1 for graph, 1 for annotated video, 1 for AI report = 6 steps
            total_progress_steps = 6 
            current_progress_step = 0

            # --- Step 1: Process video to extract landmarks and angles ---
            status_text.text('Analyzing video for pose landmarks and angles... (This may take a while for longer videos)')
            
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                st.error(f"Error: Could not open video file: {video_path}")
                st.stop()

            frame_data_list = []
            frame_count = 0
            all_frames = [] # To store frames for saving snapshots later

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_frames == 0:
                st.error("Uploaded video contains no frames.")
                cap.release()
                st.stop()
            
            with st.spinner("Extracting pose data..."):
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    all_frames.append(frame.copy()) 

                    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    # Use the imported pose model
                    results = pose.process(image_rgb) 

                    frame_dict = {'frame': frame_count}

                    if results.pose_landmarks and results.pose_world_landmarks:
                        landmarks = results.pose_landmarks.landmark
                        landmarks_3d = results.pose_world_landmarks.landmark
                        
                        # Assuming right-handed bowler for landmark extraction as per pipeline's current logic
                        right_shoulder_3d = landmarks_3d[mp_pose.PoseLandmark.RIGHT_SHOULDER]
                        right_elbow_3d = landmarks_3d[mp_pose.PoseLandmark.RIGHT_ELBOW]
                        right_wrist_3d = landmarks_3d[mp_pose.PoseLandmark.RIGHT_WRIST]
                        left_hip_3d = landmarks_3d[mp_pose.PoseLandmark.LEFT_HIP]
                        left_knee_3d = landmarks_3d[mp_pose.PoseLandmark.LEFT_KNEE]
                        left_ankle_3d = landmarks_3d[mp_pose.PoseLandmark.LEFT_ANKLE]

                        right_shoulder_2d = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
                        right_elbow_2d = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW]
                        right_wrist_2d = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST]
                        nose_2d = landmarks[mp_pose.PoseLandmark.NOSE]
                        left_hip_2d = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
                        left_knee_2d = landmarks[mp_pose.PoseLandmark.LEFT_KNEE]
                        left_ankle_2d = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE]

                        frame_dict['right_wrist_x'] = right_wrist_2d.x
                        frame_dict['right_wrist_y'] = right_wrist_2d.y
                        frame_dict['right_wrist_z'] = right_wrist_2d.z
                        frame_dict['right_shoulder_x'] = right_shoulder_2d.x
                        frame_dict['right_shoulder_y'] = right_shoulder_2d.y
                        frame_dict['right_shoulder_z'] = right_shoulder_2d.z
                        frame_dict['nose_y'] = nose_2d.y

                        elbow_angle = calculate_angle_2d(
                            (right_shoulder_2d.x, right_shoulder_2d.y),
                            (right_elbow_2d.x, right_elbow_2d.y),
                            (right_wrist_2d.x, right_wrist_2d.y)
                        )
                        frame_dict['right_elbow_angle'] = elbow_angle

                        knee_angle = calculate_angle_2d(
                            (left_hip_2d.x, left_hip_2d.y),
                            (left_knee_2d.x, left_knee_2d.y),
                            (left_ankle_2d.x, left_ankle_2d.y)
                        )
                        frame_dict['left_knee_angle'] = knee_angle
                        
                        arm_vertical_angle = calculate_arm_vertical_angle(right_shoulder_3d, right_wrist_3d)
                        frame_dict['right_arm_vertical_angle'] = arm_vertical_angle
                        
                        arm_horizontal_angle_from_plane = calculate_arm_horizontal_angle(right_shoulder_3d, right_wrist_3d)
                        frame_dict['right_arm_horizontal_angle_from_plane'] = arm_horizontal_angle_from_plane
                        
                        frame_data_list.append(frame_dict)
                    else:
                        frame_dict.update({
                            'right_wrist_x': np.nan, 'right_wrist_y': np.nan, 'right_wrist_z': np.nan,
                            'right_shoulder_x': np.nan, 'right_shoulder_y': np.nan, 'right_shoulder_z': np.nan,
                            'nose_y': np.nan,
                            'right_elbow_angle': np.nan, 'left_knee_angle': np.nan, 
                            'right_arm_vertical_angle': np.nan, 'right_arm_horizontal_angle_from_plane': np.nan
                        })
                        frame_data_list.append(frame_dict)

                    frame_count += 1
                    # Update progress bar
                    if total_frames > 0:
                        progress_bar.progress(int((frame_count / total_frames) * (100 / total_progress_steps)))


            cap.release()
            df = pd.DataFrame(frame_data_list)
            df.fillna(method='ffill', inplace=True)
            df.fillna(method='bfill', inplace=True)
            df.fillna(0, inplace=True) 
            df.to_csv(analysis_csv_path, index=False)
            current_progress_step += 1
            progress_bar.progress(current_progress_step * 100 // total_progress_steps)
            status_text.text(f"Processed video data. ({current_progress_step}/{total_progress_steps})")
            
            if df.empty or 'frame' not in df.columns or len(df) < 5:
                st.error("Not enough valid pose data was extracted from the video for analysis. Please try a different video.")
                st.stop()


            # --- Step 2: Detect Key Frames ---
            status_text.text(f'Detecting key phases... ({current_progress_step+1}/{total_progress_steps})')
            detected_frame_A = find_arm_head_level_frame_A(df)
            detected_frame_B = find_strict_release_frame_B(df)
            current_progress_step += 1
            progress_bar.progress(current_progress_step * 100 // total_progress_steps)
            status_text.text(f"Key phases detected. ({current_progress_step}/{total_progress_steps})")

            st.subheader("Key Phase Detection")
            st.write(f"**Detected Frame A (Front Foot Contact / Arm-Head Level):** Frame #{detected_frame_A}")
            st.write(f"**Detected Frame B (Strict Release Point):** Frame #{detected_frame_B}")

            st.subheader("Visual Analysis Outputs")

            # --- Step 3: Save Snapshots of Detected Frames ---
            snapshot_A_full_path = os.path.join(snapshot_output_dir, SNAPSHOT_FILENAME_A)
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
                cv2.imwrite(snapshot_A_full_path, snapshot_frame_A)
            else:
                st.warning(f"Warning: Detected Frame A ({detected_frame_A}) out of bounds. Cannot save snapshot.")
                snapshot_A_full_path = None

            snapshot_B_full_path = os.path.join(snapshot_output_dir, SNAPSHOT_FILENAME_B)
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
                cv2.imwrite(snapshot_B_full_path, snapshot_frame_B)
            else:
                st.warning(f"Warning: Detected Frame B ({detected_frame_B}) out of bounds. Cannot save snapshot.")
                snapshot_B_full_path = None

            cols = st.columns(2)
            if snapshot_A_full_path and os.path.exists(snapshot_A_full_path):
                cols[0].image(Image.open(snapshot_A_full_path), caption=f"Frame A: Arm-Head Level (Frame {detected_frame_A})", use_column_width=True)
            if snapshot_B_full_path and os.path.exists(snapshot_B_full_path):
                cols[1].image(Image.open(snapshot_B_full_path), caption=f"Frame B: Strict Release (Frame {detected_frame_B})", use_column_width=True)
            
            current_progress_step += 1
            progress_bar.progress(current_progress_step * 100 // total_progress_steps)
            status_text.text(f"Snapshots generated. ({current_progress_step}/{total_progress_steps})")


            # --- Step 4: Generate Performance Graph ---
            status_text.text(f'Generating performance graph... ({current_progress_step+1}/{total_progress_steps})')
            graph_path = os.path.join(graph_output_dir, PERFORMANCE_GRAPH_FILENAME)
            generate_performance_graph(df, graph_path, detected_frame_A, detected_frame_B)
            if os.path.exists(graph_path):
                st.image(graph_path, caption="Key Angles Over Time", use_column_width=True)
            else:
                st.error("Failed to generate performance graph.")
            current_progress_step += 1
            progress_bar.progress(current_progress_step * 100 // total_progress_steps)
            status_text.text(f"Performance graph generated. ({current_progress_step}/{total_progress_steps})")


            # --- Step 5: Generate Annotated Video ---
            status_text.text(f'Generating annotated video... ({current_progress_step+1}/{total_progress_steps})')
            annotated_video_path = os.path.join(annotated_video_output_dir, ANNOTATED_VIDEO_FILENAME)
            with st.spinner("Creating annotated video (this might take a few minutes for long videos)..."):
                # Pass the dynamically determined internal_bowler_hand
                success_video = generate_annotated_video(
                    video_path, annotated_video_path, internal_bowler_hand, 
                    detected_frame_A, detected_frame_B, analysis_df=df
                )
            if success_video and os.path.exists(annotated_video_path):
                st.video(annotated_video_path, format="video/mp4", start_time=0)
                st.download_button(
                    label="Download Annotated Video",
                    data=open(annotated_video_path, "rb").read(),
                    file_name=ANNOTATED_VIDEO_FILENAME,
                    mime="video/mp4"
                )
            else:
                st.error("Failed to generate annotated video. Please check the video format or content.")
            current_progress_step += 1
            progress_bar.progress(current_progress_step * 100 // total_progress_steps)
            status_text.text(f"Annotated video generated. ({current_progress_step}/{total_progress_steps})")


            # --- Step 6: Generate Gemini AI Feedback ---
            st.subheader("AI Coaching Feedback")
            status_text.text(f'Generating AI coaching report with Gemini... ({current_progress_step+1}/{total_progress_steps})')
            ai_report_path = os.path.join(report_output_dir, AI_REPORT_FILENAME)
            
            if snapshot_A_full_path and snapshot_B_full_path and os.path.exists(graph_path):
                frame_A_data = df[df['frame'] == detected_frame_A].iloc[0].to_dict() # Convert to dict for easier passing
                frame_B_data = df[df['frame'] == detected_frame_B].iloc[0].to_dict()
                
                ai_success = generate_generative_ai_feedback(
                    internal_bowler_hand, frame_A_data, frame_B_data, 
                    snapshot_A_full_path, snapshot_B_full_path, 
                    graph_path, ai_report_path, api_key # Pass the API key
                )
                if ai_success and os.path.exists(ai_report_path):
                    with open(ai_report_path, "r") as f:
                        st.markdown(f.read())
                    st.download_button(
                        label="Download AI Coaching Report",
                        data=open(ai_report_path, "rb").read(),
                        file_name=AI_REPORT_FILENAME,
                        mime="text/plain"
                    )
                else:
                    st.error("Failed to generate AI coaching feedback. This might be due to API issues or a problem with input files.")
            else:
                st.warning("Skipping AI report generation due to missing snapshots or graph files.")
            
            current_progress_step += 1
            progress_bar.progress(current_progress_step * 100 // total_progress_steps)
            status_text.text(f"AI coaching report generated. ({current_progress_step}/{total_progress_steps})")

            progress_bar.empty()
            status_text.success("Analysis Complete!")
            st.balloons() # Celebrate completion

            # Optional: Download raw analysis data
            st.download_button(
                label="Download Full Analysis Data (CSV)",
                data=open(analysis_csv_path, "rb").read(),
                file_name=os.path.basename(analysis_csv_path), # Get just the filename
                mime="text/csv"
            )

    else:
        st.error("Please upload a video file to proceed with the analysis.")

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("Developed by Spinvic AI Coach Team")