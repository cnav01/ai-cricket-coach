import streamlit as st
import os
import time
import cv2
import numpy as np
import pandas as pd
import tempfile
import shutil
from PIL import Image

# Import specific functions, objects, and constants directly from your pipeline script
from scripts.cricket_analysis_pipeline import (
    pose,         
    mp_pose,      
    calculate_angle_2d,
    calculate_arm_vertical_angle,
    calculate_arm_horizontal_angle,
    find_arm_head_level_frame_A,
    find_strict_release_frame_B,
    generate_performance_graph,
    generate_annotated_video, 
    generate_generative_ai_feedback,

    # All your constants
    SNAPSHOT_FILENAME_A,
    SNAPSHOT_FILENAME_B,
    PERFORMANCE_GRAPH_FILENAME,
    ANNOTATED_VIDEO_FILENAME, 
    AI_REPORT_FILENAME,
    ANALYSIS_CSV_FILENAME 
)

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

internal_bowler_hand = "right" if bowler_hand_display == "Right-handed" else "left"

# --- Cleanup Logic (NEW) ---
if "temp_dir_to_clean" not in st.session_state:
    st.session_state.temp_dir_to_clean = None

# If there's an old temp dir to clean, do it on the next rerun
if st.session_state.temp_dir_to_clean and os.path.exists(st.session_state.temp_dir_to_clean):
    try:
        shutil.rmtree(st.session_state.temp_dir_to_clean, ignore_errors=True)
        print(f"Cleaned up old temporary directory: {st.session_state.temp_dir_to_clean}")
    except Exception as e:
        print(f"Error during cleanup of old temporary directory {st.session_state.temp_dir_to_clean}: {e}")
    st.session_state.temp_dir_to_clean = None # Reset after cleanup

# Initialize variables that will hold paths/content if analysis succeeds
annotated_video_path_for_display = None # Will store actual path
annotated_video_bytes_for_display = None 
analysis_csv_path_for_download = None
ai_report_path_for_download = None # Will store actual path
ai_report_content_for_display = None 
temp_dir = None 

# --- Analysis Button ---
if st.button("Start Analysis", type="primary"):
    try:
        api_key = st.secrets["GOOGLE_API_KEY"]
    except KeyError:
        st.error("API Key not found. Please create a `.streamlit/secrets.toml` file with your `GOOGLE_API_KEY`.")
        st.stop()

    if uploaded_file is not None:
        st.video(uploaded_file)
        st.caption("Your Uploaded Video")

        try:
            # Create a NEW temp directory and store its path for future cleanup
            temp_dir = tempfile.mkdtemp()
            st.session_state.temp_dir_to_clean = temp_dir # Store for next rerun's cleanup
            
            video_path = os.path.join(temp_dir, uploaded_file.name)
            with open(video_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            # Define output paths within the temporary folder
            output_dir = temp_dir 
            snapshot_output_dir = os.path.join(output_dir, 'snapshots')
            graph_output_dir = os.path.join(output_dir, 'graphs')
            annotated_video_output_dir = os.path.join(output_dir, 'annotated_videos')
            report_output_dir = os.path.join(output_dir, 'reports')
            analysis_csv_path_internal = os.path.join(output_dir, ANALYSIS_CSV_FILENAME) 

            # Ensure all sub-directories exist
            for d in [snapshot_output_dir, graph_output_dir, annotated_video_output_dir, report_output_dir]:
                os.makedirs(d, exist_ok=True)

            st.subheader("Processing...")
            progress_bar = st.progress(0)
            status_text = st.empty()
            
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
            all_frames = [] 

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
                    results = pose.process(image_rgb) 

                    frame_dict = {'frame': frame_count}

                    if results.pose_landmarks and results.pose_world_landmarks:
                        landmarks = results.pose_landmarks.landmark
                        landmarks_3d = results.pose_world_landmarks.landmark
                        
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
                    if total_frames > 0:
                        progress_bar.progress(int((frame_count / total_frames) * (100 / total_progress_steps)))

            cap.release() 

            df = pd.DataFrame(frame_data_list)
            df.fillna(method='ffill', inplace=True)
            df.fillna(method='bfill', inplace=True)
            df.fillna(0, inplace=True) 
            df.to_csv(analysis_csv_path_internal, index=False)
            analysis_csv_path_for_download = analysis_csv_path_internal 

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
                cols[0].image(Image.open(snapshot_A_full_path), caption=f"Frame A: Arm-Head Level (Frame {detected_frame_A})", use_container_width=True)
            if snapshot_B_full_path and os.path.exists(snapshot_B_full_path):
                cols[1].image(Image.open(snapshot_B_full_path), caption=f"Frame B: Strict Release (Frame {detected_frame_B})", use_container_width=True)
            
            current_progress_step += 1
            progress_bar.progress(current_progress_step * 100 // total_progress_steps)
            status_text.text(f"Snapshots generated. ({current_progress_step}/{total_progress_steps})")


            # --- Step 4: Generate Performance Graph ---
            status_text.text(f'Generating performance graph... ({current_progress_step+1}/{total_progress_steps})')
            graph_path = os.path.join(graph_output_dir, PERFORMANCE_GRAPH_FILENAME)
            generate_performance_graph(df, graph_path, detected_frame_A, detected_frame_B)
            if os.path.exists(graph_path):
                st.image(Image.open(graph_path), caption="Key Angles Over Time", use_container_width=True)
            else:
                st.error("Failed to generate performance graph.")
            current_progress_step += 1
            progress_bar.progress(current_progress_step * 100 // total_progress_steps)
            status_text.text(f"Performance graph generated. ({current_progress_step}/{total_progress_steps})")

            # --- Step 5: Generate Annotated Video ---
            status_text.text(f'Generating annotated video... ({current_progress_step+1}/{total_progress_steps})')
            annotated_video_path_for_display = os.path.join(annotated_video_output_dir, ANNOTATED_VIDEO_FILENAME)
            with st.spinner("Creating annotated video (this might take a few minutes for long videos)..."):
                success_video = generate_annotated_video(
                    video_path, annotated_video_path_for_display, internal_bowler_hand, 
                    detected_frame_A, detected_frame_B, analysis_df=df
                )
            
            current_progress_step += 1
            progress_bar.progress(current_progress_step * 100 // total_progress_steps)
            status_text.text(f"Annotated video generated. ({current_progress_step}/{total_progress_steps})")

            # --- Read Annotated Video into Bytes and Display ---
            if annotated_video_path_for_display and os.path.exists(annotated_video_path_for_display):
                with open(annotated_video_path_for_display, "rb") as f:
                    annotated_video_bytes_for_display = f.read() # Read the bytes here
                st.subheader("Annotated Video Playback")
                # Pass bytes directly to st.video
                st.video(annotated_video_bytes_for_display, format="video/mp4", start_time=0)
                st.download_button(
                    label="Download Annotated Video",
                    data=annotated_video_bytes_for_display, # Use the bytes for download button
                    file_name=ANNOTATED_VIDEO_FILENAME,
                    mime="video/mp4"
                )
            else:
                st.error("Annotated video file not found after generation.")
            # ------------------------------------

            # --- Step 6: Generate Gemini AI Feedback ---
            st.subheader("AI Coaching Feedback")
            status_text.text(f'Generating AI coaching report with Gemini... ({current_progress_step+1}/{total_progress_steps})')
            ai_report_path_for_download = os.path.join(report_output_dir, AI_REPORT_FILENAME)
            
            if snapshot_A_full_path and snapshot_B_full_path and os.path.exists(graph_path):
                frame_A_data = df[df['frame'] == detected_frame_A].iloc[0].to_dict()
                frame_B_data = df[df['frame'] == detected_frame_B].iloc[0].to_dict()
                
                ai_success = generate_generative_ai_feedback(
                    internal_bowler_hand, frame_A_data, frame_B_data, 
                    snapshot_A_full_path, snapshot_B_full_path, 
                    graph_path, ai_report_path_for_download, api_key 
                )
                
                # --- Read the AI report content *before* temp dir cleanup ---
                if os.path.exists(ai_report_path_for_download):
                    with open(ai_report_path_for_download, "r", encoding="utf-8") as f:
                        ai_report_content_for_display = f.read()
                else:
                    ai_report_content_for_display = "## AI Coaching Feedback - File Not Found\n\n" \
                                                  "The AI report was expected to be generated but could not be found. Check terminal for Gemini API errors."
            else:
                st.warning("Skipping AI report generation due to missing snapshots or graph files.")
                ai_report_content_for_display = "## AI Coaching Feedback - Skipped\n\n" \
                                              "AI report generation was skipped because key input files (snapshots or graph) were missing."
            
            current_progress_step += 1
            progress_bar.progress(current_progress_step * 100 // total_progress_steps)
            status_text.text(f"AI coaching report generated. ({current_progress_step}/{total_progress_steps})")

            pose.close() 
            time.sleep(0.1) 

            # --- Display AI Report CONTENT here ---
            if ai_report_content_for_display:
                st.markdown(ai_report_content_for_display)
                # Only offer download if the original file path exists, though content is in memory
                if ai_report_path_for_download and os.path.exists(ai_report_path_for_download): 
                     st.download_button(
                        label="Download AI Coaching Report",
                        data=ai_report_content_for_display.encode("utf-8"), # Encode the string data
                        file_name=AI_REPORT_FILENAME,
                        mime="text/plain"
                    )

        except Exception as e:
            st.error(f"An error occurred during analysis: {e}")
            import traceback
            st.exception(e) 
        finally:
            # --- IMPORTANT: REMOVED shutil.rmtree(temp_dir) from here ---
            # Cleanup is now handled by the session_state logic at the top of the script
            pass # No direct cleanup in finally anymore

        # Download raw analysis data - This can stay here
        if analysis_csv_path_for_download and os.path.exists(analysis_csv_path_for_download):
            st.download_button(
                label="Download Full Analysis Data (CSV)",
                data=open(analysis_csv_path_for_download, "rb").read(),
                file_name=ANALYSIS_CSV_FILENAME,
                mime="text/csv"
            )

    else:
        st.error("Please upload a video file to proceed with the analysis.")

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("Developed by Spinvic AI Coach Team")