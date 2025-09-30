import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import google.generativeai as genai
from PIL import Image
import toml

# ---- Import your pipeline helpers (you already have them in your script) ----
# Place all the helper functions you provided (calculate_angle, find_arm_head_level_frame_A,
# find_strict_release_frame_B, generate_performance_graph, generate_annotated_video,
# generate_generative_ai_feedback, etc.) above this Streamlit section.

# ---- Streamlit App ----
st.set_page_config(page_title="Bowling Action Analyzer", layout="wide")

st.title("🏏 Bowling Action Analyzer")
st.markdown("Upload a bowling action video and get an in-depth biomechanical analysis.")

# ---- API Key Config ----
gemini_api_key = None
secrets_file_path = "secrets.toml"

if os.path.exists(secrets_file_path):
    try:
        secrets = toml.load(secrets_file_path)
        gemini_api_key = secrets.get("GEMINI_API_KEY")
    except:
        pass

if gemini_api_key is None:
    gemini_api_key = os.environ.get("GEMINI_API_KEY")

if gemini_api_key:
    genai.configure(api_key=gemini_api_key)

# ---- File Upload ----
uploaded_video = st.file_uploader("Upload a bowling video", type=["mp4", "avi", "mov"])
bowler_hand = st.radio("Bowler Hand", ["Right", "Left"])

if uploaded_video:
    # Save uploaded file
    video_path = os.path.join("videos", uploaded_video.name)
    os.makedirs("videos", exist_ok=True)
    with open(video_path, "wb") as f:
        f.write(uploaded_video.read())

    st.video(video_path)

    if st.button("🔍 Run Analysis"):
        with st.spinner("Processing video... this may take a while ⏳"):
            OUTPUT_DIR = "output_analysis"
            os.makedirs(OUTPUT_DIR, exist_ok=True)

            # ---------------- Run your pipeline (main loop from your script) ----------------
            # Collect landmark data into dataframe (adapted from your main())
            mp_pose = mp.solutions.pose
            pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

            cap = cv2.VideoCapture(video_path)
            frame_data_list = []
            frame_count = 0
            frames = []

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = pose.process(rgb_frame)
                if results.pose_landmarks:
                    landmarks = results.pose_landmarks.landmark
                    # Example features – extend with all angles you compute
                    frame_data_list.append({
                        "frame": frame_count,
                        "right_wrist_y": landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y,
                        "nose_y": landmarks[mp_pose.PoseLandmark.NOSE.value].y,
                        # ... include all required angles here ...
                    })
                frame_count += 1
                frames.append(frame)

            cap.release()
            df = pd.DataFrame(frame_data_list)

            # Detect key frames
            frame_A = find_arm_head_level_frame_A(df)
            frame_B = find_strict_release_frame_B(df)

            # Save snapshots
            snapshot_A_path = os.path.join(OUTPUT_DIR, "snapshot_A.jpg")
            snapshot_B_path = os.path.join(OUTPUT_DIR, "snapshot_B.jpg")
            if 0 <= frame_A < len(frames):
                Image.fromarray(cv2.cvtColor(frames[frame_A], cv2.COLOR_BGR2RGB)).save(snapshot_A_path)
            if 0 <= frame_B < len(frames):
                Image.fromarray(cv2.cvtColor(frames[frame_B], cv2.COLOR_BGR2RGB)).save(snapshot_B_path)

            # Graph
            graph_path = os.path.join(OUTPUT_DIR, "performance_graph.png")
            generate_performance_graph(df, graph_path, detected_frame_A=frame_A, detected_frame_B=frame_B)

            # Annotated video
            annotated_video_path = os.path.join(OUTPUT_DIR, "annotated_video.mp4")
            generate_annotated_video(video_path, annotated_video_path, bowler_hand, frame_A, frame_B, df)

            # AI Feedback (if key found)
            report_path = os.path.join(OUTPUT_DIR, "ai_report.txt")
            if gemini_api_key:
                generate_generative_ai_feedback(
                    bowler_hand, 
                    df[df["frame"] == frame_A].iloc[0], 
                    df[df["frame"] == frame_B].iloc[0], 
                    snapshot_A_path, 
                    snapshot_B_path, 
                    graph_path, 
                    report_path
                )

        # ---- Display results ----
        st.success("✅ Analysis complete!")

        col1, col2 = st.columns(2)
        with col1:
            st.image(snapshot_A_path, caption="Frame A: Arm-Head Level")
        with col2:
            st.image(snapshot_B_path, caption="Frame B: Release Point")

        st.image(graph_path, caption="Performance Graph")
        st.video(annotated_video_path)

        if os.path.exists(report_path):
            with open(report_path, "r") as f:
                report_content = f.read()
            st.subheader("📋 AI Analysis Report")
            st.markdown(report_content)

