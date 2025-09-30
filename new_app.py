import streamlit as st
import os
import time
from scripts.frame_capture import ( # Renamed to analysis_pipeline as it's the main version
    process_video_to_csv,
    generate_performance_graph, # Keep this for single video
    generate_annotated_video,
    generate_generative_ai_feedback # Now takes only user data
)

# --- Page Configuration ---
st.set_page_config(
    page_title="AI Cricket Coach",
    page_icon="🏏",
    layout="wide"
)

# --- App Title ---
st.title("AI Cricket Coach: Bowling Performance Lab (Single Player Analysis) 🏏")
st.write("Upload your bowling video to get a data-driven biomechanical analysis and AI coaching feedback.")

# --- File & Parameter Setup ---
st.header("Your Performance Video")
user_video_file = st.file_uploader("Upload your video", type=["mp4", "mov", "avi"], key="user")
user_bowler_hand = st.radio("Select Your Bowling Hand", ('Right', 'Left'), key='user_hand')

# --- Analysis Button ---
if st.button("Analyze Performance", type="primary"):
    # --- Securely get the API Key from secrets.toml ---
    try:
        api_key = st.secrets["GOOGLE_API_KEY"]
    except KeyError:
        st.error("API Key not found. Please create a .streamlit/secrets.toml file with your GOOGLE_API_KEY.")
        st.stop()

    if user_video_file:
        output_dir = "output"
        os.makedirs(output_dir, exist_ok=True)
        
        user_video_path = os.path.join(output_dir, "user_video.mp4")
        user_csv_path = os.path.join(output_dir, "user_analysis.csv")
        performance_graph_path = os.path.join(output_dir, "user_performance_graph.png")
        user_annotated_video_path = os.path.join(output_dir, "user_annotated.mp4")
        ai_report_path = os.path.join(output_dir, "user_ai_report.md")

        with open(user_video_path, "wb") as f:
            f.write(user_video_file.getbuffer())

        st.subheader("Processing...")
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        total_steps = 4 # Reduced total steps for single video analysis

        # 1. Process video to CSV and get detected frames
        status_text.text('Analyzing your video and detecting key frames... (1/4)')
        success, detected_frame_A, detected_frame_B = process_video_to_csv(
            user_video_path, user_bowler_hand.lower(), user_csv_path
        )
        progress_bar.progress(1*100//total_steps)

        if not success:
            st.error("Video processing failed. Could not extract pose data.")
            st.stop()

        # 2. Generate performance graph
        status_text.text('Generating your performance graph... (2/4)')
        graph_success = generate_performance_graph(user_csv_path, performance_graph_path)
        progress_bar.progress(2*100//total_steps)
        if not graph_success:
            st.warning("Could not generate performance graph. Data might be insufficient.")

        # 3. Generate annotated video with highlighted frames
        status_text.text('Generating your annotated video... (3/4)')
        annotated_video_success = generate_annotated_video(
            user_video_path, user_csv_path, user_annotated_video_path, user_bowler_hand.lower(),
            detected_frame_A=detected_frame_A, detected_frame_B=detected_frame_B
        )
        progress_bar.progress(3*100//total_steps)
        if not annotated_video_success:
            st.warning("Could not generate annotated video. Video file or pose data might be problematic.")
        
        # 4. Generate AI coaching report
        status_text.text('Generating AI coaching report with Gemini... (4/4)')
        # IMPORTANT: generate_generative_ai_feedback now expects only user_csv_path, api_key, detected_frame_A, detected_frame_B
        ai_report = generate_generative_ai_feedback(
            user_csv_path, api_key,
            detected_frame_A=detected_frame_A,
            detected_frame_B=detected_frame_B
        )
        progress_bar.progress(4*100//total_steps)
        
        with open(ai_report_path, "w") as f:
            f.write(ai_report)
        
        status_text.success("Analysis Complete!")
        time.sleep(2) # Give user time to read success message
        progress_bar.empty()
        status_text.empty()

        # --- Display Results ---
        st.header("Analysis Results")
        
        st.subheader("Your Performance Analysis")
        # Ensure the annotated video exists before trying to display it
        if os.path.exists(user_annotated_video_path):
            st.video(user_annotated_video_path)
        else:
            st.error("Annotated video could not be generated or found.")

        st.subheader("Performance Graph")
        # Ensure the graph image exists before trying to display it
        if os.path.exists(performance_graph_path):
            st.image(performance_graph_path)
        else:
            st.error("Performance graph could not be generated or found.")
        
        st.subheader("AI-Generated Coaching Report")
        st.markdown(ai_report)

    else:
        st.error("Please upload your video file to proceed.")