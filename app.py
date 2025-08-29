import streamlit as st
import os
import time
from scripts.analysis_pipeline_MAIN  import (
    process_video_to_csv,
    generate_comparison_report,
    generate_generative_ai_feedback, 
    generate_annotated_video
)

# --- Page Configuration ---
st.set_page_config(
    page_title="AI Cricket Coach",
    page_icon="🏏",
    layout="wide"
)

# --- App Title ---
st.title("AI Cricket Coach: Bowling Performance Lab 🏏")
st.write("Upload your bowling video and a professional's video to get a data-driven biomechanical analysis.")

# --- File & Parameter Setup in Columns ---
col1, col2 = st.columns(2)

with col1:
    st.header("Your Performance Video")
    user_video_file = st.file_uploader("Upload your video", type=["mp4", "mov", "avi"], key="user")
    user_bowler_hand = st.radio("Select Your Bowling Hand", ('Right', 'Left'), key='user_hand')

with col2:
    st.header("Benchmark Video (Professional)")
    benchmark_video_file = st.file_uploader("Upload a pro's video", type=["mp4", "mov", "avi"], key="benchmark")
    benchmark_bowler_hand = st.radio("Select Pro's Bowling Hand", ('Right', 'Left'), key='benchmark_hand')

# --- Analysis Button ---
if st.button("Analyze Performance", type="primary"):
    # --- Securely get the API Key from secrets.toml ---
    try:
        api_key = st.secrets["GOOGLE_API_KEY"]
    except KeyError:
        st.error("API Key not found. Please create a .streamlit/secrets.toml file with your GOOGLE_API_KEY.")
        st.stop()

    if user_video_file and benchmark_video_file:
        
        output_dir = "output"
        os.makedirs(output_dir, exist_ok=True)
        
        user_video_path = os.path.join(output_dir, "user_video.mp4")
        benchmark_video_path = os.path.join(output_dir, "benchmark_video.mp4")
        
        user_csv_path = os.path.join(output_dir, "user_analysis.csv")
        benchmark_csv_path = os.path.join(output_dir, "benchmark_analysis.csv")
        
        report_image_path = os.path.join(output_dir, "comparison_report.png")
        user_annotated_video_path = os.path.join(output_dir, "user_annotated.mp4")
        benchmark_annotated_video_path = os.path.join(output_dir, "benchmark_annotated.mp4")

        with open(user_video_path, "wb") as f:
            f.write(user_video_file.getbuffer())
        with open(benchmark_video_path, "wb") as f:
            f.write(benchmark_video_file.getbuffer())

        st.subheader("Processing...")
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        total_steps = 6

        status_text.text('Analyzing your video... (1/6)')
        process_video_to_csv(user_video_path, user_bowler_hand.lower(), user_csv_path)
        progress_bar.progress(1*100//total_steps)

        status_text.text('Analyzing the professional\'s video... (2/6)')
        process_video_to_csv(benchmark_video_path, benchmark_bowler_hand.lower(), benchmark_csv_path)
        progress_bar.progress(2*100//total_steps)

        status_text.text('Generating your annotated video... (3/6)')
        generate_annotated_video(user_video_path, user_csv_path, user_annotated_video_path, user_bowler_hand.lower())
        progress_bar.progress(3*100//total_steps)
        
        status_text.text('Generating benchmark annotated video... (4/6)')
        generate_annotated_video(benchmark_video_path, benchmark_csv_path, benchmark_annotated_video_path, benchmark_bowler_hand.lower())
        progress_bar.progress(4*100//total_steps)
        
        status_text.text('Generating comparison plot... (5/6)')
        generate_comparison_report(user_csv_path, benchmark_csv_path, report_image_path)
        progress_bar.progress(5*100//total_steps)
        
        status_text.text('Generating AI coaching report with Gemini... (6/6)')
        ai_report = generate_generative_ai_feedback(user_csv_path, benchmark_csv_path, api_key)
        progress_bar.progress(6*100//total_steps)
        
        status_text.success("Analysis Complete!")
        time.sleep(2)
        progress_bar.empty()
        status_text.empty()

        # --- Display Results ---
        st.header("Analysis Results")
        
        st.subheader("Visual Analysis")
        vid_col1, vid_col2 = st.columns(2)
        with vid_col1:
            st.markdown("#### Your Performance")
            st.video(user_annotated_video_path)
        with vid_col2:
            st.markdown("#### Professional Benchmark")
            st.video(benchmark_annotated_video_path)

        st.subheader("Performance Comparison Plot")
        st.image(report_image_path)
        
        st.subheader("AI-Generated Coaching Report")
        st.markdown(ai_report)

    else:
        st.error("Please upload both video files to proceed.")

