import streamlit as st
import os
import time
from scripts.analysis_pipeline_MAIN import process_video_to_csv, generate_comparison_report, generate_ai_feedback

# --- Page Configuration ---
st.set_page_config(
    page_title="Spinvic AI Cricket Coach",
    page_icon="🏏",
    layout="wide"
)

# --- App Title ---
st.title("Spinvic AI Cricket Coach")
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
    if user_video_file is not None and benchmark_video_file is not None:
        
        # Define file paths
        output_dir = "output"
        os.makedirs(output_dir, exist_ok=True)
        
        user_video_path = os.path.join(output_dir, "user_video.mp4")
        benchmark_video_path = os.path.join(output_dir, "benchmark_video.mp4")
        
        user_csv_path = os.path.join(output_dir, "user_analysis.csv")
        benchmark_csv_path = os.path.join(output_dir, "benchmark_analysis.csv")
        
        report_image_path = os.path.join(output_dir, "comparison_report.png")

        # Save uploaded files
        with open(user_video_path, "wb") as f:
            f.write(user_video_file.getbuffer())
        with open(benchmark_video_path, "wb") as f:
            f.write(benchmark_video_file.getbuffer())

        # --- Run Analysis Pipeline ---
        with st.spinner('Analyzing your video... This may take a moment.'):
            process_video_to_csv(user_video_path, user_bowler_hand, user_csv_path)
        st.success("Your video analysis complete!")

        with st.spinner('Analyzing the professional\'s video...'):
            process_video_to_csv(benchmark_video_path, benchmark_bowler_hand, benchmark_csv_path)
        st.success("Benchmark video analysis complete!")
        
        with st.spinner('Generating comparison report...'):
            generate_comparison_report(user_csv_path, benchmark_csv_path, report_image_path)
        st.success("Comparison report generated!")
        
        with st.spinner('Generating AI coaching feedback...'):
            ai_report = generate_ai_feedback(user_csv_path)
        st.success("AI coaching report ready!")

        # --- Display Results ---
        st.header("Analysis Results")
        
        st.subheader("Performance Comparison Plot")
        st.image(report_image_path)
        
        st.subheader("AI-Generated Coaching Report")
        st.markdown(ai_report)

    else:
        st.error("Please upload both video files to proceed.")
