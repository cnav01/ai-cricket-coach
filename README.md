# AI-Powered Cricket Coach

An AI-powered coaching tool using Python and MediaPipe to analyze cricket biomechanics. This project is currently in active development and follows the 8-week work plan detailed below for the initial bowling module.

---

## Project Goal

This project aims to provide amateur and aspiring cricketers with access to affordable, data-driven coaching for all aspects of the game. The initial phase focuses on **bowling analysis**, with a clear roadmap to expand to **batting analysis** and other skills in the future.

By using a standard smartphone camera, the application will capture a player's action, quantify key biomechanical metrics, and provide actionable feedback to improve technique and prevent injuries.

---

## Project Roadmap & Status (Phase 1: Bowling MVP)

### Week 1: Project Scaffolding & Core Logic Prototype (Complete)
* **Work Items:**
    * Initialized a Git repository and a professional project structure.
    * Developed a standalone, reusable function to calculate angles from 3D points.
    * Wrote unit tests to verify the accuracy of the angle calculator.
* **Deliverable:** A tested Python script for angle calculation, forming the core engine of the application.

### Week 2: Live Data Ingestion Pipeline (In Progress)
* **Work Items:**
    * Build a script using OpenCV to read from a live webcam feed.
    * Integrate MediaPipe Pose to detect landmarks in the live video.
    * Extract and log the 3D coordinates of key joints (shoulder, elbow, wrist).
* **Deliverable:** A functional script that displays a live webcam feed and prints a continuous stream of 3D joint coordinates to the console.

### Week 3 & 4: Real-Time Analysis & Visualization (Pending)
* **Work Items:**
    * Merge the angle calculator (Week 1) with the live data pipeline (Week 2).
    * Calculate the elbow angle for every frame in real-time.
    * Display the calculated angle directly on the video feed using OpenCV.
    * Implement a visual alert system (e.g., red text) for risky angles.
* **Deliverable:** A live demo application that displays a user's precise elbow angle on-screen as they move.

### Week 5: Offline Analysis & Data Persistence (Pending)
* **Work Items:**
    * Modify the script to process pre-recorded video files.
    * Log the angle for each frame to a list.
    * Export the collected data (frame number, angle) to a CSV file using Pandas.
* **Deliverable:** A command-line tool that takes a video file as input and produces a CSV file containing its full angle analysis.

### Week 6: Reporting & The "Pro-Blueprint" (Pending)
* **Work Items:**
    * Source a high-quality video of a professional bowler.
    * Generate a "blueprint" CSV file by running the Week 5 tool on the pro's video.
    * Build a reporting script to generate a line graph from a data CSV.
* **Deliverable:** A benchmark data file and a script that automatically generates a performance graph.

### Week 7: The Comparative Engine (MVP Complete) (Pending)
* **Work Items:**
    * Enhance the reporting script to compare two CSV files (user vs. pro).
    * Overlay both data series on the same graph for direct comparison.
    * Add titles, labels, and a legend to finalize the visualization.
* **Deliverable:** The completed MVP, capable of producing a final report comparing a user's action to a professional's.

### Week 8: Refinement, Documentation & Future Planning (Pending)
* **Work Items:**
    * Refactor and clean up the codebase.
    * Fully document all scripts and functions.
    * Draft a plan for Phase 2, with **Batting Analysis** as the top priority, followed by more advanced bowling metrics (e.g., arm speed, head stability).
* **Deliverable:** A polished, well-documented project and a presentation outlining the next steps.

---

## Getting Started

Follow these steps to set up the development environment.

### Prerequisites

* Python 3.8+
* Git
* Anaconda (recommended)

### Installation & Setup

1.  **Clone the repository:**
    ```sh
    git clone [https://github.com/cnav01/ai-cricket-coach.git](https://github.com/cnav01/ai-cricket-coach.git)
    cd ai-cricket-coach
    ```

2.  **Create and activate the Conda environment:**
    ```sh
    conda create --name cricket-coach-env python=3.9
    conda activate cricket-coach-env
    ```

3.  **Install dependencies:**
    ```sh
    conda install numpy opencv-python mediapipe open3d
    ```

4.  **Verify the setup by running the tests:**
    ```sh
    python -m unittest tests.test_angle_calculator
    ```
    You should see an output indicating that all tests passed successfully.

