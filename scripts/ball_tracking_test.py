import cv2
import mediapipe as mp
import numpy as np
import collections
# import matplotlib.pyplot as plt # Uncomment for optional plotting

# --- Configuration ---
video_path = 'videos/pro_bowler.mp4'  # Make sure this path is correct to your test video

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(f"Error: Could not open video file: {video_path}")
    exit()

# Store metrics and their differences
# We'll store the count of 'white' pixels in the ROI for each frame
white_pixel_counts = [] # Store (frame_idx, count)

# --- Parameters for Anomaly Detection (White Pixel Count Method) ---
ROI_PADDING = 70 # Half the side length of the square ROI (140x140 pixels). Smaller for tighter focus.

# Define the HSV range for a WHITE cricket ball
# These values are a good starting point but WILL likely need tuning for your specific video
lower_ball_hsv = np.array([0, 0, 160])   # Low Saturation values (close to grayscale)
upper_ball_hsv = np.array([180, 50, 255]) # High Value (brightness) and low Saturation range for white

frame_idx = 0
release_frame_idx = -1 # To store the detected release frame

print("Starting video processing for white pixel count anomaly (First Pass)...")

# --- First Pass: Collect White Pixel Counts in ROI ---
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        if frame_idx == 0:
            print("Could not read any frames from the video. Check path or file integrity.")
        break # Video ended or error

    # Store original frame for later display (if using frame_buffer for pass 2)
    # If video is very long, consider not storing all frames to save memory
    # For now, let's keep it for easy second pass.
    # frame_buffer.append(frame.copy()) 

    # Find the hand with MediaPipe to define a search area (ROI)
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)

    current_white_count = 0

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark 
        image_h, image_w, _ = frame.shape
        
        # Using RIGHT_WRIST as the center of our search area
        wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST] 
        
        roi_center_x = int(wrist.x * image_w)
        roi_center_y = int(wrist.y * image_h)
        
        x1 = max(0, roi_center_x - ROI_PADDING)
        y1 = max(0, roi_center_y - ROI_PADDING)
        x2 = min(image_w, roi_center_x + ROI_PADDING)
        y2 = min(image_h, image_w, roi_center_y + ROI_PADDING) # Fixed potential error: min(image_h, image_w, ...) to min(image_h, ...)

        if y2 > y1 and x2 > x1: # Ensure ROI is valid
            roi = frame[y1:y2, x1:x2].copy() # Ensure we work on a copy
            
            hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            
            # Create a mask for white pixels
            white_mask = cv2.inRange(hsv_roi, lower_ball_hsv, upper_ball_hsv)

            # --- Optional: Apply morphological operations to clean the white mask ---
            # This helps make the 'white' count more reliable by removing noise
            kernel = np.ones((3,3),np.uint8) # Small kernel for finer cleaning
            white_mask = cv2.erode(white_mask, kernel, iterations=1)
            white_mask = cv2.dilate(white_mask, kernel, iterations=1)

            current_white_count = cv2.countNonZero(white_mask)
            
            # For visualization/debugging:
            # cv2.imshow('White Mask in ROI', white_mask)
            # cv2.imshow('Original ROI', roi)

    white_pixel_counts.append((frame_idx, current_white_count))
    frame_idx += 1

cap.release() # Release video for the first pass

# --- Analyze White Pixel Counts to Find Release Frame ---
if white_pixel_counts:
    # We are looking for a significant DROP in white pixels as the ball leaves.
    # We'll calculate the *difference* between consecutive counts.
    # A large negative difference (a sudden drop) indicates release.
    
    # Store (frame_idx, drop_magnitude)
    drops = [] 
    
    for i in range(1, len(white_pixel_counts)):
        current_frame_idx, current_count = white_pixel_counts[i]
        prev_frame_idx, prev_count = white_pixel_counts[i-1]
        
        # Calculate the drop: prev_count - current_count. 
        # A positive value here means a drop in white pixels.
        drop_magnitude = prev_count - current_count
        
        drops.append((current_frame_idx, drop_magnitude))
    
    max_drop_idx = -1
    max_drop_magnitude = -1.0 # Looking for the largest positive drop

    # Find the frame with the maximum drop in white pixels
    for idx, drop_val in drops:
        if drop_val > max_drop_magnitude:
            max_drop_magnitude = drop_val
            max_drop_idx = idx
    
    release_frame_idx = max_drop_idx
    print(f"Detected potential ball release at frame: {release_frame_idx} with a white pixel drop of: {max_drop_magnitude:.2f}")

    # Optional: Plotting white pixel counts and drops for debugging/visualization
    # frames_plot = [d[0] for d in white_pixel_counts]
    # counts_plot = [d[1] for d in white_pixel_counts]
    #
    # plt.figure(figsize=(15, 6))
    # plt.subplot(1, 2, 1)
    # plt.plot(frames_plot, counts_plot, label='White Pixel Count in ROI')
    # plt.axvline(x=release_frame_idx, color='r', linestyle='--', label=f'Detected Release: {release_frame_idx}')
    # plt.title('White Pixel Count Over Time')
    # plt.xlabel('Frame Index')
    # plt.ylabel('Count of White Pixels')
    # plt.legend()
    # plt.grid(True)
    #
    # frames_drop_plot = [d[0] for d in drops]
    # drop_vals_plot = [d[1] for d in drops]
    # plt.subplot(1, 2, 2)
    # plt.plot(frames_drop_plot, drop_vals_plot, label='Drop in White Pixels (Prev - Current)')
    # plt.axvline(x=release_frame_idx, color='r', linestyle='--', label=f'Detected Release: {release_frame_idx}')
    # plt.title('Frame-to-Frame Drop in White Pixels')
    # plt.xlabel('Frame Index')
    # plt.ylabel('Drop Magnitude')
    # plt.legend()
    # plt.grid(True)
    # plt.tight_layout()
    # plt.show()

else:
    print("No white pixel counts could be computed. Check video and MediaPipe detection stability.")

print("\nStarting video display (Second Pass)...")

# --- Second Pass: Display Video with Release Frame Highlighted ---
cap = cv2.VideoCapture(video_path) # Reopen video for display

current_frame_for_display = 0
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    if current_frame_for_display == release_frame_idx:
        cv2.putText(frame, "BALL RELEASE!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4, cv2.LINE_AA)
        cv2.rectangle(frame, (0, 0), (frame.shape[1], frame.shape[0]), (0, 0, 255), 10) # Red border
    
    cv2.putText(frame, f"Frame: {current_frame_for_display}", (frame.shape[1] - 200, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.imshow('Ball Release Anomaly Detection (White Pixel Count)', frame)
    if cv2.waitKey(25) & 0xFF == ord('q'): # Play at ~40 FPS
        break

    current_frame_for_display += 1

cap.release()
cv2.destroyAllWindows()