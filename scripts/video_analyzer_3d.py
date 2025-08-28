import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import open3d as o3d
import sys
import os

# This helps Python find the 'scripts' folder
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import our tested angle calculator
from scripts.angle_calculator import calculate_angle

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=0.5
)

# Initialize Open3D Visualizer
vis_3d = o3d.visualization.Visualizer()
vis_3d.create_window(window_name="3D Visual", width=900, height=480)

point_cloud = o3d.geometry.PointCloud()
line_set = o3d.geometry.LineSet()
vis_3d.add_geometry(point_cloud)
vis_3d.add_geometry(line_set)

ground_plane = o3d.geometry.TriangleMesh.create_box(width=4.0, height=0.02, depth=4.0)
ground_plane.translate((-2.0, -1.0, -2.0))
ground_plane.paint_uniform_color([0.7, 0.7, 0.7])
vis_3d.add_geometry(ground_plane)

connections = mp_pose.POSE_CONNECTIONS

# Video File Path
video_path = "videos/bowling.mp4" 
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(f"Error: Could not open video file: {video_path}")
    exit()

analysis_data = []
frame_number = 0

while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Video ended")
        break # Exit loop if video ends, don't loop
        #cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        #continue

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)

    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            image, 
            results.pose_landmarks, 
            connections,
            landmark_drawing_spec=mp_drawing.DrawingSpec(color=(255, 0, 255), thickness=2, circle_radius=2), # Magenta joints
            connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=2, circle_radius=2)  # Cyan bones
        )

    if results.pose_world_landmarks:
        landmarks_3d = results.pose_world_landmarks.landmark

        # --- Angle Calculation using our imported function ---
        r_shoulder = [landmarks_3d[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks_3d[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y, landmarks_3d[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].z]
        r_elbow = [landmarks_3d[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x, landmarks_3d[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y, landmarks_3d[mp_pose.PoseLandmark.RIGHT_ELBOW.value].z]
        r_wrist = [landmarks_3d[mp_pose.PoseLandmark.RIGHT_WRIST.value].x, landmarks_3d[mp_pose.PoseLandmark.RIGHT_WRIST.value].y, landmarks_3d[mp_pose.PoseLandmark.RIGHT_WRIST.value].z]

        l_shoulder = [landmarks_3d[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks_3d[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y, landmarks_3d[mp_pose.PoseLandmark.LEFT_SHOULDER.value].z]
        l_elbow = [landmarks_3d[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks_3d[mp_pose.PoseLandmark.LEFT_ELBOW.value].y, landmarks_3d[mp_pose.PoseLandmark.LEFT_ELBOW.value].z]
        l_wrist = [landmarks_3d[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks_3d[mp_pose.PoseLandmark.LEFT_WRIST.value].y, landmarks_3d[mp_pose.PoseLandmark.LEFT_WRIST.value].z]

        # We now call our tested, reusable function
        r_elbow_angle = calculate_angle(r_shoulder, r_elbow, r_wrist)
        l_elbow_angle = calculate_angle(l_shoulder, l_elbow, l_wrist)
        
        analysis_data.append({
            "frame": frame_number,
            "right_elbow_angle": r_elbow_angle,
            "left_elbow_angle": l_elbow_angle
        })


        # Display the angles on the 2D image
        landmarks_2d = results.pose_landmarks.landmark
        image_h, image_w, _ = image.shape
        right_elbow_2d = landmarks_2d[mp_pose.PoseLandmark.RIGHT_ELBOW]
        left_elbow_2d = landmarks_2d[mp_pose.PoseLandmark.LEFT_ELBOW]
        coordinates_elbow_r = (int(right_elbow_2d.x * image_w), int(right_elbow_2d.y * image_h))
        coordinates_elbow_l = (int(left_elbow_2d.x * image_w), int(left_elbow_2d.y * image_h))

        cv2.putText(image, str(int(r_elbow_angle)), coordinates_elbow_r, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(image, str(int(l_elbow_angle)), coordinates_elbow_l, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # --- 3D Visualization ---
        points_3d = np.array([[lm.x, -lm.y, -lm.z] for lm in landmarks_3d])
        visibilities = np.array([lm.visibility for lm in landmarks_3d])

        visible_points = points_3d[visibilities > 0.1]
        point_cloud.points = o3d.utility.Vector3dVector(visible_points)
        point_cloud.paint_uniform_color([1.0, 0.0, 1.0])

        lines = [[a, b] for a, b in connections if visibilities[a] > 0.1 and visibilities[b] > 0.1]
        
        line_set.points = o3d.utility.Vector3dVector(points_3d)
        line_set.lines = o3d.utility.Vector2iVector(lines)
        line_set.paint_uniform_color([0.0, 1.0, 1.0])

        vis_3d.update_geometry(point_cloud)
        vis_3d.update_geometry(line_set)

    resized_img = cv2.resize(image, (1280, 720))
    cv2.imshow('2D Pose Overlay', resized_img)

    if not vis_3d.poll_events():
        break
    vis_3d.update_renderer()
    frame_number += 1

    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
vis_3d.destroy_window()
pose.close()

if analysis_data:
    df = pd.DataFrame(analysis_data)
    df.to_csv("output/angle_analysis.csv", index=False)
    print("Angle analysis saved to output/angle_analysis.csv")
else:
    print("No pose data detected; no analysis saved.")
