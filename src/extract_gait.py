import cv2
import mediapipe as mp
import pandas as pd
import os

# Mediapipe setup
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)

def extract_gait(video_path, output_csv="data/gait_keypoints.csv"):
    cap = cv2.VideoCapture(video_path)

    all_keypoints = []
    frame_no = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_no += 1
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            keypoints = [frame_no]
            for lm in landmarks:
                keypoints.extend([lm.x, lm.y, lm.z, lm.visibility])
            all_keypoints.append(keypoints)

    cap.release()

    # Convert to DataFrame
    columns = ["frame"]
    for i in range(len(results.pose_landmarks.landmark)):
        columns.extend([f"x_{i}", f"y_{i}", f"z_{i}", f"vis_{i}"])

    df = pd.DataFrame(all_keypoints, columns=columns)

    # Ensure data folder exists
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"[INFO] Gait keypoints saved to {output_csv}")

if __name__ == "__main__":
    video_file = "data/walking1.mp4"  # <-- put your walking video here
    extract_gait(video_file, "data/gait_keypoints.csv")
