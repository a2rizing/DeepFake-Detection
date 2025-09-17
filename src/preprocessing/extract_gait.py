import cv2
import mediapipe as mp
import pandas as pd
import os
import glob

# Mediapipe setup
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)

def extract_gait(video_path, output_csv="data/gait_keypoints.csv"):
    cap = cv2.VideoCapture(video_path)
    
    # Get video filename without extension for identification
    video_name = os.path.splitext(os.path.basename(video_path))[0]

    all_keypoints = []
    frame_no = 0

    print(f"[INFO] Processing video: {video_path}")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_no += 1
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            keypoints = [video_name, frame_no]  # Include video name for identification
            for lm in landmarks:
                keypoints.extend([lm.x, lm.y, lm.z, lm.visibility])
            all_keypoints.append(keypoints)

    cap.release()

    if not all_keypoints:
        print(f"[WARNING] No pose landmarks detected in {video_path}")
        return pd.DataFrame()

    # Convert to DataFrame
    columns = ["video_name", "frame"]
    for i in range(33):  # MediaPipe pose has 33 landmarks
        columns.extend([f"x_{i}", f"y_{i}", f"z_{i}", f"vis_{i}"])

    df = pd.DataFrame(all_keypoints, columns=columns)
    print(f"[INFO] Extracted {len(df)} frames with pose landmarks from {video_name}")
    
    return df

def process_multiple_videos(data_folder="data", output_csv="data/gait_keypoints.csv"):
    """
    Process all MP4 files in the data folder and extract gait keypoints
    """
    # Find all MP4 files in the data folder
    video_pattern = os.path.join(data_folder, "*.mp4")
    video_files = glob.glob(video_pattern)
    
    if not video_files:
        print(f"[WARNING] No MP4 files found in {data_folder}")
        return
    
    print(f"[INFO] Found {len(video_files)} video files to process")
    
    all_dataframes = []
    
    for video_file in video_files:
        print(f"[INFO] Processing: {os.path.basename(video_file)}")
        df = extract_gait(video_file)
        if not df.empty:
            all_dataframes.append(df)
    
    if all_dataframes:
        # Combine all dataframes
        combined_df = pd.concat(all_dataframes, ignore_index=True)
        
        # Ensure data folder exists
        os.makedirs(os.path.dirname(output_csv), exist_ok=True)
        combined_df.to_csv(output_csv, index=False)
        
        print(f"[INFO] Combined gait keypoints from {len(all_dataframes)} videos saved to {output_csv}")
        print(f"[INFO] Total frames processed: {len(combined_df)}")
        
        # Print summary by video
        video_summary = combined_df.groupby('video_name').size()
        print("[INFO] Frames per video:")
        for video, count in video_summary.items():
            print(f"  {video}: {count} frames")
    else:
        print("[ERROR] No valid keypoints extracted from any video")

if __name__ == "__main__":
    # Process all MP4 files in the data folder
    process_multiple_videos("data", "data/gait_keypoints.csv")
