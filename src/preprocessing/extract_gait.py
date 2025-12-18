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

def process_multiple_videos(data_folder="data", output_csv="data/gait_keypoints.csv", recursive=False):
    """
    Process all MP4 files in the data folder and extract gait keypoints.
    
    Args:
        data_folder: Folder containing videos
        output_csv: Output CSV path
        recursive: If True, search recursively in subfolders
    """
    # Find all MP4 files
    if recursive:
        # Recursively find all MP4 files in nested folders
        video_files = []
        for root, dirs, files in os.walk(data_folder):
            for file in files:
                if file.lower().endswith('.mp4'):
                    video_files.append(os.path.join(root, file))
    else:
        video_pattern = os.path.join(data_folder, "*.mp4")
        video_files = glob.glob(video_pattern)
    
    if not video_files:
        print(f"[WARNING] No MP4 files found in {data_folder}")
        return
    
    print(f"[INFO] Found {len(video_files)} video files to process")
    
    all_dataframes = []
    
    from tqdm import tqdm
    for video_file in tqdm(video_files, desc="Extracting keypoints"):
        df = extract_gait(video_file)
        if not df.empty:
            all_dataframes.append(df)
    
    if all_dataframes:
        # Combine all dataframes
        combined_df = pd.concat(all_dataframes, ignore_index=True)
        
        # Ensure output folder exists
        os.makedirs(os.path.dirname(output_csv) if os.path.dirname(output_csv) else ".", exist_ok=True)
        combined_df.to_csv(output_csv, index=False)
        
        print(f"\n[INFO] Combined gait keypoints from {len(all_dataframes)} videos saved to {output_csv}")
        print(f"[INFO] Total frames processed: {len(combined_df)}")
        
        # Print summary by person (extracted from video name)
        combined_df['person'] = combined_df['video_name'].apply(lambda x: x.split('_')[0])
        person_summary = combined_df.groupby('person')['video_name'].nunique()
        print("\n[INFO] Videos per person:")
        for person, count in person_summary.items():
            print(f"  {person}: {count} videos")
    else:
        print("[ERROR] No valid keypoints extracted from any video")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Extract gait keypoints from videos")
    parser.add_argument("--input", type=str, default="data/videos",
                        help="Input folder with videos (default: data/videos)")
    parser.add_argument("--output", type=str, default="data/gait_keypoints.csv",
                        help="Output CSV path (default: data/gait_keypoints.csv)")
    parser.add_argument("--recursive", "-r", action="store_true",
                        help="Recursively search subfolders for videos")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("GAIT KEYPOINT EXTRACTION")
    print("=" * 60)
    print(f"Input folder: {args.input}")
    print(f"Output CSV: {args.output}")
    print(f"Recursive: {args.recursive}")
    print("=" * 60)
    
    process_multiple_videos(args.input, args.output, recursive=args.recursive)

