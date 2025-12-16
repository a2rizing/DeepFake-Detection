"""
Regenerate gait_keypoints.csv from the new dataset
"""

import os
import cv2
import numpy as np
import pandas as pd
import mediapipe as mp
from tqdm import tqdm

# Initialize MediaPipe
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

def extract_keypoints_from_video(video_path, video_name, max_frames=64):
    """Extract MediaPipe keypoints from video"""
    cap = cv2.VideoCapture(video_path)
    keypoints_data = []
    
    frame_num = 0
    frames_processed = 0
    
    # Get total frames and frame skip rate
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    skip_rate = max(1, total_frames // max_frames)  # Process up to max_frames
    
    while cap.isOpened() and frames_processed < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_num += 1
        
        # Skip frames to speed up processing
        if frame_num % skip_rate != 0:
            continue
        
        frames_processed += 1
        
        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb_frame)
            
            if results.pose_landmarks:
                row_data = {'video_name': video_name, 'frame': frames_processed}
                
                for idx, landmark in enumerate(results.pose_landmarks.landmark):
                    row_data[f'x_{idx}'] = landmark.x
                    row_data[f'y_{idx}'] = landmark.y
                    row_data[f'z_{idx}'] = landmark.z
                    row_data[f'vis_{idx}'] = landmark.visibility
                
                keypoints_data.append(row_data)
        except Exception as e:
            print(f"  Error processing frame {frame_num}: {e}")
            continue
    
    cap.release()
    return keypoints_data

def main():
    print("="*80)
    print("Regenerating gait_keypoints.csv from new dataset")
    print("="*80)
    
    data_dir = 'data'
    all_keypoints = []
    
    # Get all video files
    video_files = [f for f in os.listdir(data_dir) if f.endswith('.mp4')]
    
    print(f"\nFound {len(video_files)} videos to process\n")
    
    for video_file in tqdm(video_files, desc="Processing videos"):
        video_path = os.path.join(data_dir, video_file)
        video_name = os.path.splitext(video_file)[0]
        
        keypoints = extract_keypoints_from_video(video_path, video_name)
        all_keypoints.extend(keypoints)
    
    # Create DataFrame
    df = pd.DataFrame(all_keypoints)
    
    # Save to CSV
    output_path = 'data/gait_keypoints.csv'
    df.to_csv(output_path, index=False)
    
    print(f"\n{'='*80}")
    print(f"✅ Saved {len(df)} keypoint frames to {output_path}")
    print(f"   Videos processed: {len(video_files)}")
    print(f"   Total frames: {len(df)}")
    print(f"{'='*80}\n")

if __name__ == "__main__":
    main()
