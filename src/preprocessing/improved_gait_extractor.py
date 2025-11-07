"""
Improved Gait Feature Extractor with MediaPipe
Extracts comprehensive gait features from videos
"""

import cv2
import mediapipe as mp
import numpy as np
import json
import os
from tqdm import tqdm

class ImprovedGaitExtractor:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=2,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
    
    def extract_keypoints_from_video(self, video_path, max_frames=300):
        """Extract pose keypoints from video"""
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"❌ Could not open video: {video_path}")
            return None
        
        keypoints_sequence = []
        frame_count = 0
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        pbar = tqdm(total=min(total_frames, max_frames), 
                   desc=f"Processing {os.path.basename(video_path)}", 
                   leave=False)
        
        while frame_count < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convert to RGB
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process with MediaPipe
            results = self.pose.process(image_rgb)
            
            if results.pose_landmarks:
                # Extract all 33 landmarks
                landmarks = []
                for landmark in results.pose_landmarks.landmark:
                    landmarks.append({
                        'x': landmark.x,
                        'y': landmark.y,
                        'z': landmark.z,
                        'visibility': landmark.visibility
                    })
                
                keypoints_sequence.append({
                    'frame': frame_count,
                    'landmarks': landmarks,
                    'timestamp': cap.get(cv2.CAP_PROP_POS_MSEC)
                })
            
            frame_count += 1
            pbar.update(1)
        
        pbar.close()
        cap.release()
        
        if len(keypoints_sequence) == 0:
            print(f"⚠️  No pose detected in video: {video_path}")
            return None
        
        return keypoints_sequence
    
    def calculate_angles(self, p1, p2, p3):
        """Calculate angle between three points"""
        v1 = np.array([p1['x'] - p2['x'], p1['y'] - p2['y']])
        v2 = np.array([p3['x'] - p2['x'], p3['y'] - p2['y']])
        
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
        angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
        return np.degrees(angle)
    
    def calculate_distance(self, p1, p2):
        """Calculate Euclidean distance between two points"""
        return np.sqrt((p1['x'] - p2['x'])**2 + (p1['y'] - p2['y'])**2 + (p1['z'] - p2['z'])**2)
    
    def extract_gait_features(self, keypoints_sequence):
        """Extract comprehensive gait features"""
        
        if not keypoints_sequence or len(keypoints_sequence) < 2:
            return None
        
        features = {
            'temporal': [],
            'spatial': [],
            'angles': []
        }
        
        for i in range(1, len(keypoints_sequence)):
            curr = keypoints_sequence[i]['landmarks']
            prev = keypoints_sequence[i-1]['landmarks']
            
            # Temporal features (frame-to-frame changes)
            temporal_feat = []
            
            # 1. Stride length (ankle distance)
            left_ankle = curr[27]
            right_ankle = curr[28]
            stride = self.calculate_distance(left_ankle, right_ankle)
            temporal_feat.append(stride)
            
            # 2. Step velocity (ankle movement speed)
            left_ankle_prev = prev[27]
            left_ankle_velocity = self.calculate_distance(left_ankle, left_ankle_prev)
            right_ankle_velocity = self.calculate_distance(right_ankle, prev[28])
            temporal_feat.extend([left_ankle_velocity, right_ankle_velocity])
            
            # 3. Hip movement
            left_hip = curr[23]
            right_hip = curr[24]
            hip_sway = self.calculate_distance(left_hip, prev[23])
            temporal_feat.append(hip_sway)
            
            # 4. Arm swing
            left_wrist = curr[15]
            right_wrist = curr[16]
            left_arm_swing = self.calculate_distance(left_wrist, prev[15])
            right_arm_swing = self.calculate_distance(right_wrist, prev[16])
            temporal_feat.extend([left_arm_swing, right_arm_swing])
            
            # 5. Body center movement (average of shoulders and hips)
            body_center_x = np.mean([curr[11]['x'], curr[12]['x'], curr[23]['x'], curr[24]['x']])
            body_center_y = np.mean([curr[11]['y'], curr[12]['y'], curr[23]['y'], curr[24]['y']])
            prev_center_x = np.mean([prev[11]['x'], prev[12]['x'], prev[23]['x'], prev[24]['x']])
            prev_center_y = np.mean([prev[11]['y'], prev[12]['y'], prev[23]['y'], prev[24]['y']])
            body_velocity = np.sqrt((body_center_x - prev_center_x)**2 + (body_center_y - prev_center_y)**2)
            temporal_feat.append(body_velocity)
            
            features['temporal'].append(temporal_feat)
            
            # Spatial features (pose geometry)
            spatial_feat = []
            
            # 1. Body posture (shoulder-hip distance)
            shoulder_hip_dist = self.calculate_distance(curr[11], curr[23])
            spatial_feat.append(shoulder_hip_dist)
            
            # 2. Leg length
            left_leg_length = self.calculate_distance(curr[23], curr[25]) + self.calculate_distance(curr[25], curr[27])
            right_leg_length = self.calculate_distance(curr[24], curr[26]) + self.calculate_distance(curr[26], curr[28])
            spatial_feat.extend([left_leg_length, right_leg_length])
            
            features['spatial'].append(spatial_feat)
            
            # Angular features (joint angles)
            angular_feat = []
            
            # 1. Left knee angle
            left_knee_angle = self.calculate_angles(curr[23], curr[25], curr[27])
            angular_feat.append(left_knee_angle)
            
            # 2. Right knee angle
            right_knee_angle = self.calculate_angles(curr[24], curr[26], curr[28])
            angular_feat.append(right_knee_angle)
            
            # 3. Left hip angle
            left_hip_angle = self.calculate_angles(curr[11], curr[23], curr[25])
            angular_feat.append(left_hip_angle)
            
            # 4. Right hip angle
            right_hip_angle = self.calculate_angles(curr[12], curr[24], curr[26])
            angular_feat.append(right_hip_angle)
            
            features['angles'].append(angular_feat)
        
        # Convert to numpy arrays
        features['temporal'] = np.array(features['temporal'])
        features['spatial'] = np.array(features['spatial'])
        features['angles'] = np.array(features['angles'])
        
        return features
    
    def process_video(self, video_path, output_dir=None):
        """Process a single video and extract all features"""
        
        # Extract keypoints
        keypoints = self.extract_keypoints_from_video(video_path)
        
        if keypoints is None:
            return None
        
        # Extract gait features
        gait_features = self.extract_gait_features(keypoints)
        
        if gait_features is None:
            return None
        
        # Save if output directory provided
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            video_name = os.path.splitext(os.path.basename(video_path))[0]
            
            # Save keypoints
            keypoint_path = os.path.join(output_dir, f"{video_name}_keypoints.json")
            with open(keypoint_path, 'w') as f:
                json.dump(keypoints, f)
            
            # Save features (convert numpy to lists for JSON)
            feature_path = os.path.join(output_dir, f"{video_name}_features.npz")
            np.savez(feature_path, 
                    temporal=gait_features['temporal'],
                    spatial=gait_features['spatial'],
                    angles=gait_features['angles'])
        
        return gait_features
    
    def process_dataset(self, real_dir, fake_dir, output_dir):
        """Process entire dataset of real and fake videos"""
        
        print("🎬 Processing Video Dataset")
        print("=" * 60)
        
        os.makedirs(os.path.join(output_dir, 'real'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'fake'), exist_ok=True)
        
        # Process real videos
        print("\n📹 Processing REAL videos...")
        real_videos = [f for f in os.listdir(real_dir) if f.endswith(('.mp4', '.avi', '.mov'))]
        real_features = []
        
        for video_file in tqdm(real_videos, desc="Real videos"):
            video_path = os.path.join(real_dir, video_file)
            features = self.process_video(video_path, os.path.join(output_dir, 'real'))
            if features is not None:
                real_features.append(features)
        
        # Process fake videos
        print("\n🎭 Processing FAKE videos...")
        fake_videos = [f for f in os.listdir(fake_dir) if f.endswith(('.mp4', '.avi', '.mov'))]
        fake_features = []
        
        for video_file in tqdm(fake_videos, desc="Fake videos"):
            video_path = os.path.join(fake_dir, video_file)
            features = self.process_video(video_path, os.path.join(output_dir, 'fake'))
            if features is not None:
                fake_features.append(features)
        
        print("\n" + "=" * 60)
        print(f"✅ Processed {len(real_features)} real videos")
        print(f"✅ Processed {len(fake_features)} fake videos")
        print(f"✅ Features saved to: {output_dir}")
        print("=" * 60)
        
        return real_features, fake_features


if __name__ == "__main__":
    # Example usage
    extractor = ImprovedGaitExtractor()
    
    real_dir = "data/raw/real"
    fake_dir = "data/raw/fake"
    output_dir = "data/processed"
    
    if os.path.exists(real_dir) and os.path.exists(fake_dir):
        extractor.process_dataset(real_dir, fake_dir, output_dir)
    else:
        print("❌ Please create data/raw/real and data/raw/fake directories")
        print("   Run: python download_sample_data.py for instructions")
