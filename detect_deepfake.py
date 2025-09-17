#!/usr/bin/env python3
"""
Deepfake Detection Pipeline
Analyzes MP4 videos using gait analysis to detect deepfakes
"""

import os
import sys
import json
import glob
import argparse
import numpy as np
import pandas as pd
import cv2
import joblib
import mediapipe as mp
from sklearn.preprocessing import StandardScaler

class DeepfakeDetectionPipeline:
    def __init__(self, model_dir="models"):
        """Initialize the deepfake detection pipeline"""
        self.model_dir = model_dir
        self.model = None
        self.scaler = None
        self.metadata = None
        
        # Initialize MediaPipe
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Load the best model
        if not self.load_best_model():
            print("Warning: No trained model found. Please train a model first.")
    
    def load_best_model(self):
        """Load the best performing model"""
        # First try to find the specific best_model_metadata.json
        best_metadata_path = os.path.join(self.model_dir, "best_model_metadata.json")
        
        if not os.path.exists(best_metadata_path):
            # If not found, look for any metadata files and use the latest one
            metadata_files = glob.glob(os.path.join(self.model_dir, "metadata_*.json"))
            
            if not metadata_files:
                print("No trained models found.")
                return False
            
            # Use the most recent metadata file
            best_metadata_path = max(metadata_files, key=os.path.getmtime)
            print(f"Using latest model: {os.path.basename(best_metadata_path)}")
        
        # Load the best model
        try:
            with open(best_metadata_path, 'r') as f:
                self.metadata = json.load(f)
            
            print(f"Found best model: {self.metadata['model_type']}")
            print(f"   Accuracy: {self.metadata.get('test_accuracy', self.metadata.get('accuracy', 'N/A'))}")
            print(f"   Timestamp: {self.metadata['timestamp']}")
            
            # Load model file
            timestamp = self.metadata['timestamp']
            model_files = glob.glob(os.path.join(self.model_dir, f"*{timestamp}*.joblib"))
            scaler_files = glob.glob(os.path.join(self.model_dir, f"scaler*{timestamp}*.joblib"))
            
            # Load model
            model_file = None
            for mf in model_files:
                if 'scaler' not in os.path.basename(mf):
                    model_file = mf
                    break
            
            if model_file:
                self.model = joblib.load(model_file)
                print(f"   Model loaded: {os.path.basename(model_file)}")
            else:
                print("Model file not found!")
                return False
            
            # Load scaler
            if scaler_files:
                self.scaler = joblib.load(scaler_files[0])
                print(f"   Scaler loaded: {os.path.basename(scaler_files[0])}")
            else:
                print("Scaler not found, will use default normalization")
                self.scaler = StandardScaler()
            
            return True
            
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def extract_gait_from_video(self, video_path):
        """Extract gait keypoints from a single video"""
        print(f"Extracting gait from: {os.path.basename(video_path)}")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Could not open video: {video_path}")
            return None
        
        all_keypoints = []
        frame_no = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_no += 1
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose.process(rgb)
            
            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                keypoints = [frame_no]
                for lm in landmarks:
                    keypoints.extend([lm.x, lm.y, lm.z, lm.visibility])
                all_keypoints.append(keypoints)
        
        cap.release()
        
        if not all_keypoints:
            print(f"No pose landmarks detected in {video_path}")
            return None
        
        # Convert to DataFrame
        columns = ["frame"]
        for i in range(33):  # MediaPipe pose has 33 landmarks
            columns.extend([f"x_{i}", f"y_{i}", f"z_{i}", f"vis_{i}"])
        
        df = pd.DataFrame(all_keypoints, columns=columns)
        print(f"   Extracted {len(df)} frames with pose landmarks")
        
        return df
    
    def preprocess_gait_data(self, df):
        """Convert gait DataFrame to model input format"""
        if df is None or len(df) == 0:
            return None
        
        # Extract x,y coordinates
        n_landmarks = 33
        xs = [f"x_{i}" for i in range(n_landmarks)]
        ys = [f"y_{i}" for i in range(n_landmarks)]
        
        x_arr = df[xs].values  # (T, L)
        y_arr = df[ys].values  # (T, L)
        T = x_arr.shape[0]
        
        # Create (T, L, 2) array
        gait_sequence = np.zeros((T, n_landmarks, 2), dtype=np.float32)
        gait_sequence[:, :, 0] = x_arr
        gait_sequence[:, :, 1] = y_arr
        
        # Apply the same preprocessing as in training
        normalized_sequence = self.center_and_scale_sequence(gait_sequence)
        
        # Convert to feature matrix
        feature_matrix = self.sequence_to_feature_matrix(normalized_sequence)
        
        return feature_matrix
    
    def center_and_scale_sequence(self, kps_xy):
        """Center and scale gait sequence (same as in preprocessing)"""
        T, L, _ = kps_xy.shape
        
        # Use MediaPipe landmark indices
        left_hip_idx, right_hip_idx = 23, 24
        left_sh_idx, right_sh_idx = 11, 12
        
        # Calculate mid-hip and mid-shoulder points
        hips = kps_xy[:, [left_hip_idx, right_hip_idx], :]
        mid_hip = np.nanmean(hips, axis=1)
        shoulders = kps_xy[:, [left_sh_idx, right_sh_idx], :]
        mid_sh = np.nanmean(shoulders, axis=1)
        
        # Center each frame
        centered = kps_xy - mid_hip[:, None, :]
        
        # Scale by torso length
        torso_len = np.linalg.norm(mid_sh - mid_hip, axis=1)
        torso_mean = np.nanmean(torso_len)
        if np.isnan(torso_mean) or torso_mean < 1e-8:
            torso_mean = 1.0
        
        normalized = centered / (torso_mean + 1e-8)
        return normalized
    
    def compute_angles_sequence(self, kps_xy):
        """Compute joint angles (same as in preprocessing)"""
        T, L, _ = kps_xy.shape
        
        # Define angle triplets (same as in preprocessing)
        angle_triplets = [
            (23, 25, 27),  # Left: hip-knee-ankle
            (24, 26, 28),  # Right: hip-knee-ankle
            (11, 13, 15),  # Left elbow: shoulder-elbow-wrist
            (12, 14, 16),  # Right elbow: shoulder-elbow-wrist
        ]
        
        num_angles = len(angle_triplets)
        angles = np.zeros((T, num_angles), dtype=np.float32)
        
        for t in range(T):
            for i, (a, b, c) in enumerate(angle_triplets):
                A = kps_xy[t, a, :]
                B = kps_xy[t, b, :]
                C = kps_xy[t, c, :]
                angles[t, i] = self.angle_at_point(A, B, C)
        
        return angles
    
    def angle_at_point(self, a, b, c):
        """Calculate angle at point b formed by points a-b-c"""
        ba = a - b
        bc = c - b
        
        na = np.linalg.norm(ba)
        nb = np.linalg.norm(bc)
        denom = (na * nb) + 1e-8
        
        cosang = np.dot(ba, bc) / denom
        cosang = np.clip(cosang, -1.0, 1.0)
        return np.arccos(cosang)
    
    def resample_sequence(self, seq, target_len=64):
        """Resample sequence to target length"""
        T, D = seq.shape
        if T == target_len:
            return seq.copy()
        
        old_idx = np.linspace(0, 1, T)
        new_idx = np.linspace(0, 1, target_len)
        resampled = np.zeros((target_len, D), dtype=seq.dtype)
        
        for d in range(D):
            resampled[:, d] = np.interp(new_idx, old_idx, seq[:, d])
        
        return resampled
    
    def sequence_to_feature_matrix(self, kps_norm, target_frames=64):
        """Convert normalized sequence to feature matrix"""
        T, L, _ = kps_norm.shape
        
        # Flatten x,y coordinates
        flat = kps_norm.reshape((T, L * 2))
        
        # Add joint angles
        angles = self.compute_angles_sequence(kps_norm)
        features = np.concatenate([flat, angles], axis=1)
        
        # Resample to target frames
        resampled = self.resample_sequence(features, target_frames)
        
        # Flatten for traditional ML models (shape: target_frames * features)
        flattened = resampled.flatten()
        
        return flattened.reshape(1, -1)  # Add batch dimension
    
    def predict_deepfake(self, video_path, threshold=0.5):
        """Predict if a video contains deepfake based on gait analysis"""
        print(f"\nAnalyzing video: {video_path}")
        print("-" * 50)
        
        # Check if file exists
        if not os.path.exists(video_path):
            print(f"Video file not found: {video_path}")
            return None
        
        # Check if model is loaded
        if self.model is None:
            print("No model loaded!")
            return None
        
        # Extract gait data
        gait_df = self.extract_gait_from_video(video_path)
        if gait_df is None:
            return {
                'video_path': video_path,
                'prediction': 'Unknown',
                'confidence': 0.0,
                'error': 'No pose landmarks detected'
            }
        
        # Preprocess
        features = self.preprocess_gait_data(gait_df)
        if features is None:
            return {
                'video_path': video_path,
                'prediction': 'Unknown',
                'confidence': 0.0,
                'error': 'Failed to preprocess gait data'
            }
        
        # Normalize features if scaler is available
        if self.scaler:
            try:
                features = self.scaler.transform(features)
            except Exception as e:
                print(f"Warning: Could not apply scaler: {e}")
        
        # Make prediction
        try:
            if hasattr(self.model, 'predict_proba'):
                proba = self.model.predict_proba(features)[0]
                deepfake_probability = proba[1]  # Probability of class 1 (deepfake)
            else:
                # For models without predict_proba
                prediction = self.model.predict(features)[0]
                deepfake_probability = float(prediction)
            
            # Determine prediction
            is_deepfake = deepfake_probability > threshold
            prediction_label = "Deepfake" if is_deepfake else "Authentic"
            confidence = deepfake_probability if is_deepfake else (1 - deepfake_probability)
            
            result = {
                'video_path': video_path,
                'prediction': prediction_label,
                'deepfake_probability': float(deepfake_probability),
                'confidence': float(confidence),
                'threshold': threshold,
                'model_type': self.metadata.get('model_type', 'Unknown'),
                'frames_analyzed': len(gait_df)
            }
            
            # Print results
            print(f"ANALYSIS RESULTS:")
            print(f"   Prediction: {prediction_label}")
            print(f"   Deepfake Probability: {deepfake_probability:.3f}")
            print(f"   Confidence: {confidence:.3f}")
            print(f"   Frames Analyzed: {len(gait_df)}")
            print(f"   Model: {self.metadata.get('model_type', 'Unknown')}")
            
            return result
            
        except Exception as e:
            print(f"Prediction error: {e}")
            return {
                'video_path': video_path,
                'prediction': 'Error',
                'confidence': 0.0,
                'error': str(e)
            }
    
    def batch_predict(self, video_paths, output_file=None):
        """Predict deepfake for multiple videos"""
        print(f"\nBatch Analysis: {len(video_paths)} videos")
        print("=" * 60)
        
        results = []
        for i, video_path in enumerate(video_paths, 1):
            print(f"\n[{i}/{len(video_paths)}] Processing: {os.path.basename(video_path)}")
            result = self.predict_deepfake(video_path)
            if result:
                results.append(result)
        
        # Summary
        print("\n" + "=" * 60)
        print("BATCH ANALYSIS SUMMARY")
        print("=" * 60)
        
        authentic_count = sum(1 for r in results if r['prediction'] == 'Authentic')
        deepfake_count = sum(1 for r in results if r['prediction'] == 'Deepfake')
        error_count = sum(1 for r in results if r['prediction'] in ['Unknown', 'Error'])
        
        print(f"Total Videos: {len(results)}")
        print(f"Authentic: {authentic_count}")
        print(f"Deepfake: {deepfake_count}")
        print(f"Errors: {error_count}")
        
        # Save results if requested
        if output_file:
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\nResults saved to: {output_file}")
        
        return results

def main():
    parser = argparse.ArgumentParser(description='Deepfake Detection using Gait Analysis')
    parser.add_argument('video_path', help='Path to video file or directory containing videos')
    parser.add_argument('--threshold', type=float, default=0.5, 
                       help='Threshold for deepfake classification (default: 0.5)')
    parser.add_argument('--output', type=str, 
                       help='Output file to save results (JSON format)')
    parser.add_argument('--batch', action='store_true', 
                       help='Process all MP4 files in the directory')
    
    args = parser.parse_args()
    
    # Initialize pipeline
    detector = DeepfakeDetectionPipeline()
    
    if args.batch:
        # Batch processing
        if os.path.isdir(args.video_path):
            video_files = glob.glob(os.path.join(args.video_path, "*.mp4"))
            if not video_files:
                print(f"No MP4 files found in {args.video_path}")
                return
        else:
            print(f"Directory not found: {args.video_path}")
            return
        
        results = detector.batch_predict(video_files, args.output)
        
    else:
        # Single video processing
        if os.path.isfile(args.video_path):
            result = detector.predict_deepfake(args.video_path, args.threshold)
            
            if args.output and result:
                with open(args.output, 'w') as f:
                    json.dump([result], f, indent=2)
                print(f"\nResult saved to: {args.output}")
        else:
            print(f"Video file not found: {args.video_path}")

if __name__ == "__main__":
    main()