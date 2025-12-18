#!/usr/bin/env python3
"""
DeepFake Detection using Gait Analysis
Main entry point for video authentication

Usage:
    python detect.py <video_path>                    # Single video
    python detect.py <video_path> --threshold 0.7   # Custom threshold
    python detect.py <directory> --batch            # Batch processing
"""

import argparse
import os
import sys
import json
import numpy as np
import cv2
import mediapipe as mp
import joblib
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


class GaitDetector:
    """Gait-based deepfake detection system"""
    
    def __init__(self, model_path=None, scaler_path=None, labels_path=None):
        """Initialize the detector with trained model"""
        
        # Default paths
        if model_path is None:
            model_path = "models/best_model_20250919_162456.joblib"
        if scaler_path is None:
            scaler_path = "models/scaler_20250919_162456.joblib"
        if labels_path is None:
            labels_path = "data/processed/labels.json"
        
        # Load model
        if os.path.exists(model_path):
            self.model = joblib.load(model_path)
            print(f"✅ Loaded model from {model_path}")
        else:
            print(f"❌ Model not found at {model_path}")
            print("   Run training first: python train.py")
            sys.exit(1)
        
        # Load scaler
        if os.path.exists(scaler_path):
            self.scaler = joblib.load(scaler_path)
        else:
            self.scaler = None
            print("⚠️ Scaler not found, using raw features")
        
        # Load labels
        if os.path.exists(labels_path):
            with open(labels_path) as f:
                self.labels = json.load(f)
            self.id_to_name = {v: k for k, v in self.labels.items()}
        else:
            self.labels = None
            self.id_to_name = {}
        
        # Initialize MediaPipe
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
    
    def extract_keypoints(self, video_path, max_frames=64):
        """Extract pose keypoints from video"""
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"❌ Cannot open video: {video_path}")
            return None
        
        keypoints_list = []
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        skip_rate = max(1, total_frames // max_frames)
        
        frame_num = 0
        frames_processed = 0
        
        while cap.isOpened() and frames_processed < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_num += 1
            if frame_num % skip_rate != 0:
                continue
            
            frames_processed += 1
            
            try:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.pose.process(rgb_frame)
                
                if results.pose_landmarks:
                    landmarks = []
                    for lm in results.pose_landmarks.landmark:
                        landmarks.extend([lm.x, lm.y])
                    keypoints_list.append(landmarks)
            except Exception as e:
                continue
        
        cap.release()
        
        if len(keypoints_list) < 10:
            print(f"⚠️ Only {len(keypoints_list)} frames detected (minimum 10 required)")
            return None
        
        return np.array(keypoints_list)
    
    def calculate_features(self, keypoints):
        """Calculate gait features from keypoints sequence"""
        if keypoints is None or len(keypoints) == 0:
            return None
        
        # Calculate joint angles
        angles = []
        for frame in keypoints:
            frame_angles = []
            
            # Left knee angle (hip-knee-ankle)
            left_hip = np.array([frame[23*2], frame[23*2+1]])
            left_knee = np.array([frame[25*2], frame[25*2+1]])
            left_ankle = np.array([frame[27*2], frame[27*2+1]])
            
            v1 = left_hip - left_knee
            v2 = left_ankle - left_knee
            angle = np.arctan2(np.cross(v1, v2), np.dot(v1, v2))
            frame_angles.append(angle)
            
            # Right knee angle
            right_hip = np.array([frame[24*2], frame[24*2+1]])
            right_knee = np.array([frame[26*2], frame[26*2+1]])
            right_ankle = np.array([frame[28*2], frame[28*2+1]])
            
            v1 = right_hip - right_knee
            v2 = right_ankle - right_knee
            angle = np.arctan2(np.cross(v1, v2), np.dot(v1, v2))
            frame_angles.append(angle)
            
            angles.append(frame_angles)
        
        angles = np.array(angles)
        
        # Combine coordinates and angles
        combined = np.concatenate([keypoints, angles], axis=1)
        
        # Resample to fixed length
        target_frames = 64
        if len(combined) != target_frames:
            indices = np.linspace(0, len(combined) - 1, target_frames).astype(int)
            combined = combined[indices]
        
        return combined
    
    def detect(self, video_path, threshold=0.5, claimed_identity=None):
        """
        Detect if video is authentic or deepfake
        
        Args:
            video_path: Path to video file
            threshold: Decision threshold (default 0.5)
            claimed_identity: Optional claimed identity to verify
        
        Returns:
            dict with prediction results
        """
        print(f"\n🔍 Analyzing: {os.path.basename(video_path)}")
        
        # Extract keypoints
        keypoints = self.extract_keypoints(video_path)
        if keypoints is None:
            return {
                "video_path": video_path,
                "status": "error",
                "message": "Failed to extract pose landmarks"
            }
        
        print(f"   ✓ Extracted {len(keypoints)} frames")
        
        # Calculate features
        features = self.calculate_features(keypoints)
        if features is None:
            return {
                "video_path": video_path,
                "status": "error",
                "message": "Failed to calculate features"
            }
        
        # Flatten features for prediction
        features_flat = features.flatten().reshape(1, -1)
        
        # Scale features if scaler available
        if self.scaler is not None:
            try:
                features_flat = self.scaler.transform(features_flat)
            except Exception as e:
                print(f"   ⚠️ Scaler error, using raw features")
        
        # Predict
        try:
            prediction = self.model.predict(features_flat)[0]
            
            # Get probability if available
            if hasattr(self.model, 'predict_proba'):
                probabilities = self.model.predict_proba(features_flat)[0]
                confidence = max(probabilities)
            else:
                confidence = 1.0
            
            # Get predicted identity
            predicted_identity = self.id_to_name.get(prediction, f"Person_{prediction}")
            
            # Determine if authentic
            if claimed_identity:
                is_authentic = (predicted_identity.lower() == claimed_identity.lower())
            else:
                is_authentic = True  # No claim to verify
            
            result = {
                "video_path": video_path,
                "status": "success",
                "predicted_identity": predicted_identity,
                "confidence": float(confidence),
                "threshold": threshold,
                "is_authentic": is_authentic,
                "frames_analyzed": len(keypoints),
                "timestamp": datetime.now().isoformat()
            }
            
            if claimed_identity:
                result["claimed_identity"] = claimed_identity
                if is_authentic:
                    print(f"   ✅ AUTHENTIC - Matches claimed identity: {claimed_identity}")
                else:
                    print(f"   🚨 SUSPICIOUS - Predicted: {predicted_identity}, Claimed: {claimed_identity}")
            else:
                print(f"   📋 Identified as: {predicted_identity} (confidence: {confidence:.2%})")
            
            return result
            
        except Exception as e:
            return {
                "video_path": video_path,
                "status": "error",
                "message": str(e)
            }
    
    def batch_detect(self, directory, threshold=0.5, output_file=None):
        """Process all videos in a directory"""
        video_extensions = ('.mp4', '.avi', '.mov', '.webm')
        video_files = [f for f in os.listdir(directory) 
                       if f.lower().endswith(video_extensions)]
        
        if not video_files:
            print(f"❌ No video files found in {directory}")
            return []
        
        print(f"\n📁 Processing {len(video_files)} videos from {directory}")
        print("=" * 60)
        
        results = []
        for video_file in video_files:
            video_path = os.path.join(directory, video_file)
            result = self.detect(video_path, threshold)
            results.append(result)
        
        # Save results if output file specified
        if output_file:
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\n📄 Results saved to: {output_file}")
        
        # Summary
        success_count = sum(1 for r in results if r.get('status') == 'success')
        print(f"\n{'=' * 60}")
        print(f"📊 Processed {success_count}/{len(results)} videos successfully")
        
        return results


def main():
    parser = argparse.ArgumentParser(
        description="DeepFake Detection using Gait Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python detect.py video.mp4                     # Analyze single video
  python detect.py video.mp4 --identity John     # Verify claimed identity
  python detect.py videos/ --batch               # Process all videos in folder
  python detect.py videos/ --batch -o results.json
        """
    )
    
    parser.add_argument('path', help='Video file or directory path')
    parser.add_argument('--batch', '-b', action='store_true',
                        help='Process all videos in directory')
    parser.add_argument('--threshold', '-t', type=float, default=0.5,
                        help='Decision threshold (default: 0.5)')
    parser.add_argument('--identity', '-i', type=str,
                        help='Claimed identity to verify')
    parser.add_argument('--output', '-o', type=str,
                        help='Output JSON file for results')
    parser.add_argument('--model', '-m', type=str,
                        help='Path to trained model file')
    
    args = parser.parse_args()
    
    print("\n" + "=" * 60)
    print("🎭 DEEPFAKE DETECTION - Gait Analysis System")
    print("=" * 60)
    
    # Initialize detector
    detector = GaitDetector(model_path=args.model)
    
    if args.batch:
        if not os.path.isdir(args.path):
            print(f"❌ Not a directory: {args.path}")
            sys.exit(1)
        results = detector.batch_detect(args.path, args.threshold, args.output)
    else:
        if not os.path.isfile(args.path):
            print(f"❌ File not found: {args.path}")
            sys.exit(1)
        result = detector.detect(args.path, args.threshold, args.identity)
        
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(result, f, indent=2)
            print(f"\n📄 Results saved to: {args.output}")
        
        print(f"\n{'=' * 60}")
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
