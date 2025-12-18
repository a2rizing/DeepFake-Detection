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
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


class GaitDetector:
    """Gait-based deepfake detection system"""
    
    def __init__(self, model_path=None, labels_path=None):
        """Initialize the detector with trained model"""
        
        # Default paths
        models_dir = "models"
        data_dir = "data/processed"
        
        # Try to load best model info
        best_info_path = os.path.join(models_dir, "best_model_info.json")
        
        if model_path is None:
            if os.path.exists(best_info_path):
                with open(best_info_path) as f:
                    best_info = json.load(f)
                model_path = best_info.get("model_path")
                self.num_classes = best_info.get("num_classes", 2)
                self.sequence_length = best_info.get("sequence_length", 64)
                self.num_features = best_info.get("num_features", 70)
                print(f"[OK] Using best model: {best_info.get('model_name')} (Accuracy: {best_info.get('accuracy', 0):.2%})")
            else:
                # Fallback to finding any .keras model
                keras_models = [f for f in os.listdir(models_dir) if f.endswith('.keras') and 'CNN' in f]
                if keras_models:
                    model_path = os.path.join(models_dir, sorted(keras_models)[-1])
                    self.num_classes = 13
                    self.sequence_length = 64
                    self.num_features = 70
                else:
                    print("[ERROR] No trained models found!")
                    print("   Run training first: python src/models/train_models.py")
                    sys.exit(1)
        
        if labels_path is None:
            labels_path = os.path.join(data_dir, "labels.json")
        
        # Load Keras model
        if os.path.exists(model_path):
            try:
                import tensorflow as tf
                tf.get_logger().setLevel('ERROR')
                self.model = tf.keras.models.load_model(model_path)
                print(f"[OK] Loaded model from {model_path}")
                self.is_keras_model = True
            except Exception as e:
                print(f"[ERROR] Failed to load Keras model: {e}")
                sys.exit(1)
        else:
            print(f"[ERROR] Model not found at {model_path}")
            print("   Run training first: python src/models/train_models.py")
            sys.exit(1)
        
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
            print(f"[ERROR] Cannot open video: {video_path}")
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
            print(f"[WARN] Only {len(keypoints_list)} frames detected (minimum 10 required)")
            return None
        
        return np.array(keypoints_list)
    
    def calculate_features(self, keypoints):
        """Calculate gait features from keypoints sequence (matches preprocessing pipeline)"""
        if keypoints is None or len(keypoints) == 0:
            return None
        
        EPS = 1e-8
        
        # Reshape flat keypoints to (T, L, 2) format
        # Input: (T, L*2) where L=33 landmarks
        T = len(keypoints)
        L = 33  # MediaPipe pose landmarks
        kps_xy = keypoints.reshape(T, L, 2)
        
        # Step 1: Center and scale normalization (matching preprocessing)
        left_hip_idx, right_hip_idx = 23, 24
        left_sh_idx, right_sh_idx = 11, 12
        
        hips = kps_xy[:, [left_hip_idx, right_hip_idx], :]  # (T, 2, 2)
        mid_hip = np.nanmean(hips, axis=1)  # (T, 2)
        shoulders = kps_xy[:, [left_sh_idx, right_sh_idx], :]  # (T, 2, 2)
        mid_sh = np.nanmean(shoulders, axis=1)  # (T, 2)
        
        # Center each frame on mid-hip
        centered = kps_xy - mid_hip[:, None, :]
        
        # Scale by mean torso length
        torso_len = np.linalg.norm(mid_sh - mid_hip, axis=1)
        torso_mean = np.nanmean(torso_len)
        if np.isnan(torso_mean) or torso_mean < EPS:
            torso_mean = 1.0
        normalized = centered / (torso_mean + EPS)
        
        # Step 2: Flatten to (T, L*2)
        flat = normalized.reshape(T, L * 2)
        
        # Step 3: Calculate 4 joint angles (matching preprocessing)
        angles = np.zeros((T, 4), dtype=np.float32)
        for t in range(T):
            # Left knee (hip-knee-ankle)
            angles[t, 0] = self._angle_at_point(
                normalized[t, 23], normalized[t, 25], normalized[t, 27])
            # Right knee (hip-knee-ankle)
            angles[t, 1] = self._angle_at_point(
                normalized[t, 24], normalized[t, 26], normalized[t, 28])
            # Left elbow (shoulder-elbow-wrist)
            angles[t, 2] = self._angle_at_point(
                normalized[t, 11], normalized[t, 13], normalized[t, 15])
            # Right elbow (shoulder-elbow-wrist)
            angles[t, 3] = self._angle_at_point(
                normalized[t, 12], normalized[t, 14], normalized[t, 16])
        
        # Step 4: Concatenate flat coords + angles (66 + 4 = 70 features per frame)
        combined = np.concatenate([flat, angles], axis=1)
        
        # Step 5: Resample to fixed length using interpolation (matching preprocessing)
        target_frames = 64
        if T != target_frames:
            combined = self._resample_sequence(combined, target_frames)
        
        return combined
    
    def _resample_sequence(self, seq, target_len):
        """Resample sequence to target length using linear interpolation"""
        T, D = seq.shape
        if T == target_len:
            return seq.copy()
        old_idx = np.linspace(0, 1, T)
        new_idx = np.linspace(0, 1, target_len)
        resampled = np.zeros((target_len, D), dtype=seq.dtype)
        for d in range(D):
            resampled[:, d] = np.interp(new_idx, old_idx, seq[:, d])
        return resampled
    
    def _angle_at_point(self, a, b, c):
        """Calculate angle at point b formed by points a-b-c using arccos (matches preprocessing)"""
        ba = a - b
        bc = c - b
        na = np.linalg.norm(ba)
        nb = np.linalg.norm(bc)
        denom = (na * nb) + 1e-8
        cosang = np.dot(ba, bc) / denom
        cosang = np.clip(cosang, -1.0, 1.0)
        return np.arccos(cosang)
    
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
        print(f"\nAnalyzing: {os.path.basename(video_path)}")
        
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
        
        # Reshape for Keras model (batch_size, sequence_length, num_features)
        features_reshaped = features.reshape(1, features.shape[0], features.shape[1])
        
        # Predict using Keras model
        try:
            # Keras model returns probabilities for each class
            probabilities = self.model.predict(features_reshaped, verbose=0)[0]
            prediction = int(np.argmax(probabilities))
            confidence = float(np.max(probabilities))
            
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
                "confidence": confidence,
                "threshold": threshold,
                "is_authentic": is_authentic,
                "frames_analyzed": len(keypoints),
                "timestamp": datetime.now().isoformat()
            }
            
            if claimed_identity:
                result["claimed_identity"] = claimed_identity
                if is_authentic:
                    print(f"   [OK] AUTHENTIC - Matches claimed identity: {claimed_identity}")
                else:
                    print(f"   [ALERT] SUSPICIOUS - Predicted: {predicted_identity}, Claimed: {claimed_identity}")
            else:
                print(f"   Identified as: {predicted_identity} (confidence: {confidence:.2%})")
            
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
            print(f"[ERROR] No video files found in {directory}")
            return []
        
        print(f"\nProcessing {len(video_files)} videos from {directory}")
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
            print(f"\nResults saved to: {output_file}")
        
        # Summary
        success_count = sum(1 for r in results if r.get('status') == 'success')
        print(f"\n{'=' * 60}")
        print(f"Processed {success_count}/{len(results)} videos successfully")
        
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
    print("DEEPFAKE DETECTION - Gait Analysis System")
    print("=" * 60)
    
    # Initialize detector
    detector = GaitDetector(model_path=args.model)
    
    if args.batch:
        if not os.path.isdir(args.path):
            print(f"[ERROR] Not a directory: {args.path}")
            sys.exit(1)
        results = detector.batch_detect(args.path, args.threshold, args.output)
    else:
        if not os.path.isfile(args.path):
            print(f"[ERROR] File not found: {args.path}")
            sys.exit(1)
        result = detector.detect(args.path, args.threshold, args.identity)
        
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(result, f, indent=2)
            print(f"\nResults saved to: {args.output}")
        
        print(f"\n{'=' * 60}")
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
