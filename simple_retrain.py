"""
Simple Retraining Script - No TensorFlow dependency
Extracts gait features and trains models on new videos
"""

import os
import cv2
import numpy as np
import json
import re
from collections import defaultdict
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib

# MediaPipe for pose detection
import mediapipe as mp

class SimpleGaitExtractor:
    """Extract gait features using MediaPipe"""
    
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
    
    def extract_from_video(self, video_path, max_frames=64):
        """Extract keypoints from video"""
        
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"❌ Cannot open: {video_path}")
            return None
        
        keypoints_list = []
        frame_count = 0
        
        while len(keypoints_list) < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Skip some frames for efficiency
            if frame_count % 2 != 0:
                continue
            
            # Convert to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process with MediaPipe
            results = self.pose.process(rgb_frame)
            
            if results.pose_landmarks:
                # Extract 33 landmarks (x, y, z, visibility)
                landmarks = []
                for landmark in results.pose_landmarks.landmark:
                    landmarks.extend([landmark.x, landmark.y, landmark.z, landmark.visibility])
                
                keypoints_list.append(landmarks)
        
        cap.release()
        
        if len(keypoints_list) == 0:
            return None
        
        # Pad or truncate to max_frames
        if len(keypoints_list) < max_frames:
            # Pad with last frame
            while len(keypoints_list) < max_frames:
                keypoints_list.append(keypoints_list[-1])
        else:
            keypoints_list = keypoints_list[:max_frames]
        
        return np.array(keypoints_list)
    
    def calculate_features(self, keypoints):
        """Calculate gait features from keypoints sequence"""
        
        if keypoints is None or len(keypoints) == 0:
            return None
        
        # Statistical features across frames
        features = []
        
        # Mean and std of all keypoints
        features.extend(np.mean(keypoints, axis=0))
        features.extend(np.std(keypoints, axis=0))
        
        # Min and max
        features.extend(np.min(keypoints, axis=0))
        features.extend(np.max(keypoints, axis=0))
        
        return np.array(features)

def scan_videos(data_dir='data'):
    """Scan for all MP4 videos and group by person"""
    
    print("\n" + "=" * 80)
    print("🔍 SCANNING FOR VIDEOS")
    print("=" * 80)
    
    video_files = [f for f in os.listdir(data_dir) 
                   if f.endswith('.mp4') and os.path.isfile(os.path.join(data_dir, f))]
    
    if not video_files:
        print("❌ No videos found!")
        return {}
    
    # Group by person (remove numbers from names)
    person_videos = defaultdict(list)
    
    for video_file in video_files:
        name_part = os.path.splitext(video_file)[0]
        # Remove trailing numbers: "Anshul1" -> "Anshul"
        person_name = re.sub(r'[-_]?\d+$', '', name_part).strip().lower()
        person_videos[person_name].append(video_file)
    
    print(f"\n✅ Found {len(video_files)} video(s) for {len(person_videos)} person(s):\n")
    
    for person, videos in sorted(person_videos.items()):
        print(f"   📹 {person.capitalize()}: {len(videos)} video(s)")
        for vid in sorted(videos):
            print(f"      └─ {vid}")
    
    print("\n" + "=" * 80)
    
    return person_videos

def extract_features(person_videos, data_dir='data'):
    """Extract features from all videos"""
    
    print("\n" + "=" * 80)
    print("🎬 EXTRACTING FEATURES")
    print("=" * 80)
    
    extractor = SimpleGaitExtractor()
    
    all_features = []
    all_labels = []
    label_mapping = {}
    
    # Create label mapping
    for idx, person in enumerate(sorted(person_videos.keys())):
        label_mapping[person] = idx
    
    print(f"\n📋 Label Mapping:")
    for name, idx in sorted(label_mapping.items(), key=lambda x: x[1]):
        print(f"   {idx}: {name.capitalize()}")
    
    # Process each video
    total_videos = sum(len(vids) for vids in person_videos.values())
    current = 0
    
    for person, videos in sorted(person_videos.items()):
        person_id = label_mapping[person]
        
        print(f"\n{'─'*80}")
        print(f"👤 {person.capitalize()} (ID: {person_id})")
        print(f"{'─'*80}")
        
        for video_file in sorted(videos):
            current += 1
            video_path = os.path.join(data_dir, video_file)
            
            print(f"\n[{current}/{total_videos}] Processing: {video_file}")
            
            try:
                # Extract keypoints
                keypoints = extractor.extract_from_video(video_path)
                
                if keypoints is not None:
                    # Calculate features
                    features = extractor.calculate_features(keypoints)
                    
                    if features is not None:
                        all_features.append(features)
                        all_labels.append(person_id)
                        print(f"   ✅ Extracted {len(features)} features")
                    else:
                        print(f"   ⚠️  Failed to calculate features")
                else:
                    print(f"   ⚠️  Failed to extract keypoints")
            
            except Exception as e:
                print(f"   ❌ Error: {e}")
    
    # Convert to arrays
    X = np.array(all_features)
    y = np.array(all_labels)
    
    print(f"\n{'='*80}")
    print("✅ EXTRACTION COMPLETE")
    print(f"{'='*80}")
    print(f"\n📊 Dataset:")
    print(f"   Total samples: {len(X)}")
    print(f"   Features/sample: {X.shape[1]}")
    print(f"   People: {len(label_mapping)}")
    
    print(f"\n📈 Samples per person:")
    for name, idx in sorted(label_mapping.items(), key=lambda x: x[1]):
        count = np.sum(y == idx)
        print(f"   {name.capitalize()}: {count}")
    
    return X, y, label_mapping

def train_models(X, y):
    """Train classification models"""
    
    print("\n" + "=" * 80)
    print("🤖 TRAINING MODELS")
    print("=" * 80)
    
    # Check if we have enough data for split
    # Need at least 2 samples per class for stratified split
    unique, counts = np.unique(y, return_counts=True)
    min_samples = np.min(counts)
    
    if len(X) >= 10 and min_samples >= 2:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        print(f"\n📊 Split: Train={len(X_train)}, Test={len(X_test)}")
    else:
        print(f"\n⚠️  Limited data ({len(X)} samples, min per class: {min_samples}). Using all for training.")
        X_train = X_test = X
        y_train = y_test = y
    
    # Normalize
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Define models
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
        'SVM': SVC(kernel='rbf', probability=True, random_state=42),
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000)
    }
    
    best_model = None
    best_accuracy = 0
    best_name = None
    
    print()
    for name, model in models.items():
        print(f"{'─'*80}")
        print(f"Training: {name}")
        print(f"{'─'*80}")
        
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"Accuracy: {accuracy:.2%}")
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = model
            best_name = name
    
    print(f"\n{'='*80}")
    print(f"🏆 BEST MODEL: {best_name}")
    print(f"🎯 ACCURACY: {best_accuracy:.2%}")
    print(f"{'='*80}")
    
    return best_model, scaler, best_accuracy, best_name

def save_results(X, y, label_mapping, model, scaler, accuracy, model_name):
    """Save all results"""
    
    print("\n" + "=" * 80)
    print("💾 SAVING RESULTS")
    print("=" * 80)
    
    # Create directories
    os.makedirs('data/processed', exist_ok=True)
    os.makedirs('models/saved', exist_ok=True)
    
    # Save processed data
    np.save('data/processed/X.npy', X)
    np.save('data/processed/y.npy', y)
    
    with open('data/processed/labels.json', 'w') as f:
        json.dump(label_mapping, f, indent=2)
    
    print("\n✅ Saved processed data:")
    print("   • data/processed/X.npy")
    print("   • data/processed/y.npy")
    print("   • data/processed/labels.json")
    
    # Save model and scaler
    joblib.dump(model, 'models/saved/best_model.pkl')
    joblib.dump(scaler, 'data/processed/scaler.pkl')
    
    print("\n✅ Saved models:")
    print("   • models/saved/best_model.pkl")
    print("   • data/processed/scaler.pkl")
    
    # Save metadata
    metadata = {
        'model_type': model_name,
        'accuracy': float(accuracy),
        'num_people': len(label_mapping),
        'num_samples': len(X),
        'feature_dim': X.shape[1],
        'people': list(label_mapping.keys())
    }
    
    with open('models/saved/metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print("   • models/saved/metadata.json")
    
    print("\n" + "=" * 80)

def main():
    """Main pipeline"""
    
    print("\n")
    print("█" * 80)
    print("█" + " " * 20 + "SIMPLE GAIT RECOGNITION RETRAINING" + " " * 25 + "█")
    print("█" * 80)
    
    # Scan videos
    person_videos = scan_videos('data')
    
    if not person_videos:
        print("\n❌ No videos found. Please add videos to data/ folder")
        return
    
    # Extract features
    X, y, label_mapping = extract_features(person_videos, 'data')
    
    if len(X) == 0:
        print("\n❌ No features extracted. Check videos.")
        return
    
    # Train models
    model, scaler, accuracy, model_name = train_models(X, y)
    
    # Save everything
    save_results(X, y, label_mapping, model, scaler, accuracy, model_name)
    
    print("\n" + "=" * 80)
    print("✅ ✅ ✅  PIPELINE COMPLETE  ✅ ✅ ✅")
    print("=" * 80)
    
    print("\n📋 SUMMARY:")
    print(f"   • People: {len(label_mapping)}")
    print(f"   • Videos: {len(X)}")
    print(f"   • Best Model: {model_name}")
    print(f"   • Accuracy: {accuracy:.2%}")
    
    print("\n🚀 NEXT STEPS:")
    print("   1. Generate confusion matrix:")
    print("      python quick_confusion_matrix.py")
    print()
    print("   2. Test verification system:")
    print("      python gait_verification_system.py")
    print()
    print("   3. View all outputs:")
    print("      python show_outputs.py")
    
    print("\n" + "=" * 80 + "\n")

if __name__ == "__main__":
    main()
