"""
Re-extract Gait Features and Retrain with Updated Videos
Handles multiple videos per person for better accuracy
"""

import os
import sys
import numpy as np
import json
from collections import defaultdict
import re

# Add src to path
sys.path.append('src')

from preprocessing.improved_gait_extractor import ImprovedGaitExtractor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

def scan_videos(data_dir='data'):
    """
    Scan data directory for all videos and group by person
    
    Supports naming conventions:
    - PersonName.mp4 (single video)
    - PersonName_1.mp4, PersonName_2.mp4 (multiple videos)
    - PersonName-1.mp4, PersonName-2.mp4 (alternative)
    """
    
    print("🔍 Scanning for videos...")
    print("=" * 80)
    
    video_files = [f for f in os.listdir(data_dir) 
                   if f.endswith('.mp4') and os.path.isfile(os.path.join(data_dir, f))]
    
    if not video_files:
        print("❌ No video files found in data/ directory!")
        return {}
    
    # Group videos by person
    person_videos = defaultdict(list)
    
    for video_file in video_files:
        # Remove extension
        name_part = os.path.splitext(video_file)[0]
        
        # Try to extract person name (remove numbers and separators)
        # Handles: "Aditya_1", "Aditya-1", "Aditya1", "Aditya"
        person_name = re.sub(r'[-_]?\d+$', '', name_part)
        person_name = person_name.strip().lower()
        
        person_videos[person_name].append(video_file)
    
    # Display results
    print(f"\n✅ Found {len(video_files)} video(s) for {len(person_videos)} person(s):\n")
    
    for person, videos in sorted(person_videos.items()):
        print(f"   📹 {person.capitalize()}: {len(videos)} video(s)")
        for vid in sorted(videos):
            print(f"      - {vid}")
        print()
    
    return person_videos

def confirm_and_update(person_videos):
    """Allow user to confirm or modify person names"""
    
    print("=" * 80)
    print("📝 Please verify the person names:")
    print("=" * 80)
    
    updated_videos = {}
    
    for person, videos in sorted(person_videos.items()):
        print(f"\n👤 Found videos for: '{person.capitalize()}'")
        for vid in videos:
            print(f"   - {vid}")
        
        response = input(f"   Is '{person.capitalize()}' correct? (Y/n) or enter correct name: ").strip()
        
        if response.lower() in ['y', 'yes', '']:
            updated_videos[person] = videos
            print(f"   ✅ Keeping: {person.capitalize()}")
        else:
            new_name = response.lower()
            updated_videos[new_name] = videos
            print(f"   ✅ Renamed to: {new_name.capitalize()}")
    
    return updated_videos

def extract_all_features(person_videos, data_dir='data', output_dir='data/processed'):
    """Extract gait features from all videos"""
    
    print("\n" + "=" * 80)
    print("🎬 EXTRACTING GAIT FEATURES")
    print("=" * 80)
    
    extractor = ImprovedGaitExtractor()
    
    all_features = []
    all_labels = []
    label_mapping = {}
    
    # Create label mapping
    for idx, person_name in enumerate(sorted(person_videos.keys())):
        label_mapping[person_name] = idx
    
    print(f"\n📋 Label Mapping:")
    for name, label_id in sorted(label_mapping.items(), key=lambda x: x[1]):
        print(f"   {label_id}: {name.capitalize()}")
    
    # Extract features for each person
    for person_name, videos in sorted(person_videos.items()):
        person_id = label_mapping[person_name]
        
        print(f"\n{'='*80}")
        print(f"👤 Processing: {person_name.capitalize()} (ID: {person_id})")
        print(f"{'='*80}")
        
        for video_file in videos:
            video_path = os.path.join(data_dir, video_file)
            
            print(f"\n📹 Extracting from: {video_file}")
            
            try:
                # Extract features
                features = extractor.process_video(video_path)
                
                if features is not None:
                    # Get temporal features (use mean across all frames)
                    temporal = features['temporal']
                    spatial = features['spatial']
                    angles = features['angles']
                    
                    # Create feature vector (using statistics across sequence)
                    feature_vector = np.concatenate([
                        np.mean(temporal, axis=0),
                        np.std(temporal, axis=0),
                        np.mean(spatial, axis=0),
                        np.std(spatial, axis=0),
                        np.mean(angles, axis=0),
                        np.std(angles, axis=0)
                    ])
                    
                    all_features.append(feature_vector)
                    all_labels.append(person_id)
                    
                    print(f"   ✅ Extracted {len(feature_vector)} features")
                else:
                    print(f"   ⚠️  Failed to extract features from {video_file}")
                    
            except Exception as e:
                print(f"   ❌ Error processing {video_file}: {e}")
    
    # Convert to numpy arrays
    X = np.array(all_features)
    y = np.array(all_labels)
    
    print(f"\n{'='*80}")
    print(f"✅ FEATURE EXTRACTION COMPLETE")
    print(f"{'='*80}")
    print(f"\n📊 Dataset Summary:")
    print(f"   Total samples: {len(X)}")
    print(f"   Features per sample: {X.shape[1]}")
    print(f"   Number of people: {len(label_mapping)}")
    
    print(f"\n📈 Samples per person:")
    for person_name, person_id in sorted(label_mapping.items(), key=lambda x: x[1]):
        count = np.sum(y == person_id)
        print(f"   {person_name.capitalize()}: {count} sample(s)")
    
    # Save data
    os.makedirs(output_dir, exist_ok=True)
    
    np.save(os.path.join(output_dir, 'X.npy'), X)
    np.save(os.path.join(output_dir, 'y.npy'), y)
    
    with open(os.path.join(output_dir, 'labels.json'), 'w') as f:
        json.dump(label_mapping, f, indent=2)
    
    print(f"\n💾 Data saved to: {output_dir}")
    print(f"   - X.npy (features)")
    print(f"   - y.npy (labels)")
    print(f"   - labels.json (name mapping)")
    
    return X, y, label_mapping

def retrain_models(X, y):
    """Train models with new data"""
    
    print("\n" + "=" * 80)
    print("🤖 TRAINING MODELS")
    print("=" * 80)
    
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.svm import SVC
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, classification_report
    
    # Split data if we have enough samples
    if len(X) > 10:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        print(f"\n📊 Data split:")
        print(f"   Training: {len(X_train)} samples")
        print(f"   Testing: {len(X_test)} samples")
    else:
        print(f"\n⚠️  Limited data ({len(X)} samples). Using all for training.")
        X_train = X_test = X
        y_train = y_test = y
    
    # Normalize
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Save scaler
    joblib.dump(scaler, 'data/processed/scaler.pkl')
    
    # Train models
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
        'SVM': SVC(kernel='rbf', probability=True, random_state=42),
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000)
    }
    
    best_model = None
    best_accuracy = 0
    best_name = None
    
    for name, model in models.items():
        print(f"\n{'='*80}")
        print(f"🔄 Training {name}...")
        print(f"{'='*80}")
        
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"\n✅ {name} Accuracy: {accuracy:.2%}")
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = model
            best_name = name
    
    # Save best model
    os.makedirs('models/saved', exist_ok=True)
    joblib.dump(best_model, 'models/saved/best_model.pkl')
    
    print(f"\n{'='*80}")
    print(f"🏆 BEST MODEL: {best_name}")
    print(f"🎯 Accuracy: {best_accuracy:.2%}")
    print(f"{'='*80}")
    print(f"\n💾 Saved to: models/saved/best_model.pkl")
    
    return best_model, best_accuracy

def main():
    """Main pipeline"""
    
    print("\n")
    print("█" * 80)
    print("█" + " " * 78 + "█")
    print("█" + " " * 15 + "GAIT FEATURE RE-EXTRACTION & RETRAINING" + " " * 23 + "█")
    print("█" + " " * 78 + "█")
    print("█" * 80)
    print()
    
    # Scan videos
    person_videos = scan_videos('data')
    
    if not person_videos:
        return
    
    # Confirm names
    print("\n" + "=" * 80)
    response = input("Would you like to verify/update person names? (Y/n): ").strip()
    
    if response.lower() in ['y', 'yes', '']:
        person_videos = confirm_and_update(person_videos)
    
    # Extract features
    X, y, label_mapping = extract_all_features(person_videos)
    
    if len(X) == 0:
        print("\n❌ No features extracted. Please check your videos.")
        return
    
    # Retrain models
    model, accuracy = retrain_models(X, y)
    
    print("\n" + "=" * 80)
    print("✅ PIPELINE COMPLETE!")
    print("=" * 80)
    
    print("\n📋 Summary:")
    print(f"   • People: {len(label_mapping)}")
    print(f"   • Total videos: {len(X)}")
    print(f"   • Best accuracy: {accuracy:.2%}")
    
    print("\n🚀 Next Steps:")
    print("   1. Generate confusion matrices: python quick_confusion_matrix.py")
    print("   2. Run verification demo: python gait_verification_system.py")
    print("   3. View outputs: python show_outputs.py")
    
    print("\n" + "=" * 80)

if __name__ == "__main__":
    main()
