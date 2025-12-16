"""
Complete Pipeline: Extract features and train model
Run this after you have videos in data/raw/real and data/raw/fake
"""

import os
import sys
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

# Add src to path
sys.path.append('src')

from preprocessing.improved_gait_extractor import ImprovedGaitExtractor

def check_data_exists():
    """Check if data directories exist and contain videos"""
    real_dir = "data/raw/real"
    fake_dir = "data/raw/fake"
    
    if not os.path.exists(real_dir) or not os.path.exists(fake_dir):
        print("❌ Data directories not found!")
        print("\n📁 Please create:")
        print("   data/raw/real/  (for authentic videos)")
        print("   data/raw/fake/  (for deepfake videos)")
        print("\n💡 Run: python download_sample_data.py for instructions")
        return False
    
    real_videos = [f for f in os.listdir(real_dir) if f.endswith(('.mp4', '.avi', '.mov'))]
    fake_videos = [f for f in os.listdir(fake_dir) if f.endswith(('.mp4', '.avi', '.mov'))]
    
    if len(real_videos) == 0 or len(fake_videos) == 0:
        print(f"❌ Not enough videos found!")
        print(f"   Real videos: {len(real_videos)}")
        print(f"   Fake videos: {len(fake_videos)}")
        print("\n💡 You need at least 5 videos in each directory")
        print("💡 Run: python download_sample_data.py for instructions")
        return False
    
    print(f"✅ Found {len(real_videos)} real videos")
    print(f"✅ Found {len(fake_videos)} fake videos")
    return True

def extract_features():
    """Extract gait features from all videos"""
    
    print("\n" + "="*60)
    print("STEP 1: Extracting Gait Features")
    print("="*60)
    
    extractor = ImprovedGaitExtractor()
    
    real_features, fake_features = extractor.process_dataset(
        real_dir="data/raw/real",
        fake_dir="data/raw/fake",
        output_dir="data/processed"
    )
    
    if len(real_features) == 0 or len(fake_features) == 0:
        print("❌ Feature extraction failed!")
        return None, None
    
    return real_features, fake_features

def prepare_training_data(real_features, fake_features):
    """Prepare data for training"""
    
    print("\n" + "="*60)
    print("STEP 2: Preparing Training Data")
    print("="*60)
    
    X = []
    y = []
    
    # Process real videos (label = 0)
    for features in real_features:
        # Flatten all features
        temporal = features['temporal'].flatten()
        spatial = features['spatial'].flatten()
        angles = features['angles'].flatten()
        
        # Combine all features
        combined = np.concatenate([temporal, spatial, angles])
        
        # Use statistics if sequences have different lengths
        feature_vector = np.array([
            np.mean(temporal), np.std(temporal), np.min(temporal), np.max(temporal),
            np.mean(spatial), np.std(spatial), np.min(spatial), np.max(spatial),
            np.mean(angles), np.std(angles), np.min(angles), np.max(angles)
        ])
        
        X.append(feature_vector)
        y.append(0)  # Real
    
    # Process fake videos (label = 1)
    for features in fake_features:
        temporal = features['temporal'].flatten()
        spatial = features['spatial'].flatten()
        angles = features['angles'].flatten()
        
        feature_vector = np.array([
            np.mean(temporal), np.std(temporal), np.min(temporal), np.max(temporal),
            np.mean(spatial), np.std(spatial), np.min(spatial), np.max(spatial),
            np.mean(angles), np.std(angles), np.min(angles), np.max(angles)
        ])
        
        X.append(feature_vector)
        y.append(1)  # Fake
    
    X = np.array(X)
    y = np.array(y)
    
    print(f"✅ Dataset shape: {X.shape}")
    print(f"✅ Real samples: {np.sum(y == 0)}")
    print(f"✅ Fake samples: {np.sum(y == 1)}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Normalize
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Save processed data
    os.makedirs('data/processed', exist_ok=True)
    np.savez('data/processed/training_data.npz',
             X_train=X_train_scaled, X_test=X_test_scaled,
             y_train=y_train, y_test=y_test)
    
    joblib.dump(scaler, 'data/processed/scaler.pkl')
    
    print("✅ Data saved to: data/processed/training_data.npz")
    
    return X_train_scaled, X_test_scaled, y_train, y_test

def train_models(X_train, X_test, y_train, y_test):
    """Train multiple models"""
    
    print("\n" + "="*60)
    print("STEP 3: Training Models")
    print("="*60)
    
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.svm import SVC
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
    
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
        'SVM': SVC(kernel='rbf', probability=True, random_state=42),
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000)
    }
    
    results = {}
    best_model = None
    best_accuracy = 0
    
    for name, model in models.items():
        print(f"\n🔄 Training {name}...")
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"✅ {name} Accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=['Real', 'Fake']))
        
        results[name] = {
            'model': model,
            'accuracy': accuracy,
            'predictions': y_pred
        }
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = (name, model)
    
    # Save best model
    os.makedirs('models/saved', exist_ok=True)
    joblib.dump(best_model[1], 'models/saved/best_model.pkl')
    
    print("\n" + "="*60)
    print(f"🏆 BEST MODEL: {best_model[0]}")
    print(f"🎯 Accuracy: {best_accuracy:.4f}")
    print(f"✅ Saved to: models/saved/best_model.pkl")
    print("="*60)
    
    return results

def main():
    """Main pipeline"""
    
    print("🎭 DEEPFAKE DETECTION - COMPLETE PIPELINE")
    print("="*60)
    
    # Check data
    if not check_data_exists():
        return
    
    # Extract features
    real_features, fake_features = extract_features()
    
    if real_features is None or fake_features is None:
        return
    
    # Prepare training data
    X_train, X_test, y_train, y_test = prepare_training_data(real_features, fake_features)
    
    # Train models
    results = train_models(X_train, X_test, y_train, y_test)
    
    print("\n" + "="*60)
    print("✅ PIPELINE COMPLETE!")
    print("="*60)
    print("\n🚀 Next steps:")
    print("   1. Test on new videos: python simple_demo.py <video_path>")
    print("   2. View results: Check models/saved/")
    print("="*60)

if __name__ == "__main__":
    main()
