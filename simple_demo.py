"""
Simple Demo Script - Test deepfake detection on a video
Usage: python simple_demo.py <path_to_video>
"""

import sys
import os
import numpy as np
import joblib
from src.preprocessing.improved_gait_extractor import ImprovedGaitExtractor
import warnings
warnings.filterwarnings('ignore')

def load_model():
    """Load the trained model"""
    model_path = 'models/saved/best_model.pkl'
    scaler_path = 'data/processed/scaler.pkl'
    
    if not os.path.exists(model_path):
        print("❌ Model not found!")
        print("   Please train the model first: python extract_and_train.py")
        return None, None
    
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    
    return model, scaler

def predict_video(video_path, model, scaler):
    """Predict if video is deepfake"""
    
    print(f"\n🎬 Analyzing video: {os.path.basename(video_path)}")
    print("=" * 60)
    
    # Extract features
    extractor = ImprovedGaitExtractor()
    print("📊 Extracting gait features...")
    features = extractor.process_video(video_path)
    
    if features is None:
        print("❌ Could not extract features from video!")
        print("   Make sure the video shows a person walking with full body visible.")
        return None
    
    # Prepare feature vector (same as training)
    temporal = features['temporal'].flatten()
    spatial = features['spatial'].flatten()
    angles = features['angles'].flatten()
    
    feature_vector = np.array([
        np.mean(temporal), np.std(temporal), np.min(temporal), np.max(temporal),
        np.mean(spatial), np.std(spatial), np.min(spatial), np.max(spatial),
        np.mean(angles), np.std(angles), np.min(angles), np.max(angles)
    ]).reshape(1, -1)
    
    # Normalize
    feature_vector_scaled = scaler.transform(feature_vector)
    
    # Predict
    print("🔍 Running prediction...")
    prediction = model.predict(feature_vector_scaled)[0]
    probabilities = model.predict_proba(feature_vector_scaled)[0]
    
    # Display results
    print("\n" + "=" * 60)
    print("🎯 RESULTS")
    print("=" * 60)
    
    if prediction == 0:
        print("✅ Prediction: AUTHENTIC (Real)")
        print(f"   Confidence: {probabilities[0]:.2%}")
    else:
        print("⚠️  Prediction: DEEPFAKE (Fake)")
        print(f"   Confidence: {probabilities[1]:.2%}")
    
    print(f"\n📊 Probability Breakdown:")
    print(f"   Real: {probabilities[0]:.2%}")
    print(f"   Fake: {probabilities[1]:.2%}")
    print("=" * 60)
    
    return {
        'prediction': 'REAL' if prediction == 0 else 'FAKE',
        'confidence': float(probabilities[prediction]),
        'probabilities': {
            'real': float(probabilities[0]),
            'fake': float(probabilities[1])
        }
    }

def main():
    """Main demo function"""
    
    print("🎭 DEEPFAKE DETECTION DEMO")
    print("=" * 60)
    
    # Check arguments
    if len(sys.argv) < 2:
        print("\n❌ Please provide a video path!")
        print("\nUsage:")
        print("   python simple_demo.py <path_to_video>")
        print("\nExample:")
        print("   python simple_demo.py data/raw/real/video1.mp4")
        print("=" * 60)
        return
    
    video_path = sys.argv[1]
    
    # Check if video exists
    if not os.path.exists(video_path):
        print(f"❌ Video not found: {video_path}")
        return
    
    # Load model
    model, scaler = load_model()
    if model is None:
        return
    
    # Predict
    result = predict_video(video_path, model, scaler)
    
    if result:
        print("\n✅ Analysis complete!")
        print("\n💡 Tips for better accuracy:")
        print("   - Ensure full body is visible in the video")
        print("   - Video should show natural walking/movement")
        print("   - Better lighting improves pose detection")
        print("=" * 60)

if __name__ == "__main__":
    main()
