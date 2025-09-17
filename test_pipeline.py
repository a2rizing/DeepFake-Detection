#!/usr/bin/env python3
"""
Quick Test Script for DeepFake Detection Pipeline
Tests the complete pipeline without requiring all dependencies
"""

import os
import sys
import numpy as np

def test_imports():
    """Test essential imports"""
    print("🔍 Testing essential imports...")
    
    try:
        import cv2
        print("✅ OpenCV imported successfully")
    except ImportError:
        print("❌ OpenCV not found - install with: pip install opencv-python")
        return False
    
    try:
        import mediapipe
        print("✅ MediaPipe imported successfully")
    except ImportError:
        print("❌ MediaPipe not found - install with: pip install mediapipe")
        return False
    
    try:
        import sklearn
        print("✅ Scikit-learn imported successfully")
    except ImportError:
        print("❌ Scikit-learn not found - install with: pip install scikit-learn")
        return False
    
    try:
        import pandas as pd
        print("✅ Pandas imported successfully")
    except ImportError:
        print("❌ Pandas not found - install with: pip install pandas")
        return False
    
    try:
        import numpy as np
        print("✅ NumPy imported successfully")
    except ImportError:
        print("❌ NumPy not found - install with: pip install numpy")
        return False
    
    try:
        import matplotlib.pyplot as plt
        print("✅ Matplotlib imported successfully")
    except ImportError:
        print("❌ Matplotlib not found - install with: pip install matplotlib")
        return False
    
    # Optional imports
    try:
        import tensorflow as tf
        print("✅ TensorFlow imported successfully")
    except ImportError:
        print("⚠️  TensorFlow not found (optional for deep learning)")
    
    return True

def test_data_structure():
    """Test data directory structure"""
    print("\n📁 Testing data directory structure...")
    
    data_dir = "data"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        print(f"✅ Created {data_dir} directory")
    else:
        print(f"✅ {data_dir} directory exists")
    
    processed_dir = os.path.join(data_dir, "processed")
    if not os.path.exists(processed_dir):
        os.makedirs(processed_dir)
        print(f"✅ Created {processed_dir} directory")
    else:
        print(f"✅ {processed_dir} directory exists")
    
    models_dir = "models"
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
        print(f"✅ Created {models_dir} directory")
    else:
        print(f"✅ {models_dir} directory exists")
    
    return True

def test_video_files():
    """Check for video files in data directory"""
    print("\n🎥 Checking for video files...")
    
    data_dir = "data"
    video_extensions = ['.mp4', '.avi', '.mov', '.webm']
    video_files = []
    
    for file in os.listdir(data_dir):
        if any(file.lower().endswith(ext) for ext in video_extensions):
            video_files.append(file)
    
    if video_files:
        print(f"✅ Found {len(video_files)} video files:")
        for video in video_files:
            print(f"   - {video}")
    else:
        print("⚠️  No video files found in data/ directory")
        print("   Add some MP4 files to test the pipeline")
    
    return len(video_files) > 0

def test_syntax():
    """Test syntax of main Python files"""
    print("\n🔍 Testing Python file syntax...")
    
    test_files = [
        "src/preprocessing/extract_gait.py",
        "src/preprocessing/preprocess_gait.py",
        "src/models/train_baseline.py",
        "src/models/hyperparameter_tuning.py",
        "detect_deepfake.py",
        "evaluation_and_visualization.py"
    ]
    
    for file_path in test_files:
        if os.path.exists(file_path):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    compile(f.read(), file_path, 'exec')
                print(f"✅ {file_path} - syntax OK")
            except (SyntaxError, UnicodeDecodeError) as e:
                print(f"❌ {file_path} - error: {e}")
                return False
        else:
            print(f"⚠️  {file_path} - file not found")
    
    return True

def create_sample_requirements():
    """Create requirements.txt if it doesn't exist"""
    requirements_content = """# Core dependencies
numpy>=1.21.0
pandas>=1.3.0
opencv-python>=4.5.0
mediapipe>=0.8.0
scikit-learn>=1.0.0
matplotlib>=3.4.0
seaborn>=0.11.0
joblib>=1.0.0

# Optional dependencies
tensorflow>=2.8.0  # For deep learning models
scipy>=1.7.0       # For statistical functions

# Development dependencies (optional)
pytest>=6.0.0      # For testing
black>=21.0.0      # For code formatting
flake8>=3.9.0      # For linting
"""
    
    if not os.path.exists("requirements.txt"):
        with open("requirements.txt", "w") as f:
            f.write(requirements_content)
        print("✅ Created requirements.txt")
    else:
        print("✅ requirements.txt exists")

def main():
    """Run all tests"""
    print("🚀 DeepFake Detection Pipeline Test")
    print("=" * 50)
    
    all_tests_passed = True
    
    # Test imports
    if not test_imports():
        all_tests_passed = False
        print("\n❌ Import test failed. Install missing dependencies:")
        print("   pip install -r requirements.txt")
    
    # Test directory structure
    test_data_structure()
    
    # Test for video files
    has_videos = test_video_files()
    
    # Test syntax
    if not test_syntax():
        all_tests_passed = False
    
    # Create requirements.txt
    create_sample_requirements()
    
    print("\n" + "=" * 50)
    
    if all_tests_passed:
        print("🎉 ALL TESTS PASSED!")
        print("\nNext steps:")
        if has_videos:
            print("1. Run: python src/preprocessing/extract_gait.py")
            print("2. Run: python src/preprocessing/preprocess_gait.py")
            print("3. Run: python src/models/train_baseline.py")
            print("4. Run: python detect_deepfake.py data/your_video.mp4")
        else:
            print("1. Add MP4 video files to the data/ directory")
            print("2. Run the pipeline as shown in README.md")
        
        print("\nFor detailed instructions, see README.md")
    else:
        print("❌ SOME TESTS FAILED!")
        print("Fix the issues above before running the pipeline")
    
    return all_tests_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)