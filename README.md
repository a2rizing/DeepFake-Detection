# Gait-Based Person Recognition & Deepfake Detection

Deep learning pipeline for person identification and deepfake detection using gait analysis from video. Achieves **98.99% accuracy** on 13-person dataset.

## Overview

This system uses gait (walking pattern) analysis to:
1. **Identify individuals** based on their unique walking patterns
2. **Detect deepfakes** by flagging videos where gait doesn't match any known person
3. **Verify identity claims** by comparing gait against claimed identity

## Setup

### Prerequisites: Git LFS

This project uses Git LFS for large CSV files. Install it first:

```bash
# Windows (using Chocolatey or direct download)
choco install git-lfs
# OR download from https://git-lfs.com

# Linux
sudo apt-get install git-lfs

# macOS
brew install git-lfs
```

Then initialize Git LFS in the repository:

```bash
git lfs install
git lfs pull  # Download the CSV files
```

### Python Environment

```bash
# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### 1. Prepare Data

Place videos in `data/videos_augmented/{PersonName}/{F|S}/` structure:
- `F` = Front view, `S` = Side view
- Example: `data/videos_augmented/Aarav/F/Aarav_F1_original.mp4`

### 2. Extract Keypoints & Preprocess

```bash
# Extract pose keypoints from videos
python src/preprocessing/extract_gait.py --input data/videos_augmented --output data/gait_keypoints.csv --recursive

# Preprocess for training (creates X.npy, y.npy, labels.json)
python src/preprocessing/preprocess_gait.py --input_glob data/gait_keypoints.csv --out_dir data/processed_augmented
```

### 3. Train Models

```bash
# Train all models with cross-validation
python src/models/train_comprehensive.py --all
```

### 4. Run Detection

```bash
python detect.py path/to/video.mp4
```

## Testing

Run the structured test suite:

```bash
python run_structured_tests.py
```

Results are saved to `structured_test_results.json`.

## Requirements

- Python 3.8+
- TensorFlow 2.x
- MediaPipe
- OpenCV
- scikit-learn
