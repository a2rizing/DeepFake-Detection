# Gait-Based Deepfake Detection

A deep learning system for detecting deepfakes using gait analysis from video pose keypoints.

## Quick Start

```powershell
# 1. Extract gait keypoints from videos
python src/preprocessing/extract_gait.py

# 2. Preprocess into training features
python src/preprocessing/preprocess_gait.py

# 3. Train all models
python src/models/train_models.py

# 4. Evaluate and generate confusion matrices
python src/models/evaluate_models.py

# 5. Test on a video
python detect.py data/videos/Arhaan_F1.mp4
```

---

## Pipeline Overview

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  Raw Videos     │───▶│  Pose Keypoints  │───▶│  Training Data  │
│  data/videos/   │    │  (MediaPipe)     │    │  data/processed/│
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                                        │
                       ┌────────────────────────────────┘
                       ▼
              ┌─────────────────┐    ┌──────────────────┐
              │  Train Models   │───▶│  Evaluate Models │
              │  (7 architectures)   │  (Confusion Matrix)
              └─────────────────┘    └──────────────────┘
```

---

## Step-by-Step Instructions

### Step 1: Extract Gait Keypoints

Extract pose landmarks from all videos using MediaPipe:

```powershell
python src/preprocessing/extract_gait.py
```

**Output:** `data/gait_keypoints.csv` containing 33 body landmarks per frame

---

### Step 2: Preprocess Data

Convert raw keypoints into normalized feature matrices:

```powershell
python src/preprocessing/preprocess_gait.py
```

**Output:**
- `data/processed/X.npy` - Feature matrices (N, 64, 70)
- `data/processed/y.npy` - Labels
- `data/processed/labels.json` - Label mapping

---

### Step 3: Train Models

Train all 7 deep learning architectures:

```powershell
python src/models/train_models.py
```

**Options:**
```powershell
# Custom epochs and batch size
python src/models/train_models.py --epochs 100 --batch_size 32

# Train specific models only
python src/models/train_models.py --models LSTM BiLSTM CNN_LSTM
```

**Available Models:**
| Model | Description |
|-------|-------------|
| LSTM | Basic temporal modeling |
| BiLSTM | Bidirectional LSTM |
| GRU | Gated Recurrent Unit |
| CNN | 1D Convolution |
| CNN_LSTM | Hybrid CNN + LSTM |
| CNN_Transformer | CNN + Transformer attention |
| Attention_LSTM | Self-attention + LSTM |

**Output:**
- `models/*.keras` - Trained model files
- `models/best_model_info.json` - Best model metadata
- `results/training_results_*.json` - Training metrics

---

### Step 4: Evaluate Models

Generate confusion matrices and performance reports:

```powershell
python src/models/evaluate_models.py
```

**Output:**
- `results/confusion_matrices.png` - Confusion matrix plots
- `results/model_comparison.png` - Performance bar chart
- `results/evaluation_report_*.txt` - Detailed text report

---

### Step 5: Test Detection

Test the trained model on a video:

```powershell
# Single video
python detect.py data/videos/Arhaan_F1.mp4

# Verify claimed identity
python detect.py video.mp4 --identity Arhaan

# Batch process folder
python detect.py data/videos/ --batch -o results.json
```

---

## Data Augmentation (Optional)

If you need more training data:

```powershell
# Extract frames from videos
python src/preprocessing/video_to_frames.py

# Apply 5 augmentations per video
python src/preprocessing/augment_frames.py

# Convert frames back to videos
python src/preprocessing/frames_to_video.py
```

---

## Project Structure

```
DeepFake-Detection/
├── data/
│   ├── videos/              # Input videos (PersonName_F1.mp4, etc.)
│   ├── frames/              # Extracted frames
│   ├── frames_augmented/    # Augmented frames
│   ├── processed/           # Training data (X.npy, y.npy)
│   └── gait_keypoints.csv   # Raw pose keypoints
├── models/                  # Trained model files
├── results/                 # Evaluation results
├── src/
│   ├── preprocessing/       # Data preprocessing scripts
│   └── models/              # Model architectures & training
├── detect.py                # Main detection script
└── README.md
```

---

## Requirements

```powershell
pip install -r requirements.txt
```

Key dependencies:
- TensorFlow 2.x
- MediaPipe
- OpenCV
- scikit-learn
- numpy, pandas
- matplotlib, seaborn (for visualization)
