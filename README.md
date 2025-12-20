# Gait-Based Person Recognition

Deep learning pipeline for person identification using gait analysis from video. Achieves **98.99% accuracy** on 13-person dataset.

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
python src/models/train_comprehensive.py

# Or train specific model
python src/models/train_models.py --models CNN_Transformer --epochs 100
```

### 4. Evaluate

```bash
# Full evaluation with visualizations
python src/models/final_evaluation.py

# Test specific model
python src/models/test_models.py --model models/CNN_Transformer_V2_best.keras --all
```

### 5. Run Detection

```bash
python detect.py path/to/video.mp4
```

## Project Structure

```
├── src/
│   ├── preprocessing/     # Keypoint extraction & data prep
│   ├── models/           # Model architectures & training
│   └── visualization/    # GradCAM & interpretability
├── data/                 # Videos & processed data (gitignored)
├── models/               # Trained models (gitignored)
├── results/              # Metrics & visualizations (gitignored)
└── detect.py            # Main detection script
```

## Models

| Model | Description |
|-------|-------------|
| Enhanced_LSTM | 3-layer LSTM with recurrent dropout |
| Enhanced_BiLSTM | Bidirectional LSTM + attention |
| CNN_Transformer_V2 | CNN + Transformer (best performer) |
| MultiScale_CNN | Parallel convolutions (3,5,7 kernels) |
| ResNet_CNN | 1D ResNet with skip connections |

## Key Scripts

| Script | Purpose |
|--------|---------|
| `train_comprehensive.py` | Full training with CV & regularization |
| `test_models.py` | Test models, per-person accuracy |
| `final_evaluation.py` | Complete evaluation + visualizations |
| `build_ensemble.py` | Create ensemble from top models |
| `hyperparameter_tuning.py` | Grid search with Keras Tuner |

## Results

- **Accuracy**: 98.99%
- **ROC-AUC**: 99.84%
- **10/13 persons**: 100% accuracy
- **Deepfake detection**: Low confidence flagging works

## Requirements

- Python 3.8+
- TensorFlow 2.x
- MediaPipe
- OpenCV
- scikit-learn
