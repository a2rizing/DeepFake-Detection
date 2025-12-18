# Deepfake Detection via Gait Analysis

Gait-based deepfake detection using pose estimation and deep learning models.

## Quick Start

### 1. Extract Gait Keypoints
```bash
python src/preprocessing/extract_gait.py --input data/videos_augmented --output data/gait_keypoints.csv --recursive
```

### 2. Preprocess Data for Training
```bash
python src/preprocessing/preprocess_gait.py --input_glob data/gait_keypoints.csv --out_dir data/processed
```

### 3. Train All Models
```bash
python src/models/train_models.py --data_dir data/processed --epochs 50 --batch_size 32
```

Train specific models only:
```bash
python src/models/train_models.py --models CNN LSTM CNN_LSTM
```

### 4. Run Detection
```bash
python detect.py <video_path>
```

---

## Project Structure

```
DeepFake-Detection/
├── data/
│   ├── videos/              # Original videos
│   ├── videos_augmented/    # Augmented videos ({Person}/{F|S}/*.mp4)
│   ├── gait_keypoints.csv   # Extracted pose keypoints
│   └── processed/           # Training-ready numpy arrays
│       ├── X.npy            # Features (N, seq_len, features)
│       ├── y.npy            # Labels
│       └── labels.json      # Label mapping
├── models/                  # Saved trained models (.keras)
├── results/                 # Training results and metrics
├── src/
│   ├── preprocessing/
│   │   ├── extract_gait.py      # MediaPipe pose extraction
│   │   └── preprocess_gait.py   # Feature normalization
│   └── models/
│       ├── models_extended.py   # Model architectures
│       ├── train_models.py      # Training script
│       └── evaluate_models.py   # Evaluation metrics
└── detect.py                # Main detection script
```

## Video Naming Convention

- **F** = FrontView (videos recorded from the front)
- **S** = SideView (videos recorded from the side)
- Pattern: `{PersonName}_{View}{Number}_{AugmentationType}.mp4`
- Example: `Arhaan_F1_aug3_brightness.mp4`

## Available Models

| Model | Description |
|-------|-------------|
| LSTM | Basic temporal sequence modeling |
| BiLSTM | Bidirectional LSTM for forward/backward context |
| GRU | Faster LSTM alternative |
| CNN | 1D CNN for spatial pattern extraction |
| CNN_LSTM | Hybrid spatial + temporal features |
| CNN_Transformer | CNN + self-attention mechanism |
| Attention_LSTM | Self-attention + LSTM |

## Future Improvements

- [ ] GradCAM visualization for model interpretability
- [ ] Vision Transformers (ViT) for video analysis
- [ ] Ensemble methods for improved accuracy
- [ ] Hyperparameter tuning with Optuna/Ray Tune
- [ ] Boosting methods (XGBoost, LightGBM)
