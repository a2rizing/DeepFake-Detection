# 🚀 Quick Start Guide - DeepFake Detection

## TL;DR - Run This Stuff

### 1. First Time Setup (Do Once)

```bash
# Install dependencies
pip install -r requirements.txt

# Test everything works
python test_pipeline.py
```

### 2. Basic Training Pipeline (Do Once)

```bash
# Step 1: Extract gait from your videos
python src/preprocessing/extract_gait.py

# Step 2: Prepare data for machine learning
python src/preprocessing/preprocess_gait.py

# Step 3: Train baseline models (Random Forest, SVM, etc.)
python src/models/train_baseline.py

# Step 4: Test detection on a video
python detect_deepfake.py data/aditya.mp4
```

### 3. Advanced Deep Learning Pipeline (Optional but Better)

```bash
# Step 1-3: Same as above (if not done already)

# Step 4: Train neural networks (LSTM, CNN, Hybrid)
python src/models/deep_learning_models.py

# Step 5: Test with deep learning models
python detect_deepfake.py data/aditya.mp4

# Step 6: Generate comprehensive evaluation
python evaluation_and_visualization.py

# Step 7: See detailed performance charts
# Check evaluation_results/ folder for:
# - confusion_matrices/
# - roc_curves/
# - model_comparison.png
# - evaluation_report.txt
```

### 4. Production Use (Use Anytime)

```bash
# Single video detection
python detect_deepfake.py data/suspicious_video.mp4

# Batch process all videos
python detect_deepfake.py data/ --batch --output results.json

# Custom threshold (higher = stricter)
python detect_deepfake.py data/video.mp4 --threshold 0.8
```

## Complete Workflow Steps

### Phase 1: Data Preparation

1. **Add Videos**: Place MP4 files in `data/` folder
2. **Extract Poses**: `python src/preprocessing/extract_gait.py`
   - ✅ Creates `data/gait_keypoints.csv`
   - ✅ Processes all MP4 files automatically
3. **Engineer Features**: `python src/preprocessing/preprocess_gait.py`
   - ✅ Creates `data/processed/X.npy` (features)
   - ✅ Creates `data/processed/y.npy` (labels)
   - ✅ Normalizes and augments data

### Phase 2: Model Training

4. **Train Basic Models**: `python src/models/train_baseline.py`

   - ✅ RandomForest, SVM, Logistic Regression, etc.
   - ✅ Cross-validation and model selection
   - ✅ Saves best model to `models/`

5. **Train Deep Learning** (Optional): `python src/models/deep_learning_models.py`
   - ✅ LSTM (temporal gait patterns)
   - ✅ CNN (spatial pose features)
   - ✅ Hybrid CNN-LSTM model
   - ✅ Advanced metrics and validation

### Phase 3: Evaluation & Testing

6. **Comprehensive Analysis**: `python evaluation_and_visualization.py`

   - ✅ Confusion matrices for all models
   - ✅ ROC curves and precision-recall analysis
   - ✅ Model comparison charts
   - ✅ Detailed performance report

7. **Test Detection**: `python detect_deepfake.py data/test_video.mp4`
   - ✅ Real-time deepfake detection
   - ✅ Confidence scoring
   - ✅ Automatic best model selection

### Phase 4: Production Deployment

8. **Batch Processing**: `python detect_deepfake.py data/ --batch`
9. **Custom Analysis**: Adjust thresholds and parameters
10. **Visualization**: `python visualize_menu.py` for gait analysis

## What Happens After Each Step

### After Basic Training

```bash
# You can immediately test detection
python detect_deepfake.py data/aditya.mp4
# Expected: Authentic/Deepfake prediction with confidence
```

### After Deep Learning Training

```bash
# Test with potentially better models
python detect_deepfake.py data/aditya.mp4
# Expected: Higher accuracy, better confidence scores

# Generate comprehensive evaluation
python evaluation_and_visualization.py
# Creates: evaluation_results/ with detailed analysis
```

### After Evaluation

```bash
# Check results in evaluation_results/
ls evaluation_results/
# - model_comparison.png (performance chart)
# - combined_roc_curves.png (ROC analysis)
# - evaluation_report.txt (detailed metrics)
# - [ModelName]/ folders with individual analysis
```

## Expected Outputs

### Training Success

```
🎉 TRAINING COMPLETE!
✅ RandomForest: Accuracy = 0.892, F1 = 0.888
✅ LSTM: Accuracy = 0.914, F1 = 0.910
🏆 Best Model: LSTM saved to models/
```

### Detection Success

```
📊 ANALYSIS RESULTS:
   Prediction: Authentic
   Deepfake Probability: 0.176
   Confidence: 0.824
   Frames Analyzed: 127
   Model: LSTM
```

### Evaluation Success

```
📈 EVALUATION COMPLETE!
Results saved in: evaluation_results/
  - Individual model plots: evaluation_results/[model_name]/
  - Comparison plots: evaluation_results/model_comparison.png
  - Detailed report: evaluation_results/evaluation_report.txt

SUMMARY:
Best Model: LSTM (Accuracy: 0.9140)
```

## File Structure After Complete Pipeline

```
DeepFake-Detection/
├── data/
│   ├── *.mp4 (your input videos)
│   ├── gait_keypoints.csv (extracted poses)
│   └── processed/
│       ├── X.npy (ML features)
│       ├── y.npy (labels)
│       └── labels.json (label mapping)
├── models/ (all trained models)
│   ├── best_model_*.joblib
│   ├── *_lstm_*.h5 (deep learning models)
│   ├── metadata_*.json (model info)
│   └── scaler_*.joblib (normalization)
├── evaluation_results/ (performance analysis)
│   ├── model_comparison.png
│   ├── combined_roc_curves.png
│   ├── evaluation_report.txt
│   └── [ModelName]/ (individual analysis)
```

## Performance Optimization Tips

### For Better Accuracy

1. **More Data**: Add more diverse video samples
2. **Deep Learning**: Use neural networks for temporal patterns
3. **Feature Engineering**: Experiment with different gait features
4. **Data Quality**: Ensure clear, well-lit videos

### For Faster Processing

1. **Reduce Frames**: Modify `target_frames=32` in preprocessing
2. **Simpler Models**: Use RandomForest instead of deep learning
3. **Batch Processing**: Process multiple videos together
4. **GPU Acceleration**: Install `tensorflow-gpu` for neural networks

## Troubleshooting & Next Steps

### Common Issues

```bash
# "No module named 'tensorflow'"
pip install tensorflow

# "No models found"
python src/models/train_baseline.py

# "No pose landmarks detected"
# - Check video quality and lighting
# - Ensure person is clearly visible
# - Try different video files
```

### After Successful Setup

1. **Experiment**: Try different videos and thresholds
2. **Analyze**: Study the evaluation results for insights
3. **Optimize**: Adjust parameters based on your specific needs
4. **Deploy**: Integrate into your application or workflow

---

**Next Steps After This Guide:**

1. ✅ Train your models with the above steps
2. ✅ Test detection on your videos
3. ✅ Analyze performance with evaluation script
4. ✅ Fine-tune based on results
5. ✅ Deploy for production use

**Remember**: This detects deepfakes by analyzing how people walk/move, not their faces! 🚶‍♂️🔍

## Expected Output

### Training Output

```
✅ RandomForest: Accuracy = 0.892, F1 = 0.888
✅ SVM: Accuracy = 0.876, F1 = 0.875
✅ Best Model: RandomForest saved to models/
```

### Detection Output

```
📊 ANALYSIS RESULTS:
   Prediction: Deepfake
   Deepfake Probability: 0.824
   Confidence: 0.824
   Frames Analyzed: 127
   Model: RandomForest
```

## File Structure After Training

```
DeepFake-Detection/
├── data/
│   ├── *.mp4 (your videos)
│   ├── gait_keypoints.csv
│   └── processed/
│       ├── X.npy
│       ├── y.npy
│       └── labels.json
├── models/ (trained models)
├── evaluation_results/ (charts & reports)
```

## Troubleshooting

### "No module named 'X'"

```bash
pip install -r requirements.txt
```

### "No pose landmarks detected"

- Check video quality
- Ensure person is visible
- Try different video

### "No models found"

```bash
python src/models/train_baseline.py
```

### "No preprocessed data found"

```bash
python src/preprocessing/extract_gait.py
python src/preprocessing/preprocess_gait.py
```

## Performance Tips

- **Best Video Quality**: 480p+, good lighting, person clearly visible
- **Optimal Duration**: 2-10 seconds of walking/movement
- **Fast Training**: Use fewer cross-validation folds
- **Better Accuracy**: Run hyperparameter tuning

## What Each Script Does

| Script                            | Purpose                  | Input               | Output               |
| --------------------------------- | ------------------------ | ------------------- | -------------------- |
| `extract_gait.py`                 | Extract pose from videos | MP4 files           | CSV with keypoints   |
| `preprocess_gait.py`              | Create ML features       | CSV keypoints       | NumPy arrays         |
| `train_baseline.py`               | Train ML models          | NumPy arrays        | Trained models       |
| `deep_learning_models.py`         | Train neural networks    | NumPy arrays        | Deep learning models |
| `detect_deepfake.py`              | Detect deepfakes         | MP4 + trained model | Prediction           |
| `evaluation_and_visualization.py` | Analyze performance      | Models + data       | Charts & reports     |

## Quick Commands Cheat Sheet

```bash
# Complete first-time pipeline
python test_pipeline.py && \
python src/preprocessing/extract_gait.py && \
python src/preprocessing/preprocess_gait.py && \
python src/models/train_baseline.py

# Detect deepfake in video
python detect_deepfake.py data/test_video.mp4

# Batch process all videos
python detect_deepfake.py data/ --batch --output results.json

# Improve models
python src/models/hyperparameter_tuning.py

# Generate evaluation report
python evaluation_and_visualization.py

# Visualize gait patterns
python visualize_menu.py
```

## Success Indicators

✅ **Training Successful**: Models saved in `models/` folder  
✅ **Good Performance**: Accuracy > 0.85, F1-Score > 0.85  
✅ **Detection Working**: Clear Authentic/Deepfake predictions  
✅ **High Confidence**: Confidence scores > 0.7

## Need Help?

1. Check the full `README.md` for detailed explanations
2. Run `python test_pipeline.py` to diagnose issues
3. Ensure all dependencies are installed correctly
4. Verify video files are in MP4 format and good quality

---

**Remember**: This detects deepfakes by analyzing how people walk/move, not their faces! 🚶‍♂️🔍
