# Implementation Plan: Gait ML Training Pipeline

## Overview

This implementation plan breaks down the gait ML training pipeline into discrete coding tasks. The pipeline will train multiple deep learning models on the videos_augmented dataset to achieve >95% accuracy without overfitting.

## Tasks

- [x] 1. Set up project structure and verify existing data
  - [x] 1.1 Verify videos_augmented dataset structure and count samples per person
    - Check all 13 persons have augmented videos
    - Count total samples available
    - _Requirements: 1.1_
  - [x] 1.2 Verify existing preprocessed data in data/processed_augmented
    - Check X.npy shape is (N, 64, 70)
    - Check y.npy has 13 classes
    - Verify labels.json mapping
    - _Requirements: 1.5_

- [x] 2. Implement data augmentation module
  - [x] 2.1 Create src/preprocessing/gait_augmentation.py with GaitAugmentation class
    - Implement temporal_warp() for non-uniform speed variations
    - Implement magnitude_scale() for joint angle scaling (0.9-1.1)
    - Implement add_gaussian_noise() for coordinate perturbation
    - Implement mixup() for sequence interpolation
    - _Requirements: 4.1, 4.2, 4.3, 4.4_
  - [ ]* 2.2 Write property tests for augmentation module
    - **Property 5: Augmentation Validity**
    - **Validates: Requirements 4.1, 4.2, 4.3, 4.4**

- [x] 3. Implement cross-validation module
  - [x] 3.1 Create src/models/cross_validation.py with StratifiedCrossValidator class
    - Implement stratified 5-fold split maintaining class proportions
    - Implement repeated k-fold (3 repetitions)
    - Implement cross_validate() method for full CV training
    - Implement get_fold_statistics() for mean/std computation
    - _Requirements: 5.1, 5.2, 5.3, 5.4_
  - [ ]* 3.2 Write property tests for cross-validation
    - **Property 6: Cross-Validation Stratification**
    - **Validates: Requirements 5.1, 5.4**

- [x] 4. Implement comprehensive metrics module
  - [x] 4.1 Create src/models/metrics_evaluator.py with GaitMetricsEvaluator class
    - Implement compute_all_metrics() for accuracy, precision, recall, F1
    - Implement compute_roc_auc() using one-vs-rest strategy
    - Implement generate_confusion_matrix() for 13x13 matrix
    - Implement per-class metrics computation
    - _Requirements: 7.1, 7.2, 7.3, 7.4_
  - [x] 4.2 Implement visualization methods
    - Implement generate_roc_curves() for all classes
    - Implement generate_pr_curves() for precision-recall
    - Implement generate_training_plots() for loss/accuracy
    - Implement save_metrics_json() for reproducibility
    - _Requirements: 7.5, 7.6, 7.7_
  - [ ]* 4.3 Write property tests for metrics module
    - **Property 8: Metrics Completeness**
    - **Validates: Requirements 7.1, 7.2, 7.3, 7.4**

- [x] 5. Checkpoint - Verify supporting modules
  - Ensure augmentation, cross-validation, and metrics modules work correctly
  - Run any existing tests
  - Ask user if questions arise

- [x] 6. Enhance model architectures
  - [x] 6.1 Update src/models/models_enhanced.py with improved regularization
    - Ensure all models use AdamW optimizer with weight decay
    - Ensure label smoothing (epsilon=0.1) in loss function
    - Verify L2 regularization coefficient (0.0001-0.001)
    - Verify dropout rates (0.2-0.3 conv, 0.3-0.5 dense)
    - _Requirements: 2.8, 3.1, 3.2, 3.3, 3.6_
  - [x] 6.2 Add GRU and TCN architectures to model builder
    - Implement build_enhanced_gru() with similar regularization
    - Implement build_tcn() for temporal convolutional network
    - _Requirements: 2.1-2.7_
  - [ ]* 6.3 Write property tests for model configuration
    - **Property 3: Model Compilation Configuration**
    - **Property 4: Regularization Bounds**
    - **Validates: Requirements 2.8, 2.9, 3.2, 3.3**

- [x] 7. Implement ensemble module
  - [x] 7.1 Create src/models/ensemble.py with GaitEnsemble class
    - Implement soft_vote() for probability averaging
    - Implement weighted_vote() based on validation performance
    - Implement predict() returning ensemble and individual predictions
    - _Requirements: 6.1, 6.2, 6.4_
  - [x] 7.2 Implement model stacking
    - Implement fit_stacking() with logistic regression meta-learner
    - _Requirements: 6.5_
  - [ ]* 7.3 Write property tests for ensemble
    - **Property 7: Ensemble Prediction Consistency**
    - **Validates: Requirements 6.1, 6.4**

- [x] 8. Implement comprehensive training script
  - [x] 8.1 Create src/models/train_comprehensive.py
    - Implement training with all callbacks (early stopping, LR reduction)
    - Implement cross-validation training loop
    - Implement model saving with metadata
    - Implement TensorBoard logging
    - _Requirements: 3.4, 3.5, 3.7, 12.1, 12.2, 12.5_
  - [x] 8.2 Add hyperparameter configuration
    - Support configurable learning rates, batch sizes, dropout rates
    - Support configurable L2 regularization coefficients
    - _Requirements: 8.1, 8.2, 8.3, 8.4_

- [x] 9. Checkpoint - Verify training pipeline
  - Run training on small subset to verify pipeline works
  - Check model saving and loading
  - Verify metrics are computed correctly
  - Ask user if questions arise

- [x] 10. Implement hyperparameter tuning
  - [x] 10.1 Create src/models/hyperparameter_tuning.py
    - Implement grid search over learning rates (0.0001, 0.0005, 0.001)
    - Implement grid search over batch sizes (32, 64)
    - Implement grid search over dropout rates (0.3, 0.4, 0.5)
    - Use Keras Tuner for efficient search
    - _Requirements: 8.1, 8.2, 8.3, 8.4, 8.5, 8.6_

- [x] 11. Implement GradCAM visualization
  - [x] 11.1 Update src/visualization/gradcam_gait.py
    - Implement compute_gradcam() for temporal importance
    - Implement visualize_temporal_importance() for plotting
    - Implement get_attention_weights() for Transformer models
    - _Requirements: 11.1, 11.2, 11.3, 11.4_

- [x] 12. Implement testing and validation script
  - [x] 12.1 Create src/models/test_models.py
    - Implement testing on videos_augmented with per-person accuracy
    - Implement testing on data/deepfake videos
    - Implement confidence score output
    - Generate detailed test report
    - _Requirements: 9.1, 9.2, 9.3, 9.4, 9.5_

- [x] 13. Train all models with cross-validation
  - [x] 13.1 Run full training pipeline
    - Train all 7+ architectures with 5-fold CV, 3 repetitions
    - Save all models and metrics
    - Generate comparison report
    - _Requirements: 2.1-2.9, 5.1-5.4_

- [x] 14. Build and evaluate ensemble
  - [x] 14.1 Create ensemble from best 3 models
    - Select top 3 models by validation accuracy
    - Build soft-voting ensemble
    - Build weighted ensemble
    - Evaluate ensemble performance
    - _Requirements: 6.1, 6.2, 6.3_

- [x] 15. Final checkpoint - Complete evaluation
  - Run full evaluation on test set
  - Generate all visualizations (ROC, PR curves, confusion matrix)
  - Generate GradCAM visualizations
  - Test on deepfake videos
  - Compile final metrics report
  - Ask user if questions arise

## Notes

- Tasks marked with `*` are optional property-based tests
- Each task references specific requirements for traceability
- Checkpoints ensure incremental validation
- Property tests validate universal correctness properties
- Use Python with TensorFlow/Keras for implementation
- Use Hypothesis library for property-based testing
