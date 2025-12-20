#!/usr/bin/env python3
"""
Checkpoint 5 validation tests for supporting modules.

This script validates that Tasks 2, 3, and 4 are working correctly:
- GaitAugmentation module (Task 2)
- StratifiedCrossValidator module (Task 3)
- GaitMetricsEvaluator module (Task 4)

Requirements validated:
- 4.1, 4.2, 4.3, 4.4 (Augmentation)
- 5.1, 5.2, 5.3, 5.4 (Cross-validation)
- 7.1, 7.2, 7.3, 7.4, 7.5, 7.6, 7.7 (Metrics)
"""

import sys
import os
import tempfile
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Suppress TensorFlow warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class TestGaitAugmentation:
    """Tests for GaitAugmentation module."""
    
    def __init__(self):
        from src.preprocessing.gait_augmentation import GaitAugmentation
        self.augmenter = GaitAugmentation(random_state=42)
    
    def test_temporal_warp_shape_preservation(self):
        """Test that temporal_warp preserves sequence shape."""
        print("Testing temporal_warp shape preservation...")
        
        sequence = np.random.rand(64, 70)
        warped = self.augmenter.temporal_warp(sequence, warp_factor=0.2)
        
        assert warped.shape == sequence.shape, f"Shape mismatch: {warped.shape} vs {sequence.shape}"
        print("  ✓ temporal_warp preserves shape (64, 70)")
        return True
    
    def test_magnitude_scale_bounds(self):
        """Test that magnitude_scale keeps angles within bounds."""
        print("Testing magnitude_scale bounds...")
        
        sequence = np.random.rand(64, 70)
        original_angles = sequence[:, 66:70].copy()
        
        scaled = self.augmenter.magnitude_scale(sequence, scale_range=(0.9, 1.1))
        scaled_angles = scaled[:, 66:70]
        
        # Check shape preserved
        assert scaled.shape == sequence.shape
        
        # Check angles are scaled within range
        ratios = scaled_angles / (original_angles + 1e-8)
        assert np.all(ratios >= 0.89), "Some angles scaled below 0.9"
        assert np.all(ratios <= 1.11), "Some angles scaled above 1.1"
        
        print("  ✓ magnitude_scale keeps angles within 0.9-1.1x bounds")
        return True
    
    def test_gaussian_noise_shape_preservation(self):
        """Test that add_gaussian_noise preserves shape."""
        print("Testing add_gaussian_noise shape preservation...")
        
        sequence = np.random.rand(64, 70)
        noisy = self.augmenter.add_gaussian_noise(sequence, std=0.01)
        
        assert noisy.shape == sequence.shape
        
        # Check that noise was actually added (sequences should differ)
        assert not np.allclose(noisy, sequence), "No noise was added"
        
        print("  ✓ add_gaussian_noise preserves shape and adds noise")
        return True
    
    def test_mixup_shape_and_coefficient(self):
        """Test that mixup produces correct shape and coefficient."""
        print("Testing mixup shape and coefficient...")
        
        seq1 = np.random.rand(64, 70)
        seq2 = np.random.rand(64, 70)
        
        mixed, coef = self.augmenter.mixup(seq1, seq2, alpha=0.2)
        
        assert mixed.shape == seq1.shape, f"Shape mismatch: {mixed.shape}"
        assert len(coef) == 2, "Coefficient should have 2 elements"
        assert np.isclose(coef.sum(), 1.0), "Coefficients should sum to 1"
        assert coef[0] >= 0.5, "First coefficient should be >= 0.5"
        
        print("  ✓ mixup produces correct shape and valid coefficients")
        return True
    
    def test_augment_batch(self):
        """Test batch augmentation."""
        print("Testing augment_batch...")
        
        X = np.random.rand(30, 64, 70)
        y = np.random.randint(0, 13, 30)
        
        X_aug, y_aug = self.augmenter.augment_batch(X, y, augment_factor=2)
        
        # Should have original + 2x augmented
        expected_samples = 30 * 3  # original + 2 augmented copies
        assert X_aug.shape[0] == expected_samples, f"Expected {expected_samples} samples, got {X_aug.shape[0]}"
        assert y_aug.shape[0] == expected_samples
        assert X_aug.shape[1:] == (64, 70), "Feature shape should be preserved"
        
        print(f"  ✓ augment_batch produces {expected_samples} samples from 30 originals")
        return True
    
    def test_augment_single_combined(self):
        """Test combined augmentation on single sequence."""
        print("Testing augment_single combined augmentations...")
        
        sequence = np.random.rand(64, 70)
        
        augmented = self.augmenter.augment_single(
            sequence,
            apply_temporal_warp=True,
            apply_magnitude_scale=True,
            apply_noise=True
        )
        
        assert augmented.shape == sequence.shape
        assert not np.allclose(augmented, sequence), "Augmentation should modify sequence"
        
        print("  ✓ augment_single applies combined augmentations correctly")
        return True


class TestStratifiedCrossValidator:
    """Tests for StratifiedCrossValidator module."""
    
    def __init__(self):
        from src.models.cross_validation import StratifiedCrossValidator
        self.cv = StratifiedCrossValidator(n_splits=5, n_repeats=3, random_state=42)
    
    def test_split_generates_correct_folds(self):
        """Test that split generates correct number of folds."""
        print("Testing split fold generation...")
        
        X = np.random.rand(130, 64, 70)  # 10 samples per class
        y = np.repeat(np.arange(13), 10)  # 13 classes, 10 each
        
        # Non-repeated
        folds = list(self.cv.split(X, y, repeated=False))
        assert len(folds) == 5, f"Expected 5 folds, got {len(folds)}"
        
        # Repeated
        folds_repeated = list(self.cv.split(X, y, repeated=True))
        assert len(folds_repeated) == 15, f"Expected 15 folds (5*3), got {len(folds_repeated)}"
        
        print("  ✓ split generates correct number of folds (5 non-repeated, 15 repeated)")
        return True
    
    def test_stratification_maintains_proportions(self):
        """Test that stratification maintains class proportions."""
        print("Testing stratification class proportions...")
        
        X = np.random.rand(130, 64, 70)
        y = np.repeat(np.arange(13), 10)
        
        for fold_idx, (train_idx, val_idx) in enumerate(self.cv.split(X, y, repeated=False)):
            # Check all classes present in validation
            val_classes = np.unique(y[val_idx])
            assert len(val_classes) == 13, f"Fold {fold_idx}: Not all classes in validation"
            
            # Check proportions are roughly equal
            train_counts = np.bincount(y[train_idx], minlength=13)
            val_counts = np.bincount(y[val_idx], minlength=13)
            
            # Each class should have ~8 train and ~2 val samples
            assert np.all(train_counts >= 7), f"Fold {fold_idx}: Some classes have too few train samples"
            assert np.all(val_counts >= 1), f"Fold {fold_idx}: Some classes have no val samples"
        
        print("  ✓ Stratification maintains class proportions in all folds")
        return True
    
    def test_get_fold_info(self):
        """Test fold information retrieval."""
        print("Testing get_fold_info...")
        
        X = np.random.rand(130, 64, 70)
        y = np.repeat(np.arange(13), 10)
        
        for train_idx, val_idx in self.cv.split(X, y, repeated=False):
            info = self.cv.get_fold_info(y, train_idx, val_idx)
            
            assert 'train_size' in info
            assert 'val_size' in info
            assert 'num_classes_in_train' in info
            assert 'num_classes_in_val' in info
            assert info['train_size'] + info['val_size'] == 130
            break  # Just test first fold
        
        print("  ✓ get_fold_info returns complete fold statistics")
        return True
    
    def test_get_fold_statistics(self):
        """Test fold statistics computation."""
        print("Testing get_fold_statistics...")
        
        fold_metrics = [
            {'accuracy': 0.85, 'loss': 0.4},
            {'accuracy': 0.87, 'loss': 0.35},
            {'accuracy': 0.83, 'loss': 0.45},
            {'accuracy': 0.86, 'loss': 0.38},
            {'accuracy': 0.84, 'loss': 0.42},
        ]
        
        mean_metrics, std_metrics = self.cv.get_fold_statistics(fold_metrics)
        
        assert 'accuracy' in mean_metrics
        assert 'loss' in mean_metrics
        assert 'accuracy' in std_metrics
        assert 'loss' in std_metrics
        
        expected_mean_acc = np.mean([0.85, 0.87, 0.83, 0.86, 0.84])
        assert np.isclose(mean_metrics['accuracy'], expected_mean_acc)
        
        print(f"  ✓ get_fold_statistics computes mean={mean_metrics['accuracy']:.4f}, std={std_metrics['accuracy']:.4f}")
        return True
    
    def test_validate_stratification(self):
        """Test stratification validation."""
        print("Testing validate_stratification...")
        
        X = np.random.rand(130, 64, 70)
        y = np.repeat(np.arange(13), 10)
        
        validation = self.cv.validate_stratification(y, tolerance=0.15)
        
        assert 'all_classes_present' in validation
        assert 'proportions_within_tolerance' in validation
        assert 'max_proportion_deviation' in validation
        assert validation['num_classes'] == 13
        
        print(f"  ✓ validate_stratification: all_classes={validation['all_classes_present']}, within_tolerance={validation['proportions_within_tolerance']}")
        return True


class TestGaitMetricsEvaluator:
    """Tests for GaitMetricsEvaluator module."""
    
    def __init__(self):
        from src.models.metrics_evaluator import GaitMetricsEvaluator
        self.evaluator = GaitMetricsEvaluator(num_classes=13)
    
    def test_compute_all_metrics_completeness(self):
        """Test that compute_all_metrics returns all required metrics."""
        print("Testing compute_all_metrics completeness...")
        
        np.random.seed(42)
        y_true = np.random.randint(0, 13, 100)
        y_pred = y_true.copy()
        y_pred[:20] = np.random.randint(0, 13, 20)  # Add some errors
        y_proba = np.random.rand(100, 13)
        y_proba = y_proba / y_proba.sum(axis=1, keepdims=True)
        
        metrics = self.evaluator.compute_all_metrics(y_true, y_pred, y_proba)
        
        # Check all required metrics are present
        required_metrics = [
            'accuracy', 'precision_macro', 'precision_weighted',
            'recall_macro', 'recall_weighted', 'f1_macro', 'f1_weighted',
            'roc_auc', 'confusion_matrix', 'per_class'
        ]
        
        for metric in required_metrics:
            assert metric in metrics, f"Missing metric: {metric}"
        
        # Check confusion matrix is 13x13
        cm = np.array(metrics['confusion_matrix'])
        assert cm.shape == (13, 13), f"Confusion matrix shape: {cm.shape}"
        
        # Check per-class metrics
        assert len(metrics['per_class']) == 13
        
        print("  ✓ compute_all_metrics returns all required metrics")
        return True
    
    def test_compute_roc_auc(self):
        """Test ROC-AUC computation."""
        print("Testing compute_roc_auc...")
        
        np.random.seed(42)
        y_true = np.random.randint(0, 13, 100)
        y_proba = np.random.rand(100, 13)
        y_proba = y_proba / y_proba.sum(axis=1, keepdims=True)
        
        roc_auc = self.evaluator.compute_roc_auc(y_true, y_proba)
        
        assert 0.0 <= roc_auc <= 1.0, f"ROC-AUC out of bounds: {roc_auc}"
        
        print(f"  ✓ compute_roc_auc returns valid score: {roc_auc:.4f}")
        return True
    
    def test_generate_confusion_matrix(self):
        """Test confusion matrix generation."""
        print("Testing generate_confusion_matrix...")
        
        y_true = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2])
        y_pred = np.array([0, 1, 2, 0, 2, 1, 1, 1, 2])
        
        # Use 3-class evaluator for this test
        from src.models.metrics_evaluator import GaitMetricsEvaluator
        evaluator_3class = GaitMetricsEvaluator(num_classes=3)
        
        cm = evaluator_3class.generate_confusion_matrix(y_true, y_pred)
        
        assert cm.shape == (3, 3), f"Confusion matrix shape: {cm.shape}"
        assert cm.sum() == len(y_true), "Confusion matrix sum should equal sample count"
        
        print(f"  ✓ generate_confusion_matrix produces correct shape and sum")
        return True
    
    def test_save_and_load_metrics_json(self):
        """Test metrics JSON save and load."""
        print("Testing save_metrics_json and load_metrics_json...")
        
        metrics = {
            'accuracy': 0.85,
            'precision_macro': 0.84,
            'confusion_matrix': [[10, 2], [1, 12]],
            'per_class': {'Class_0': {'precision': 0.9, 'recall': 0.83, 'f1': 0.86}}
        }
        
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            temp_path = f.name
        
        try:
            self.evaluator.save_metrics_json(metrics, temp_path)
            loaded = self.evaluator.load_metrics_json(temp_path)
            
            assert loaded['accuracy'] == metrics['accuracy']
            assert loaded['confusion_matrix'] == metrics['confusion_matrix']
        finally:
            os.unlink(temp_path)
        
        print("  ✓ save_metrics_json and load_metrics_json work correctly")
        return True
    
    def test_per_class_metrics(self):
        """Test per-class metrics computation."""
        print("Testing per-class metrics...")
        
        np.random.seed(42)
        y_true = np.random.randint(0, 13, 100)
        y_pred = y_true.copy()
        
        metrics = self.evaluator.compute_all_metrics(y_true, y_pred)
        
        for class_name, class_metrics in metrics['per_class'].items():
            assert 'precision' in class_metrics
            assert 'recall' in class_metrics
            assert 'f1' in class_metrics
            assert 0.0 <= class_metrics['precision'] <= 1.0
            assert 0.0 <= class_metrics['recall'] <= 1.0
            assert 0.0 <= class_metrics['f1'] <= 1.0
        
        print("  ✓ Per-class metrics contain precision, recall, F1 for all classes")
        return True
    
    def test_visualization_methods_exist(self):
        """Test that visualization methods exist and are callable."""
        print("Testing visualization methods existence...")
        
        assert hasattr(self.evaluator, 'generate_roc_curves')
        assert hasattr(self.evaluator, 'generate_pr_curves')
        assert hasattr(self.evaluator, 'generate_confusion_matrix_plot')
        assert hasattr(self.evaluator, 'generate_training_plots')
        assert callable(self.evaluator.generate_roc_curves)
        assert callable(self.evaluator.generate_pr_curves)
        
        print("  ✓ All visualization methods exist and are callable")
        return True


def run_all_tests():
    """Run all checkpoint 5 validation tests."""
    print("=" * 60)
    print("CHECKPOINT 5: Supporting Modules Validation Tests")
    print("=" * 60)
    print()
    
    passed = 0
    failed = 0
    
    # GaitAugmentation tests
    print("-" * 40)
    print("GaitAugmentation Tests (Requirements 4.1-4.4)")
    print("-" * 40)
    
    try:
        aug_tests = TestGaitAugmentation()
        for test_method in [
            aug_tests.test_temporal_warp_shape_preservation,
            aug_tests.test_magnitude_scale_bounds,
            aug_tests.test_gaussian_noise_shape_preservation,
            aug_tests.test_mixup_shape_and_coefficient,
            aug_tests.test_augment_batch,
            aug_tests.test_augment_single_combined,
        ]:
            try:
                if test_method():
                    passed += 1
                else:
                    failed += 1
            except Exception as e:
                print(f"  ✗ {test_method.__name__} failed: {e}")
                failed += 1
            print()
    except Exception as e:
        print(f"  ✗ Failed to initialize GaitAugmentation tests: {e}")
        failed += 1
    
    # StratifiedCrossValidator tests
    print("-" * 40)
    print("StratifiedCrossValidator Tests (Requirements 5.1-5.4)")
    print("-" * 40)
    
    try:
        cv_tests = TestStratifiedCrossValidator()
        for test_method in [
            cv_tests.test_split_generates_correct_folds,
            cv_tests.test_stratification_maintains_proportions,
            cv_tests.test_get_fold_info,
            cv_tests.test_get_fold_statistics,
            cv_tests.test_validate_stratification,
        ]:
            try:
                if test_method():
                    passed += 1
                else:
                    failed += 1
            except Exception as e:
                print(f"  ✗ {test_method.__name__} failed: {e}")
                failed += 1
            print()
    except Exception as e:
        print(f"  ✗ Failed to initialize CrossValidator tests: {e}")
        failed += 1
    
    # GaitMetricsEvaluator tests
    print("-" * 40)
    print("GaitMetricsEvaluator Tests (Requirements 7.1-7.7)")
    print("-" * 40)
    
    try:
        metrics_tests = TestGaitMetricsEvaluator()
        for test_method in [
            metrics_tests.test_compute_all_metrics_completeness,
            metrics_tests.test_compute_roc_auc,
            metrics_tests.test_generate_confusion_matrix,
            metrics_tests.test_save_and_load_metrics_json,
            metrics_tests.test_per_class_metrics,
            metrics_tests.test_visualization_methods_exist,
        ]:
            try:
                if test_method():
                    passed += 1
                else:
                    failed += 1
            except Exception as e:
                print(f"  ✗ {test_method.__name__} failed: {e}")
                failed += 1
            print()
    except Exception as e:
        print(f"  ✗ Failed to initialize MetricsEvaluator tests: {e}")
        failed += 1
    
    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
