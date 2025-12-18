#!/usr/bin/env python3
"""
Unified Training Script for Gait-based Deepfake Detection

Trains all 7 model architectures and saves the best performing ones.

Usage:
    python src/models/train_models.py
    python src/models/train_models.py --epochs 100 --batch_size 32
    python src/models/train_models.py --models LSTM BiLSTM CNN_LSTM
"""

import os
import sys
import json
import argparse
import numpy as np
from datetime import datetime

# Suppress TensorFlow warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.models.models_extended import GaitModelBuilder


class ModelTrainer:
    """Trains and evaluates multiple gait recognition models"""
    
    def __init__(self, data_dir="data/processed", models_dir="models", results_dir="results"):
        self.data_dir = data_dir
        self.models_dir = models_dir
        self.results_dir = results_dir
        
        os.makedirs(models_dir, exist_ok=True)
        os.makedirs(results_dir, exist_ok=True)
        
        # Set seeds
        tf.random.set_seed(42)
        np.random.seed(42)
        
        print("=" * 70)
        print("GAIT MODEL TRAINER")
        print("=" * 70)
        print(f"TensorFlow version: {tf.__version__}")
        print(f"GPU Available: {tf.config.list_physical_devices('GPU')}")
        print(f"Data directory: {data_dir}")
        print(f"Models directory: {models_dir}")
        print("=" * 70)
    
    def load_data(self):
        """Load preprocessed data"""
        X_path = os.path.join(self.data_dir, "X.npy")
        y_path = os.path.join(self.data_dir, "y.npy")
        labels_path = os.path.join(self.data_dir, "labels.json")
        
        if not os.path.exists(X_path) or not os.path.exists(y_path):
            print("❌ Preprocessed data not found!")
            print(f"   Expected: {X_path}")
            print(f"   Expected: {y_path}")
            print("\n   Run this first:")
            print("   python src/preprocessing/preprocess_gait.py")
            return None, None, None
        
        X = np.load(X_path)
        y = np.load(y_path)
        
        # Load labels mapping
        labels_map = {}
        if os.path.exists(labels_path):
            with open(labels_path) as f:
                labels_map = json.load(f)
        
        print(f"\n✅ Loaded data: X{X.shape}, y{y.shape}")
        print(f"   Unique labels: {np.unique(y)}")
        print(f"   Labels map: {labels_map}")
        
        return X, y, labels_map
    
    def prepare_data(self, X, y, test_size=0.2, val_size=0.2):
        """Prepare data for training with train/val/test splits"""
        
        # Get number of classes
        num_classes = len(np.unique(y))
        samples_per_class = len(X) / num_classes
        
        print(f"\n📊 Dataset Statistics:")
        print(f"   Total samples: {len(X)}")
        print(f"   Number of classes: {num_classes}")
        print(f"   Avg samples per class: {samples_per_class:.1f}")
        
        # For person verification, we need at least 2 classes
        if num_classes < 2:
            print("⚠️ Only one class found. Need at least 2 for classification.")
            return None
        
        # Encode labels
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        
        # Convert to one-hot
        y_onehot = keras.utils.to_categorical(y_encoded, num_classes=num_classes)
        
        # Check if we have enough samples per class for stratified split
        class_counts = np.bincount(y_encoded)
        min_samples = class_counts.min()
        
        if min_samples < 3:
            print(f"\n⚠️ WARNING: Small dataset detected!")
            print(f"   Minimum samples per class: {min_samples}")
            print(f"   Using simple holdout split (no stratification)")
            
            # Shuffle data
            indices = np.random.permutation(len(X))
            X_shuffled = X[indices]
            y_shuffled = y_onehot[indices]
            
            # Simple split: 60% train, 20% val, 20% test
            n_train = int(len(X) * 0.6)
            n_val = int(len(X) * 0.2)
            
            X_train = X_shuffled[:n_train]
            y_train = y_shuffled[:n_train]
            X_val = X_shuffled[n_train:n_train+n_val]
            y_val = y_shuffled[n_train:n_train+n_val]
            X_test = X_shuffled[n_train+n_val:]
            y_test = y_shuffled[n_train+n_val:]
        else:
            # Normal stratified split
            X_temp, X_test, y_temp, y_test = train_test_split(
                X, y_onehot, test_size=test_size, random_state=42, 
                stratify=y_encoded
            )
            
            y_temp_encoded = np.argmax(y_temp, axis=1)
            X_train, X_val, y_train, y_val = train_test_split(
                X_temp, y_temp, test_size=val_size/(1-test_size), random_state=42,
                stratify=y_temp_encoded
            )
        
        print(f"   Train samples: {len(X_train)}")
        print(f"   Validation samples: {len(X_val)}")
        print(f"   Test samples: {len(X_test)}")
        
        return {
            'X_train': X_train, 'y_train': y_train,
            'X_val': X_val, 'y_val': y_val,
            'X_test': X_test, 'y_test': y_test,
            'num_classes': num_classes,
            'label_encoder': le
        }
    
    def get_callbacks(self, model_name, patience=10):
        """Get training callbacks"""
        return [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=patience,
                restore_best_weights=True,
                verbose=1
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            ),
            keras.callbacks.ModelCheckpoint(
                filepath=os.path.join(self.models_dir, f"{model_name}_best.keras"),
                monitor='val_accuracy',
                save_best_only=True,
                verbose=0
            )
        ]
    
    def train_model(self, model, data, model_name, epochs=50, batch_size=32):
        """Train a single model"""
        print(f"\n{'='*20} Training {model_name} {'='*20}")
        print(f"Parameters: {model.count_params():,}")
        
        callbacks = self.get_callbacks(model_name)
        
        history = model.fit(
            data['X_train'], data['y_train'],
            validation_data=(data['X_val'], data['y_val']),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        return history
    
    def evaluate_model(self, model, data, model_name):
        """Evaluate a trained model"""
        print(f"\n📈 Evaluating {model_name}...")
        
        # Test set evaluation
        results = model.evaluate(data['X_test'], data['y_test'], verbose=0)
        
        # Get predictions
        y_pred_proba = model.predict(data['X_test'], verbose=0)
        y_pred = np.argmax(y_pred_proba, axis=1)
        y_true = np.argmax(data['y_test'], axis=1)
        
        metrics = {
            'loss': float(results[0]),
            'accuracy': float(results[1]),
            'precision': float(results[2]) if len(results) > 2 else 0.0,
            'recall': float(results[3]) if len(results) > 3 else 0.0
        }
        
        # Calculate F1
        if metrics['precision'] + metrics['recall'] > 0:
            metrics['f1'] = 2 * (metrics['precision'] * metrics['recall']) / \
                           (metrics['precision'] + metrics['recall'])
        else:
            metrics['f1'] = 0.0
        
        print(f"   Accuracy: {metrics['accuracy']:.4f}")
        print(f"   Precision: {metrics['precision']:.4f}")
        print(f"   Recall: {metrics['recall']:.4f}")
        print(f"   F1 Score: {metrics['f1']:.4f}")
        
        return metrics, y_pred, y_true, y_pred_proba
    
    def save_results(self, all_results, timestamp):
        """Save training results"""
        results_path = os.path.join(self.results_dir, f"training_results_{timestamp}.json")
        
        # Convert numpy types to Python types for JSON
        serializable = {}
        for model_name, result in all_results.items():
            serializable[model_name] = {
                'metrics': result['metrics'],
                'training_epochs': len(result['history'].history['loss']),
                'final_train_acc': float(result['history'].history['accuracy'][-1]),
                'final_val_acc': float(result['history'].history['val_accuracy'][-1])
            }
        
        with open(results_path, 'w') as f:
            json.dump(serializable, f, indent=2)
        
        print(f"\n📄 Results saved to: {results_path}")
        return results_path
    
    def train_all(self, models_to_train=None, epochs=50, batch_size=32):
        """Train all models"""
        
        # Load data
        X, y, labels_map = self.load_data()
        if X is None:
            return None
        
        # Prepare data
        data = self.prepare_data(X, y)
        if data is None:
            return None
        
        # Build models
        sequence_length = X.shape[1]
        num_features = X.shape[2]
        num_classes = data['num_classes']
        
        print(f"\n🔧 Building models...")
        print(f"   Sequence length: {sequence_length}")
        print(f"   Features per frame: {num_features}")
        print(f"   Number of classes: {num_classes}")
        
        builder = GaitModelBuilder(sequence_length, num_features, num_classes)
        all_models = builder.build_all_models()
        
        # Filter models if specified
        if models_to_train:
            all_models = {k: v for k, v in all_models.items() if k in models_to_train}
        
        print(f"\n📋 Models to train: {list(all_models.keys())}")
        
        # Train each model
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        all_results = {}
        
        for model_name, model in all_models.items():
            history = self.train_model(model, data, model_name, epochs, batch_size)
            metrics, y_pred, y_true, y_pred_proba = self.evaluate_model(model, data, model_name)
            
            # Save model
            model_path = os.path.join(self.models_dir, f"{model_name}_{timestamp}.keras")
            model.save(model_path)
            print(f"   💾 Saved: {model_path}")
            
            all_results[model_name] = {
                'history': history,
                'metrics': metrics,
                'y_pred': y_pred,
                'y_true': y_true,
                'y_pred_proba': y_pred_proba,
                'model_path': model_path
            }
        
        # Print summary
        print("\n" + "="*70)
        print("TRAINING SUMMARY")
        print("="*70)
        print(f"{'Model':<20} | {'Accuracy':>10} | {'Precision':>10} | {'Recall':>10} | {'F1':>10}")
        print("-"*70)
        
        best_model = None
        best_acc = 0
        
        for model_name, result in all_results.items():
            m = result['metrics']
            print(f"{model_name:<20} | {m['accuracy']:>10.4f} | {m['precision']:>10.4f} | "
                  f"{m['recall']:>10.4f} | {m['f1']:>10.4f}")
            
            if m['accuracy'] > best_acc:
                best_acc = m['accuracy']
                best_model = model_name
        
        print("-"*70)
        print(f"\n🏆 Best Model: {best_model} (Accuracy: {best_acc:.4f})")
        
        # Save results
        self.save_results(all_results, timestamp)
        
        # Save best model info
        best_info = {
            'model_name': best_model,
            'model_path': all_results[best_model]['model_path'],
            'accuracy': best_acc,
            'timestamp': timestamp,
            'num_classes': num_classes,
            'sequence_length': sequence_length,
            'num_features': num_features
        }
        
        best_info_path = os.path.join(self.models_dir, "best_model_info.json")
        with open(best_info_path, 'w') as f:
            json.dump(best_info, f, indent=2)
        print(f"📄 Best model info saved: {best_info_path}")
        
        return all_results


def main():
    parser = argparse.ArgumentParser(description="Train gait recognition models")
    parser.add_argument("--data_dir", type=str, default="data/processed",
                        help="Directory with preprocessed data")
    parser.add_argument("--models_dir", type=str, default="models",
                        help="Directory to save trained models")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Training batch size")
    parser.add_argument("--models", nargs='+', default=None,
                        help="Specific models to train (default: all)")
    
    args = parser.parse_args()
    
    trainer = ModelTrainer(
        data_dir=args.data_dir,
        models_dir=args.models_dir
    )
    
    results = trainer.train_all(
        models_to_train=args.models,
        epochs=args.epochs,
        batch_size=args.batch_size
    )
    
    if results:
        print("\n✅ Training complete!")
        print("\nNext steps:")
        print("  1. Evaluate models: python src/models/evaluate_models.py")
        print("  2. Test detection:  python detect.py <video_path>")


if __name__ == "__main__":
    main()
