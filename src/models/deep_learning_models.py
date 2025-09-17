#!/usr/bin/env python3
"""
Deep Learning Models for Gait-based Deepfake Detection
Implements LSTM, CNN, and Hybrid models using TensorFlow/Keras
"""

import os
import json
import numpy as np

# Suppress TensorFlow warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import joblib
from datetime import datetime

class GaitDeepLearningModels:
    def __init__(self, models_dir="models"):
        """Initialize deep learning models for gait analysis"""
        self.models_dir = models_dir
        os.makedirs(models_dir, exist_ok=True)
        
        # Set random seeds for reproducibility
        tf.random.set_seed(42)
        np.random.seed(42)
        
        # Model configurations
        self.sequence_length = 64
        self.num_features = 70  # 66 coordinates + 4 angles
        self.num_classes = 2    # Authentic vs Deepfake
        
        print("TensorFlow version:", tf.__version__)
        print("GPU Available:", tf.config.list_physical_devices('GPU'))
    
    def prepare_data_for_deep_learning(self, X, y):
        """Prepare data for deep learning models"""
        print("Preparing data for deep learning...")
        
        # Check if X is already in the right shape (n_samples, sequence_length, num_features)
        if len(X.shape) == 3:
            n_samples, seq_len, n_features = X.shape
            print(f"Data already in sequence format: {X.shape}")
            
            # Update our expected dimensions to match the data
            self.sequence_length = seq_len
            self.num_features = n_features
            X_seq = X
            
        else:
            # X should be shape (n_samples, sequence_length * num_features)
            # Reshape to (n_samples, sequence_length, num_features)
            n_samples = X.shape[0]
            expected_size = self.sequence_length * self.num_features
            
            if X.shape[1] != expected_size:
                print(f"Warning: Expected {expected_size} features, got {X.shape[1]}")
                # Pad or truncate if necessary
                if X.shape[1] < expected_size:
                    padding = np.zeros((n_samples, expected_size - X.shape[1]))
                    X = np.concatenate([X, padding], axis=1)
                else:
                    X = X[:, :expected_size]
            
            # Reshape to sequence format
            X_seq = X.reshape(n_samples, self.sequence_length, self.num_features)
        
        # Convert labels to categorical if needed
        # Check if we have person labels instead of binary labels
        unique_labels = np.unique(y)
        if len(unique_labels) > 2:
            print(f"Found {len(unique_labels)} unique labels: {unique_labels}")
            print("Converting to binary classification: person 0 = authentic, others = deepfake")
            # Convert to binary: person 0 is authentic, others are deepfake
            y_binary = (y > 0).astype(int)
            print(f"Binary labels: {np.unique(y_binary)} (0=Authentic, 1=Deepfake)")
        else:
            y_binary = y
        
        if len(y_binary.shape) == 1:
            y_cat = tf.keras.utils.to_categorical(y_binary, num_classes=self.num_classes)
        else:
            y_cat = y_binary
        
        print(f"Data shape: {X_seq.shape}")
        print(f"Labels shape: {y_cat.shape}")
        
        return X_seq, y_cat
    
    def create_lstm_model(self, lstm_units=[64, 32], dropout_rate=0.3):
        """Create LSTM model for temporal gait analysis"""
        model = keras.Sequential([
            layers.Input(shape=(self.sequence_length, self.num_features)),
            
            # First LSTM layer
            layers.LSTM(lstm_units[0], return_sequences=True, dropout=dropout_rate),
            layers.BatchNormalization(),
            
            # Second LSTM layer
            layers.LSTM(lstm_units[1], dropout=dropout_rate),
            layers.BatchNormalization(),
            
            # Dense layers
            layers.Dense(32, activation='relu'),
            layers.Dropout(dropout_rate),
            layers.Dense(16, activation='relu'),
            layers.Dropout(dropout_rate),
            
            # Output layer
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        return model
    
    def create_cnn_model(self, conv_filters=[32, 64, 128], kernel_size=3, dropout_rate=0.3):
        """Create 1D CNN model for gait pattern recognition"""
        model = keras.Sequential([
            layers.Input(shape=(self.sequence_length, self.num_features)),
            
            # First conv block
            layers.Conv1D(conv_filters[0], kernel_size, activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling1D(2),
            layers.Dropout(dropout_rate),
            
            # Second conv block
            layers.Conv1D(conv_filters[1], kernel_size, activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling1D(2),
            layers.Dropout(dropout_rate),
            
            # Third conv block
            layers.Conv1D(conv_filters[2], kernel_size, activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.GlobalAveragePooling1D(),
            layers.Dropout(dropout_rate),
            
            # Dense layers
            layers.Dense(64, activation='relu'),
            layers.Dropout(dropout_rate),
            layers.Dense(32, activation='relu'),
            layers.Dropout(dropout_rate),
            
            # Output layer
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        return model
    
    def create_hybrid_model(self, lstm_units=64, conv_filters=64, dropout_rate=0.3):
        """Create hybrid CNN-LSTM model"""
        model = keras.Sequential([
            layers.Input(shape=(self.sequence_length, self.num_features)),
            
            # CNN feature extraction
            layers.Conv1D(conv_filters, 3, activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling1D(2),
            layers.Dropout(dropout_rate),
            
            layers.Conv1D(conv_filters*2, 3, activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Dropout(dropout_rate),
            
            # LSTM temporal modeling
            layers.LSTM(lstm_units, return_sequences=True, dropout=dropout_rate),
            layers.LSTM(lstm_units//2, dropout=dropout_rate),
            layers.BatchNormalization(),
            
            # Dense layers
            layers.Dense(32, activation='relu'),
            layers.Dropout(dropout_rate),
            layers.Dense(16, activation='relu'),
            layers.Dropout(dropout_rate),
            
            # Output layer
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        return model
    
    def train_model(self, model, X_train, y_train, X_val, y_val, model_name="model", 
                   epochs=50, batch_size=32, patience=10):
        """Train a deep learning model with early stopping"""
        print(f"\nTraining {model_name}...")
        print(f"Train samples: {X_train.shape[0]}, Val samples: {X_val.shape[0]}")
        
        # Callbacks
        callbacks = [
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
            )
        ]
        
        # Train model
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        return history
    
    def evaluate_model(self, model, X_test, y_test, model_name="model"):
        """Evaluate model performance"""
        print(f"\nEvaluating {model_name}...")
        
        # Predictions
        y_pred_proba = model.predict(X_test)
        y_pred = np.argmax(y_pred_proba, axis=1)
        y_true = np.argmax(y_test, axis=1)
        
        # Metrics
        test_loss, test_acc, test_prec, test_rec = model.evaluate(X_test, y_test, verbose=0)
        
        # Additional metrics
        try:
            auc_score = roc_auc_score(y_true, y_pred_proba[:, 1])
        except:
            auc_score = 0.0
        
        f1_score = 2 * (test_prec * test_rec) / (test_prec + test_rec + 1e-8)
        
        metrics = {
            'test_loss': float(test_loss),
            'test_accuracy': float(test_acc),
            'test_precision': float(test_prec),
            'test_recall': float(test_rec),
            'test_f1': float(f1_score),
            'test_auc': float(auc_score)
        }
        
        print(f"Test Accuracy: {test_acc:.4f}")
        print(f"Test Precision: {test_prec:.4f}")
        print(f"Test Recall: {test_rec:.4f}")
        print(f"Test F1: {f1_score:.4f}")
        print(f"Test AUC: {auc_score:.4f}")
        
        # Classification report
        print("\nClassification Report:")
        try:
            unique_classes = np.unique(y_true)
            if len(unique_classes) > 1:
                print(classification_report(y_true, y_pred, target_names=['Authentic', 'Deepfake']))
            else:
                print(f"⚠️  Only one class ({unique_classes[0]}) in test set - classification report not meaningful")
                print("This is expected with very small datasets.")
        except Exception as e:
            print(f"⚠️  Could not generate classification report: {e}")
        
        return metrics, y_pred, y_pred_proba
    
    def save_model(self, model, metrics, model_name, scaler=None):
        """Save trained model and metadata"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save model
        model_path = os.path.join(self.models_dir, f"{model_name}_{timestamp}.h5")
        model.save(model_path)
        
        # Save scaler if provided
        scaler_path = None
        if scaler is not None:
            scaler_path = os.path.join(self.models_dir, f"scaler_{model_name}_{timestamp}.joblib")
            joblib.dump(scaler, scaler_path)
        
        # Save metadata
        metadata = {
            'model_type': model_name,
            'model_path': model_path,
            'scaler_path': scaler_path,
            'timestamp': timestamp,
            'metrics': metrics,
            'sequence_length': self.sequence_length,
            'num_features': self.num_features,
            'tensorflow_version': tf.__version__
        }
        
        metadata_path = os.path.join(self.models_dir, f"{model_name}_metadata_{timestamp}.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Model saved: {model_path}")
        print(f"Metadata saved: {metadata_path}")
        
        return metadata
    
    def train_all_deep_models(self, X, y, test_size=0.2, val_size=0.2):
        """Train all deep learning models"""
        print("=" * 60)
        print("TRAINING DEEP LEARNING MODELS")
        print("=" * 60)
        
        # Prepare data
        X_seq, y_cat = self.prepare_data_for_deep_learning(X, y)
        
        # Check if we have enough data for deep learning
        n_samples = X_seq.shape[0]
        if n_samples < 10:
            print(f"⚠️  WARNING: Only {n_samples} samples available!")
            print("Deep learning works best with 100+ samples.")
            print("For demonstration, we'll use simple train/test split without validation.")
            
            # Simple split for small datasets
            if n_samples < 4:
                print("❌ Not enough data for training. Need at least 4 samples.")
                return None
            
            # Use simple split
            split_idx = max(1, n_samples // 2)
            X_train = X_seq[:split_idx]
            X_test = X_seq[split_idx:]
            y_train = y_cat[:split_idx]
            y_test = y_cat[split_idx:]
            X_val = X_test  # Use test as validation for small datasets
            y_val = y_test
            
            print(f"Train: {X_train.shape[0]}, Test: {X_test.shape[0]} (using test as validation)")
        else:
            # Normal split for larger datasets
            X_temp, X_test, y_temp, y_test = train_test_split(
                X_seq, y_cat, test_size=test_size, random_state=42, stratify=np.argmax(y_cat, axis=1)
            )
            
            X_train, X_val, y_train, y_val = train_test_split(
                X_temp, y_temp, test_size=val_size/(1-test_size), random_state=42, 
                stratify=np.argmax(y_temp, axis=1)
            )
            
            print(f"Train: {X_train.shape[0]}, Val: {X_val.shape[0]}, Test: {X_test.shape[0]}")
        
        # Models to train
        models_config = [
            ('LSTM', self.create_lstm_model),
            ('CNN', self.create_cnn_model),
            ('Hybrid', self.create_hybrid_model)
        ]
        
        results = {}
        
        for model_name, model_creator in models_config:
            print(f"\n{'-'*20} {model_name} {'-'*20}")
            
            # Create model
            model = model_creator()
            print(f"Model architecture created: {model.count_params()} parameters")
            
            # Train model
            history = self.train_model(
                model, X_train, y_train, X_val, y_val, 
                model_name=model_name, epochs=50, batch_size=16
            )
            
            # Evaluate model
            metrics, y_pred, y_pred_proba = self.evaluate_model(
                model, X_test, y_test, model_name=model_name
            )
            
            # Save model
            model_metadata = self.save_model(model, metrics, model_name)
            
            # Store results
            results[model_name] = {
                'metrics': metrics,
                'metadata': model_metadata,
                'history': {
                    'loss': history.history['loss'],
                    'val_loss': history.history['val_loss'],
                    'accuracy': history.history['accuracy'],
                    'val_accuracy': history.history['val_accuracy']
                }
            }
        
        # Find best model
        best_model = max(results.keys(), key=lambda x: results[x]['metrics']['test_accuracy'])
        best_accuracy = results[best_model]['metrics']['test_accuracy']
        
        print(f"\n{'='*60}")
        print("DEEP LEARNING RESULTS SUMMARY")
        print(f"{'='*60}")
        
        for model_name, result in results.items():
            acc = result['metrics']['test_accuracy']
            f1 = result['metrics']['test_f1']
            auc = result['metrics']['test_auc']
            print(f"{model_name:10} | Acc: {acc:.4f} | F1: {f1:.4f} | AUC: {auc:.4f}")
        
        print(f"\nBest Deep Learning Model: {best_model} (Accuracy: {best_accuracy:.4f})")
        
        # Save best model info
        best_metadata_path = os.path.join(self.models_dir, "best_deep_model_metadata.json")
        with open(best_metadata_path, 'w') as f:
            json.dump(results[best_model]['metadata'], f, indent=2)
        
        print(f"Best deep model metadata saved: {best_metadata_path}")
        
        return results

def main():
    """Main function to train deep learning models"""
    # Load preprocessed data
    data_dir = "data/processed"
    X_path = os.path.join(data_dir, "X.npy")
    y_path = os.path.join(data_dir, "y.npy")
    
    if not os.path.exists(X_path) or not os.path.exists(y_path):
        print("Error: Preprocessed data not found!")
        print("Please run preprocess_gait.py first to generate X.npy and y.npy")
        return
    
    # Load data
    X = np.load(X_path)
    y = np.load(y_path)
    
    print(f"Loaded data: X{X.shape}, y{y.shape}")
    
    # Initialize and train models
    dl_models = GaitDeepLearningModels()
    results = dl_models.train_all_deep_models(X, y)
    
    print("\nDeep learning training completed!")
    print("You can now use the trained models for deepfake detection.")

if __name__ == "__main__":
    main()