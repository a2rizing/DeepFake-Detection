import numpy as np
import json
import joblib
import os
from datetime import datetime
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (classification_report, confusion_matrix, 
                           accuracy_score, precision_score, recall_score, 
                           f1_score, roc_auc_score, roc_curve)
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

class GaitDeepfakeDetector:
    def __init__(self, data_path="data/processed"):
        self.data_path = data_path
        self.models = {}
        self.best_model = None
        self.best_score = 0
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        
        # Load data
        self.load_data()
        
    def load_data(self):
        """Load and prepare the dataset"""
        print("Loading dataset...")
        
        # Load processed gait data
        X = np.load(os.path.join(self.data_path, "X.npy"))  # (N_samples, N_frames, N_features)
        y = np.load(os.path.join(self.data_path, "y.npy"))
        
        with open(os.path.join(self.data_path, "labels.json")) as f:
            self.labels_json = json.load(f)
        
        print(f"Original data shape: {X.shape}")
        print(f"Labels: {list(self.labels_json.keys())}")
        
        # For deepfake detection, we need to create binary labels
        # For now, let's simulate this by treating each person as authentic (0) or deepfake (1)
        # In a real scenario, you'd have actual deepfake vs real labels
        self.create_binary_labels(X, y)
        
    def create_binary_labels(self, X, y):
        """Create binary deepfake detection labels"""
        print("\nCreating binary classification dataset...")
        
        # Flatten the temporal dimension for traditional ML models
        # X shape: (N_samples, N_frames, N_features) -> (N_samples, N_frames * N_features)
        X_flat = X.reshape(X.shape[0], -1)
        
        # For demonstration, let's create synthetic deepfake data
        # by adding noise to existing authentic data
        print("Generating synthetic deepfake samples...")
        
        authentic_X = X_flat.copy()
        authentic_y = np.zeros(len(authentic_X))  # 0 = authentic
        
        # Create deepfake samples by adding controlled noise
        deepfake_X = []
        for i in range(len(X_flat)):
            # Add gaussian noise to create "deepfake" versions
            noise_factor = 0.1
            noise = np.random.normal(0, noise_factor, X_flat[i].shape)
            deepfake_sample = X_flat[i] + noise
            deepfake_X.append(deepfake_sample)
        
        deepfake_X = np.array(deepfake_X)
        deepfake_y = np.ones(len(deepfake_X))  # 1 = deepfake
        
        # Combine authentic and deepfake data
        self.X = np.vstack([authentic_X, deepfake_X])
        self.y = np.hstack([authentic_y, deepfake_y])
        
        print(f"Final dataset shape: {self.X.shape}")
        print(f"Authentic samples: {np.sum(self.y == 0)}")
        print(f"Deepfake samples: {np.sum(self.y == 1)}")
        
        # Normalize features
        self.X = self.scaler.fit_transform(self.X)
        
    def train_baseline_models(self):
        """Train multiple baseline models"""
        print("\n" + "="*50)
        print("TRAINING BASELINE MODELS")
        print("="*50)
        
        # Define models to test
        models_config = {
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'SVM': SVC(probability=True, random_state=42),
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'K-Neighbors': KNeighborsClassifier(n_neighbors=5)
        }
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42, stratify=self.y
        )
        
        print(f"Training set size: {X_train.shape[0]}")
        print(f"Test set size: {X_test.shape[0]}")
        
        results = {}
        
        for name, model in models_config.items():
            print(f"\n--- Training {name} ---")
            
            # Train model
            model.fit(X_train, y_train)
            
            # Predictions
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            
            if y_pred_proba is not None:
                roc_auc = roc_auc_score(y_test, y_pred_proba)
            else:
                roc_auc = None
            
            # Cross-validation
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
            
            results[name] = {
                'model': model,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'roc_auc': roc_auc,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'y_pred': y_pred,
                'y_pred_proba': y_pred_proba,
                'y_test': y_test
            }
            
            print(f"Accuracy: {accuracy:.4f}")
            print(f"Precision: {precision:.4f}")
            print(f"Recall: {recall:.4f}")
            print(f"F1-Score: {f1:.4f}")
            if roc_auc:
                print(f"ROC-AUC: {roc_auc:.4f}")
            print(f"CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
            
            # Update best model
            if accuracy > self.best_score:
                self.best_score = accuracy
                self.best_model = model
                self.best_model_name = name
        
        self.models = results
        print(f"\n🏆 Best Model: {self.best_model_name} (Accuracy: {self.best_score:.4f})")
        
        return results
    
    def save_model(self, model_dir="models"):
        """Save the best trained model"""
        os.makedirs(model_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = os.path.join(model_dir, f"best_model_{timestamp}.joblib")
        scaler_path = os.path.join(model_dir, f"scaler_{timestamp}.joblib")
        
        # Save model and scaler
        joblib.dump(self.best_model, model_path)
        joblib.dump(self.scaler, scaler_path)
        
        # Save metadata
        metadata = {
            'model_type': self.best_model_name,
            'accuracy': self.best_score,
            'timestamp': timestamp,
            'feature_shape': self.X.shape[1],
            'labels': {'0': 'Authentic', '1': 'Deepfake'}
        }
        
        metadata_path = os.path.join(model_dir, f"metadata_{timestamp}.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"\n💾 Model saved:")
        print(f"  Model: {model_path}")
        print(f"  Scaler: {scaler_path}")
        print(f"  Metadata: {metadata_path}")
        
        return model_path, scaler_path, metadata_path

def main():
    # Initialize detector
    detector = GaitDeepfakeDetector()
    
    # Train models
    results = detector.train_baseline_models()
    
    # Save best model
    detector.save_model()
    
    print("\n" + "="*50)
    print("TRAINING COMPLETE!")
    print("="*50)
    
    return detector, results

if __name__ == "__main__":
    detector, results = main()
