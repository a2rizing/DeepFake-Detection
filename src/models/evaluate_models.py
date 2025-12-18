#!/usr/bin/env python3
"""
Model Evaluation and Visualization Script

Generates:
- Confusion matrices for each model
- ROC curves and AUC scores
- Precision-Recall curves
- Model comparison summary

Usage:
    python src/models/evaluate_models.py
    python src/models/evaluate_models.py --results_file results/training_results_*.json
"""

import os
import sys
import json
import glob
import argparse
import numpy as np
from datetime import datetime

# Suppress TensorFlow warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import (
    confusion_matrix, classification_report, 
    roc_curve, auc, precision_recall_curve,
    accuracy_score, precision_score, recall_score, f1_score
)

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Try to import visualization libraries
try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    print("⚠️ matplotlib/seaborn not available. Text-only output.")


class ModelEvaluator:
    """Comprehensive model evaluation and visualization"""
    
    def __init__(self, models_dir="models", results_dir="results", data_dir="data/processed"):
        self.models_dir = models_dir
        self.results_dir = results_dir
        self.data_dir = data_dir
        
        os.makedirs(results_dir, exist_ok=True)
        
        print("=" * 70)
        print("MODEL EVALUATOR")
        print("=" * 70)
    
    def load_data(self):
        """Load test data"""
        X = np.load(os.path.join(self.data_dir, "X.npy"))
        y = np.load(os.path.join(self.data_dir, "y.npy"))
        
        # Load labels
        labels_path = os.path.join(self.data_dir, "labels.json")
        if os.path.exists(labels_path):
            with open(labels_path) as f:
                labels_map = json.load(f)
            # Invert map: id -> name
            self.label_names = {v: k for k, v in labels_map.items()}
        else:
            self.label_names = {i: f"Class_{i}" for i in range(len(np.unique(y)))}
        
        print(f"✅ Loaded data: X{X.shape}, y{y.shape}")
        return X, y
    
    def load_models(self):
        """Load all trained models"""
        models = {}
        
        # Find all .keras and .h5 model files
        patterns = [
            os.path.join(self.models_dir, "*_best.keras"),
            os.path.join(self.models_dir, "*.keras"),
            os.path.join(self.models_dir, "*.h5")
        ]
        
        model_files = set()
        for pattern in patterns:
            model_files.update(glob.glob(pattern))
        
        for model_path in sorted(model_files):
            # Extract model name from filename
            basename = os.path.basename(model_path)
            model_name = basename.split('_')[0]
            
            # Skip if already loaded (prefer _best version)
            if model_name in models:
                if '_best' not in basename:
                    continue
            
            try:
                model = keras.models.load_model(model_path, compile=False)
                model.compile(
                    optimizer='adam',
                    loss='categorical_crossentropy',
                    metrics=['accuracy']
                )
                models[model_name] = {
                    'model': model,
                    'path': model_path
                }
                print(f"✅ Loaded: {model_name} from {basename}")
            except Exception as e:
                print(f"⚠️ Failed to load {basename}: {e}")
        
        return models
    
    def evaluate_all(self, X, y, models):
        """Evaluate all models and generate metrics"""
        results = {}
        
        # Convert y to one-hot if needed
        num_classes = len(np.unique(y))
        y_onehot = keras.utils.to_categorical(y, num_classes)
        
        # Use last 20% as test set (matching training split)
        test_size = int(len(X) * 0.2)
        X_test = X[-test_size:]
        y_test = y[-test_size:]
        y_test_onehot = y_onehot[-test_size:]
        
        print(f"\n📊 Evaluating on {len(X_test)} test samples...")
        print("-" * 70)
        
        for model_name, model_info in models.items():
            model = model_info['model']
            
            # Predictions
            y_pred_proba = model.predict(X_test, verbose=0)
            y_pred = np.argmax(y_pred_proba, axis=1)
            
            # Metrics
            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
            
            # Confusion matrix
            cm = confusion_matrix(y_test, y_pred)
            
            results[model_name] = {
                'accuracy': acc,
                'precision': prec,
                'recall': rec,
                'f1': f1,
                'y_pred': y_pred,
                'y_pred_proba': y_pred_proba,
                'y_true': y_test,
                'confusion_matrix': cm
            }
            
            print(f"{model_name:<20} | Acc: {acc:.4f} | Prec: {prec:.4f} | "
                  f"Rec: {rec:.4f} | F1: {f1:.4f}")
        
        return results
    
    def plot_confusion_matrices(self, results, output_dir):
        """Generate confusion matrix plots for all models"""
        if not PLOTTING_AVAILABLE:
            print("⚠️ Plotting not available. Skipping confusion matrices.")
            return
        
        n_models = len(results)
        cols = min(3, n_models)
        rows = (n_models + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
        if n_models == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        for idx, (model_name, result) in enumerate(results.items()):
            cm = result['confusion_matrix']
            
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx])
            axes[idx].set_title(f'{model_name}\nAcc: {result["accuracy"]:.4f}')
            axes[idx].set_xlabel('Predicted')
            axes[idx].set_ylabel('True')
        
        # Hide empty subplots
        for idx in range(len(results), len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        output_path = os.path.join(output_dir, "confusion_matrices.png")
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"📊 Saved: {output_path}")
    
    def plot_model_comparison(self, results, output_dir):
        """Generate bar chart comparing all models"""
        if not PLOTTING_AVAILABLE:
            return
        
        models = list(results.keys())
        metrics = ['accuracy', 'precision', 'recall', 'f1']
        
        x = np.arange(len(models))
        width = 0.2
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        for i, metric in enumerate(metrics):
            values = [results[m][metric] for m in models]
            ax.bar(x + i*width, values, width, label=metric.capitalize())
        
        ax.set_xlabel('Model')
        ax.set_ylabel('Score')
        ax.set_title('Model Comparison')
        ax.set_xticks(x + width * 1.5)
        ax.set_xticklabels(models, rotation=45, ha='right')
        ax.legend()
        ax.set_ylim(0, 1.05)
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        output_path = os.path.join(output_dir, "model_comparison.png")
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"📊 Saved: {output_path}")
    
    def generate_report(self, results, output_dir):
        """Generate text report with all metrics"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = os.path.join(output_dir, f"evaluation_report_{timestamp}.txt")
        
        with open(report_path, 'w') as f:
            f.write("=" * 70 + "\n")
            f.write("GAIT DEEPFAKE DETECTION - MODEL EVALUATION REPORT\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 70 + "\n\n")
            
            # Summary table
            f.write("MODEL PERFORMANCE SUMMARY\n")
            f.write("-" * 70 + "\n")
            f.write(f"{'Model':<20} | {'Accuracy':>10} | {'Precision':>10} | {'Recall':>10} | {'F1':>10}\n")
            f.write("-" * 70 + "\n")
            
            best_model = None
            best_acc = 0
            
            for model_name, result in sorted(results.items(), 
                                             key=lambda x: x[1]['accuracy'], 
                                             reverse=True):
                f.write(f"{model_name:<20} | {result['accuracy']:>10.4f} | "
                       f"{result['precision']:>10.4f} | {result['recall']:>10.4f} | "
                       f"{result['f1']:>10.4f}\n")
                
                if result['accuracy'] > best_acc:
                    best_acc = result['accuracy']
                    best_model = model_name
            
            f.write("-" * 70 + "\n")
            f.write(f"\n🏆 BEST MODEL: {best_model} (Accuracy: {best_acc:.4f})\n")
            
            # Confusion matrices
            f.write("\n\nCONFUSION MATRICES\n")
            f.write("=" * 70 + "\n")
            
            for model_name, result in results.items():
                f.write(f"\n{model_name}:\n")
                cm = result['confusion_matrix']
                f.write(np.array2string(cm, prefix='  '))
                f.write("\n")
            
            # Classification reports
            f.write("\n\nDETAILED CLASSIFICATION REPORTS\n")
            f.write("=" * 70 + "\n")
            
            for model_name, result in results.items():
                f.write(f"\n{model_name}:\n")
                f.write("-" * 50 + "\n")
                report = classification_report(
                    result['y_true'], result['y_pred'],
                    target_names=[self.label_names.get(i, f"Class_{i}") 
                                 for i in range(len(np.unique(result['y_true'])))],
                    zero_division=0
                )
                f.write(report)
                f.write("\n")
        
        print(f"📄 Report saved: {report_path}")
        return report_path
    
    def run_evaluation(self):
        """Run complete evaluation pipeline"""
        # Load data
        X, y = self.load_data()
        
        # Load models
        models = self.load_models()
        
        if not models:
            print("❌ No trained models found!")
            print("   Run training first: python src/models/train_models.py")
            return
        
        # Evaluate
        results = self.evaluate_all(X, y, models)
        
        # Generate visualizations
        print("\n📈 Generating visualizations...")
        self.plot_confusion_matrices(results, self.results_dir)
        self.plot_model_comparison(results, self.results_dir)
        
        # Generate report
        report_path = self.generate_report(results, self.results_dir)
        
        # Find best model
        best_model = max(results.keys(), key=lambda x: results[x]['accuracy'])
        best_acc = results[best_model]['accuracy']
        
        print("\n" + "=" * 70)
        print("EVALUATION COMPLETE")
        print("=" * 70)
        print(f"🏆 Best Model: {best_model} (Accuracy: {best_acc:.4f})")
        print(f"📊 Results saved to: {self.results_dir}/")
        
        return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained models")
    parser.add_argument("--models_dir", type=str, default="models",
                        help="Directory with trained models")
    parser.add_argument("--results_dir", type=str, default="results",
                        help="Directory to save evaluation results")
    parser.add_argument("--data_dir", type=str, default="data/processed",
                        help="Directory with preprocessed data")
    
    args = parser.parse_args()
    
    evaluator = ModelEvaluator(
        models_dir=args.models_dir,
        results_dir=args.results_dir,
        data_dir=args.data_dir
    )
    
    results = evaluator.run_evaluation()


if __name__ == "__main__":
    main()
