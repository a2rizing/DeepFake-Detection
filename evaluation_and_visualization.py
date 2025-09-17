#!/usr/bin/env python3
"""
Model Evaluation and Visualization Script
Comprehensive analysis of deepfake detection models with visualizations
"""

import os
import json
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_curve, roc_auc_score,
    precision_recall_curve, average_precision_score, accuracy_score,
    precision_score, recall_score, f1_score
)
from sklearn.model_selection import cross_val_score
import joblib
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class ModelEvaluator:
    def __init__(self, models_dir="models", output_dir="evaluation_results"):
        """Initialize model evaluator"""
        self.models_dir = models_dir
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        self.models = {}
        self.scalers = {}
        self.metadata = {}
        
        # Load all available models
        self.load_all_models()
    
    def load_all_models(self):
        """Load all trained models and their metadata"""
        print("Loading available models...")
        
        # Find all metadata files
        metadata_files = glob.glob(os.path.join(self.models_dir, "*metadata*.json"))
        
        for metadata_file in metadata_files:
            try:
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                
                model_type = metadata['model_type']
                timestamp = metadata['timestamp']
                
                # Load traditional ML models (joblib)
                if metadata_file.endswith('metadata.json') and 'deep' not in metadata_file:
                    model_files = glob.glob(os.path.join(self.models_dir, f"*{timestamp}*.joblib"))
                    scaler_files = glob.glob(os.path.join(self.models_dir, f"scaler*{timestamp}*.joblib"))
                    
                    # Find model file (not scaler)
                    model_file = None
                    for mf in model_files:
                        if 'scaler' not in os.path.basename(mf):
                            model_file = mf
                            break
                    
                    if model_file and os.path.exists(model_file):
                        self.models[model_type] = joblib.load(model_file)
                        if scaler_files and os.path.exists(scaler_files[0]):
                            self.scalers[model_type] = joblib.load(scaler_files[0])
                        self.metadata[model_type] = metadata
                        print(f"  Loaded {model_type}: {os.path.basename(model_file)}")
                
                # Load deep learning models (h5)
                elif 'deep' in metadata_file:
                    try:
                        import tensorflow as tf
                        model_path = metadata.get('model_path')
                        if model_path and os.path.exists(model_path):
                            self.models[model_type] = tf.keras.models.load_model(model_path)
                            scaler_path = metadata.get('scaler_path')
                            if scaler_path and os.path.exists(scaler_path):
                                self.scalers[model_type] = joblib.load(scaler_path)
                            self.metadata[model_type] = metadata
                            print(f"  Loaded {model_type}: {os.path.basename(model_path)}")
                    except ImportError:
                        print(f"  Skipping {model_type}: TensorFlow not available")
                
            except Exception as e:
                print(f"  Error loading {metadata_file}: {e}")
        
        print(f"Total models loaded: {len(self.models)}")
        if not self.models:
            print("No models found! Please train models first.")
    
    def load_test_data(self, data_dir="data/processed"):
        """Load preprocessed test data"""
        X_path = os.path.join(data_dir, "X.npy")
        y_path = os.path.join(data_dir, "y.npy")
        labels_path = os.path.join(data_dir, "labels.json")
        
        if not all(os.path.exists(p) for p in [X_path, y_path]):
            raise FileNotFoundError("Preprocessed data not found! Run preprocess_gait.py first.")
        
        X = np.load(X_path)
        y = np.load(y_path)
        
        # Load labels if available
        labels = {}
        if os.path.exists(labels_path):
            with open(labels_path, 'r') as f:
                labels = json.load(f)
        
        print(f"Loaded test data: X{X.shape}, y{y.shape}")
        return X, y, labels
    
    def predict_with_model(self, model_name, X):
        """Make predictions with a specific model"""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found!")
        
        model = self.models[model_name]
        X_processed = X.copy()
        
        # Apply scaler if available
        if model_name in self.scalers:
            X_processed = self.scalers[model_name].transform(X_processed)
        
        # Handle different model types
        if hasattr(model, 'predict_proba'):
            # Scikit-learn models
            y_pred_proba = model.predict_proba(X_processed)
            y_pred = model.predict(X_processed)
            return y_pred, y_pred_proba
        else:
            # Deep learning models
            try:
                # Reshape for deep learning if needed
                if len(X_processed.shape) == 2:
                    sequence_length = self.metadata[model_name].get('sequence_length', 64)
                    num_features = self.metadata[model_name].get('num_features', 70)
                    n_samples = X_processed.shape[0]
                    X_processed = X_processed.reshape(n_samples, sequence_length, num_features)
                
                y_pred_proba = model.predict(X_processed)
                y_pred = np.argmax(y_pred_proba, axis=1)
                return y_pred, y_pred_proba
            except Exception as e:
                print(f"Error predicting with {model_name}: {e}")
                return None, None
    
    def evaluate_single_model(self, model_name, X, y):
        """Evaluate a single model and return metrics"""
        print(f"\nEvaluating {model_name}...")
        
        y_pred, y_pred_proba = self.predict_with_model(model_name, X)
        if y_pred is None:
            return None
        
        # Calculate metrics
        accuracy = accuracy_score(y, y_pred)
        precision = precision_score(y, y_pred, average='weighted')
        recall = recall_score(y, y_pred, average='weighted')
        f1 = f1_score(y, y_pred, average='weighted')
        
        # AUC score
        try:
            if y_pred_proba.shape[1] == 2:  # Binary classification
                auc = roc_auc_score(y, y_pred_proba[:, 1])
            else:
                auc = roc_auc_score(y, y_pred_proba, multi_class='ovr')
        except:
            auc = 0.0
        
        metrics = {
            'model_name': model_name,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'auc_score': auc,
            'predictions': y_pred,
            'probabilities': y_pred_proba
        }
        
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        print(f"  F1-Score: {f1:.4f}")
        print(f"  AUC: {auc:.4f}")
        
        return metrics
    
    def plot_confusion_matrix(self, y_true, y_pred, model_name, save_path):
        """Plot confusion matrix"""
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Authentic', 'Deepfake'],
                   yticklabels=['Authentic', 'Deepfake'])
        plt.title(f'Confusion Matrix - {model_name}')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_roc_curve(self, y_true, y_pred_proba, model_name, save_path):
        """Plot ROC curve"""
        try:
            if y_pred_proba.shape[1] == 2:
                fpr, tpr, _ = roc_curve(y_true, y_pred_proba[:, 1])
                auc = roc_auc_score(y_true, y_pred_proba[:, 1])
            else:
                fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
                auc = roc_auc_score(y_true, y_pred_proba)
            
            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, linewidth=2, label=f'{model_name} (AUC = {auc:.3f})')
            plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'ROC Curve - {model_name}')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        except Exception as e:
            print(f"Error plotting ROC curve for {model_name}: {e}")
    
    def plot_precision_recall_curve(self, y_true, y_pred_proba, model_name, save_path):
        """Plot Precision-Recall curve"""
        try:
            if y_pred_proba.shape[1] == 2:
                precision, recall, _ = precision_recall_curve(y_true, y_pred_proba[:, 1])
                avg_precision = average_precision_score(y_true, y_pred_proba[:, 1])
            else:
                precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
                avg_precision = average_precision_score(y_true, y_pred_proba)
            
            plt.figure(figsize=(8, 6))
            plt.plot(recall, precision, linewidth=2, 
                    label=f'{model_name} (AP = {avg_precision:.3f})')
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title(f'Precision-Recall Curve - {model_name}')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        except Exception as e:
            print(f"Error plotting PR curve for {model_name}: {e}")
    
    def plot_model_comparison(self, all_metrics, save_path):
        """Plot comparison of all models"""
        if not all_metrics:
            return
        
        # Prepare data for plotting
        models = list(all_metrics.keys())
        metrics_names = ['accuracy', 'precision', 'recall', 'f1_score', 'auc_score']
        
        # Create comparison plot
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, metric in enumerate(metrics_names):
            values = [all_metrics[model][metric] for model in models]
            
            ax = axes[i]
            bars = ax.bar(models, values, color=sns.color_palette("husl", len(models)))
            ax.set_title(f'{metric.replace("_", " ").title()}')
            ax.set_ylabel('Score')
            ax.set_ylim(0, 1)
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                       f'{value:.3f}', ha='center', va='bottom')
            
            # Rotate x-axis labels if needed
            if len(max(models, key=len)) > 8:
                ax.tick_params(axis='x', rotation=45)
        
        # Remove the last subplot if we have an odd number of metrics
        if len(metrics_names) % 2 != 0:
            fig.delaxes(axes[-1])
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_combined_roc_curves(self, all_metrics, y_true, save_path):
        """Plot ROC curves for all models on one plot"""
        plt.figure(figsize=(10, 8))
        
        colors = sns.color_palette("husl", len(all_metrics))
        
        for i, (model_name, metrics) in enumerate(all_metrics.items()):
            try:
                y_pred_proba = metrics['probabilities']
                if y_pred_proba.shape[1] == 2:
                    fpr, tpr, _ = roc_curve(y_true, y_pred_proba[:, 1])
                    auc = roc_auc_score(y_true, y_pred_proba[:, 1])
                else:
                    continue  # Skip if not binary classification
                
                plt.plot(fpr, tpr, linewidth=2, color=colors[i],
                        label=f'{model_name} (AUC = {auc:.3f})')
            except Exception as e:
                print(f"Error plotting ROC for {model_name}: {e}")
        
        plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves - All Models')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_evaluation_report(self, all_metrics, save_path):
        """Generate comprehensive evaluation report"""
        with open(save_path, 'w') as f:
            f.write("DEEPFAKE DETECTION MODEL EVALUATION REPORT\n")
            f.write("=" * 50 + "\n\n")
            
            # Model summary
            f.write(f"Total Models Evaluated: {len(all_metrics)}\n")
            f.write(f"Evaluation Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Performance table
            f.write("PERFORMANCE SUMMARY\n")
            f.write("-" * 30 + "\n")
            f.write(f"{'Model':<15} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'AUC':<10}\n")
            f.write("-" * 75 + "\n")
            
            for model_name, metrics in all_metrics.items():
                f.write(f"{model_name:<15} {metrics['accuracy']:<10.4f} {metrics['precision']:<10.4f} "
                       f"{metrics['recall']:<10.4f} {metrics['f1_score']:<10.4f} {metrics['auc_score']:<10.4f}\n")
            
            # Best model
            best_model = max(all_metrics.keys(), key=lambda x: all_metrics[x]['accuracy'])
            f.write(f"\nBEST MODEL: {best_model}\n")
            f.write(f"Best Accuracy: {all_metrics[best_model]['accuracy']:.4f}\n\n")
            
            # Detailed analysis for each model
            f.write("DETAILED ANALYSIS\n")
            f.write("-" * 30 + "\n")
            
            for model_name, metrics in all_metrics.items():
                f.write(f"\n{model_name.upper()}\n")
                f.write("-" * len(model_name) + "\n")
                
                # Classification report
                y_true = [0] * len(metrics['predictions'])  # Placeholder
                y_pred = metrics['predictions']
                
                try:
                    # Get the actual y_true from the first model's data
                    if 'y_true' in metrics:
                        y_true = metrics['y_true']
                    
                    report = classification_report(y_true, y_pred, target_names=['Authentic', 'Deepfake'])
                    f.write(f"Classification Report:\n{report}\n")
                except:
                    f.write("Classification report not available\n")
            
            # Recommendations
            f.write("\nRECOMMENDATIONS\n")
            f.write("-" * 15 + "\n")
            
            # Find models with best specific metrics
            best_acc = max(all_metrics.keys(), key=lambda x: all_metrics[x]['accuracy'])
            best_f1 = max(all_metrics.keys(), key=lambda x: all_metrics[x]['f1_score'])
            best_auc = max(all_metrics.keys(), key=lambda x: all_metrics[x]['auc_score'])
            
            f.write(f"• For highest accuracy: Use {best_acc}\n")
            f.write(f"• For balanced performance: Use {best_f1}\n")
            f.write(f"• For probability calibration: Use {best_auc}\n\n")
            
            f.write("EVALUATION COMPLETE\n")
            f.write("=" * 20 + "\n")
    
    def run_comprehensive_evaluation(self):
        """Run complete evaluation pipeline"""
        print("=" * 60)
        print("COMPREHENSIVE MODEL EVALUATION")
        print("=" * 60)
        
        if not self.models:
            print("No models available for evaluation!")
            return None
        
        # Load test data
        try:
            X, y, labels = self.load_test_data()
        except FileNotFoundError as e:
            print(f"Error: {e}")
            return None
        
        # Evaluate all models
        all_metrics = {}
        
        for model_name in self.models.keys():
            try:
                metrics = self.evaluate_single_model(model_name, X, y)
                if metrics:
                    metrics['y_true'] = y  # Store true labels
                    all_metrics[model_name] = metrics
            except Exception as e:
                print(f"Error evaluating {model_name}: {e}")
        
        if not all_metrics:
            print("No models could be evaluated!")
            return None
        
        # Generate visualizations
        print(f"\nGenerating visualizations...")
        
        # Individual model plots
        for model_name, metrics in all_metrics.items():
            model_dir = os.path.join(self.output_dir, model_name)
            os.makedirs(model_dir, exist_ok=True)
            
            # Confusion matrix
            cm_path = os.path.join(model_dir, f"{model_name}_confusion_matrix.png")
            self.plot_confusion_matrix(y, metrics['predictions'], model_name, cm_path)
            
            # ROC curve
            roc_path = os.path.join(model_dir, f"{model_name}_roc_curve.png")
            self.plot_roc_curve(y, metrics['probabilities'], model_name, roc_path)
            
            # Precision-Recall curve
            pr_path = os.path.join(model_dir, f"{model_name}_pr_curve.png")
            self.plot_precision_recall_curve(y, metrics['probabilities'], model_name, pr_path)
        
        # Comparison plots
        comparison_path = os.path.join(self.output_dir, "model_comparison.png")
        self.plot_model_comparison(all_metrics, comparison_path)
        
        combined_roc_path = os.path.join(self.output_dir, "combined_roc_curves.png")
        self.plot_combined_roc_curves(all_metrics, y, combined_roc_path)
        
        # Generate report
        report_path = os.path.join(self.output_dir, "evaluation_report.txt")
        self.generate_evaluation_report(all_metrics, report_path)
        
        # Save metrics as JSON
        metrics_path = os.path.join(self.output_dir, "all_metrics.json")
        # Convert numpy arrays to lists for JSON serialization
        json_metrics = {}
        for model_name, metrics in all_metrics.items():
            json_metrics[model_name] = {
                'accuracy': float(metrics['accuracy']),
                'precision': float(metrics['precision']),
                'recall': float(metrics['recall']),
                'f1_score': float(metrics['f1_score']),
                'auc_score': float(metrics['auc_score'])
            }
        
        with open(metrics_path, 'w') as f:
            json.dump(json_metrics, f, indent=2)
        
        print(f"\nEvaluation complete!")
        print(f"Results saved in: {self.output_dir}")
        print(f"  - Individual model plots: {self.output_dir}/[model_name]/")
        print(f"  - Comparison plots: {comparison_path}")
        print(f"  - Combined ROC curves: {combined_roc_path}")
        print(f"  - Detailed report: {report_path}")
        print(f"  - Metrics JSON: {metrics_path}")
        
        # Print summary
        print(f"\nSUMMARY:")
        best_model = max(all_metrics.keys(), key=lambda x: all_metrics[x]['accuracy'])
        best_accuracy = all_metrics[best_model]['accuracy']
        print(f"Best Model: {best_model} (Accuracy: {best_accuracy:.4f})")
        
        return all_metrics

def main():
    """Main evaluation function"""
    evaluator = ModelEvaluator()
    results = evaluator.run_comprehensive_evaluation()
    
    if results:
        print("\nModel evaluation completed successfully!")
        print("Check the 'evaluation_results' directory for detailed analysis.")
    else:
        print("Model evaluation failed. Please check if models are trained and data is available.")

if __name__ == "__main__":
    main()