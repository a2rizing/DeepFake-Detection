#!/usr/bin/env python3
"""
Best Model Report Generator
Generates comprehensive metrics, visualizations, and figures for the best trained model.

Outputs are saved to: results/best_model_report/

Usage:
    python generate_best_model_report.py
"""

import os
import sys
import json
import numpy as np
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
tf.get_logger().setLevel('ERROR')

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc,
    precision_recall_curve, average_precision_score
)
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import train_test_split


def setup_output_dir():
    """Create and return the output directory path"""
    output_dir = os.path.join("results", "best_model_report")
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def load_best_model_info():
    """Load information about the best model"""
    best_info_path = os.path.join("models", "best_model_info.json")
    
    if not os.path.exists(best_info_path):
        print("[ERROR] best_model_info.json not found!")
        sys.exit(1)
    
    with open(best_info_path) as f:
        best_info = json.load(f)
    
    return best_info


def load_model(model_path):
    """Load the trained Keras model"""
    if not os.path.exists(model_path):
        print(f"[ERROR] Model not found at {model_path}")
        sys.exit(1)
    
    model = tf.keras.models.load_model(model_path)
    return model


def load_data():
    """Load preprocessed data"""
    data_dir = os.path.join("data", "processed")
    
    X = np.load(os.path.join(data_dir, "X.npy"))
    y = np.load(os.path.join(data_dir, "y.npy"))
    
    with open(os.path.join(data_dir, "labels.json")) as f:
        labels = json.load(f)
    
    # Create id to name mapping
    id_to_name = {v: k for k, v in labels.items()}
    class_names = [id_to_name[i] for i in range(len(labels))]
    
    return X, y, labels, class_names


def evaluate_model(model, X, y, class_names, random_state=42):
    """
    Evaluate model and compute all metrics
    Uses the same test split as training (20% test)
    """
    # Split data (same as training)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )
    
    # Get predictions
    y_proba = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_proba, axis=1)
    
    # Compute metrics
    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision_macro": float(precision_score(y_test, y_pred, average='macro', zero_division=0)),
        "precision_weighted": float(precision_score(y_test, y_pred, average='weighted', zero_division=0)),
        "recall_macro": float(recall_score(y_test, y_pred, average='macro', zero_division=0)),
        "recall_weighted": float(recall_score(y_test, y_pred, average='weighted', zero_division=0)),
        "f1_macro": float(f1_score(y_test, y_pred, average='macro', zero_division=0)),
        "f1_weighted": float(f1_score(y_test, y_pred, average='weighted', zero_division=0)),
    }
    
    # ROC-AUC (one-vs-rest)
    num_classes = len(class_names)
    y_test_bin = label_binarize(y_test, classes=list(range(num_classes)))
    
    if num_classes == 2:
        y_test_bin = np.hstack((1 - y_test_bin, y_test_bin))
    
    try:
        from sklearn.metrics import roc_auc_score
        metrics["roc_auc"] = float(roc_auc_score(y_test_bin, y_proba, average='macro', multi_class='ovr'))
    except Exception as e:
        metrics["roc_auc"] = None
        print(f"[WARN] Could not compute ROC-AUC: {e}")
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    metrics["confusion_matrix"] = cm.tolist()
    
    # Per-class metrics
    per_class = {}
    for i, class_name in enumerate(class_names):
        y_true_binary = (y_test == i).astype(int)
        y_pred_binary = (y_pred == i).astype(int)
        
        per_class[class_name] = {
            "precision": float(precision_score(y_true_binary, y_pred_binary, zero_division=0)),
            "recall": float(recall_score(y_true_binary, y_pred_binary, zero_division=0)),
            "f1": float(f1_score(y_true_binary, y_pred_binary, zero_division=0)),
            "support": int(np.sum(y_test == i))
        }
    
    metrics["per_class"] = per_class
    
    return metrics, y_test, y_pred, y_proba, y_test_bin


def plot_confusion_matrix(y_true, y_pred, class_names, output_dir, normalize=False):
    """Generate and save confusion matrix heatmap"""
    cm = confusion_matrix(y_true, y_pred)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
        title = 'Normalized Confusion Matrix'
        filename = 'confusion_matrix_normalized.png'
    else:
        fmt = 'd'
        title = 'Confusion Matrix'
        filename = 'confusion_matrix.png'
    
    plt.figure(figsize=(14, 12))
    sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count' if not normalize else 'Proportion'})
    plt.title(title, fontsize=16, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    save_path = os.path.join(output_dir, filename)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   ✓ Saved: {filename}")
    
    return save_path


def plot_roc_curves(y_true_bin, y_proba, class_names, output_dir):
    """Generate and save ROC curves for all classes"""
    num_classes = len(class_names)
    
    # Compute ROC curve for each class
    fpr = {}
    tpr = {}
    roc_auc = {}
    
    for i in range(num_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_proba[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    # Micro-average
    fpr["micro"], tpr["micro"], _ = roc_curve(y_true_bin.ravel(), y_proba.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    
    # Macro-average
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(num_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(num_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= num_classes
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    
    # Plot
    plt.figure(figsize=(14, 10))
    
    # Plot micro and macro averages
    plt.plot(fpr["micro"], tpr["micro"],
             label=f'Micro-average (AUC = {roc_auc["micro"]:.3f})',
             color='deeppink', linestyle=':', linewidth=3)
    plt.plot(fpr["macro"], tpr["macro"],
             label=f'Macro-average (AUC = {roc_auc["macro"]:.3f})',
             color='navy', linestyle=':', linewidth=3)
    
    # Plot each class
    colors = plt.cm.tab20(np.linspace(0, 1, num_classes))
    for i, (color, class_name) in enumerate(zip(colors, class_names)):
        plt.plot(fpr[i], tpr[i], color=color, lw=2, alpha=0.8,
                 label=f'{class_name} (AUC = {roc_auc[i]:.3f})')
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier')
    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.02])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curves (One-vs-Rest)', fontsize=16, fontweight='bold')
    plt.legend(loc='lower right', fontsize=8, ncol=2)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    save_path = os.path.join(output_dir, 'roc_curves.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   ✓ Saved: roc_curves.png")
    
    return save_path


def plot_precision_recall_curves(y_true_bin, y_proba, class_names, output_dir):
    """Generate and save precision-recall curves"""
    num_classes = len(class_names)
    
    precision = {}
    recall = {}
    avg_precision = {}
    
    for i in range(num_classes):
        precision[i], recall[i], _ = precision_recall_curve(y_true_bin[:, i], y_proba[:, i])
        avg_precision[i] = average_precision_score(y_true_bin[:, i], y_proba[:, i])
    
    # Micro-average
    precision["micro"], recall["micro"], _ = precision_recall_curve(
        y_true_bin.ravel(), y_proba.ravel())
    avg_precision["micro"] = average_precision_score(y_true_bin, y_proba, average="micro")
    
    # Plot
    plt.figure(figsize=(14, 10))
    
    # Plot micro-average
    plt.plot(recall["micro"], precision["micro"],
             label=f'Micro-average (AP = {avg_precision["micro"]:.3f})',
             color='gold', linestyle=':', linewidth=3)
    
    # Plot each class
    colors = plt.cm.tab20(np.linspace(0, 1, num_classes))
    for i, (color, class_name) in enumerate(zip(colors, class_names)):
        plt.plot(recall[i], precision[i], color=color, lw=2, alpha=0.8,
                 label=f'{class_name} (AP = {avg_precision[i]:.3f})')
    
    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.02])
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('Precision-Recall Curves', fontsize=16, fontweight='bold')
    plt.legend(loc='lower left', fontsize=8, ncol=2)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    save_path = os.path.join(output_dir, 'precision_recall_curves.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   ✓ Saved: precision_recall_curves.png")
    
    return save_path


def plot_per_class_metrics(metrics, class_names, output_dir):
    """Generate per-class metrics bar chart"""
    per_class = metrics["per_class"]
    
    precisions = [per_class[cn]["precision"] for cn in class_names]
    recalls = [per_class[cn]["recall"] for cn in class_names]
    f1s = [per_class[cn]["f1"] for cn in class_names]
    
    x = np.arange(len(class_names))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(16, 8))
    
    bars1 = ax.bar(x - width, precisions, width, label='Precision', color='#2ecc71', alpha=0.8)
    bars2 = ax.bar(x, recalls, width, label='Recall', color='#3498db', alpha=0.8)
    bars3 = ax.bar(x + width, f1s, width, label='F1-Score', color='#9b59b6', alpha=0.8)
    
    ax.set_xlabel('Class', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Per-Class Metrics', fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.legend()
    ax.set_ylim([0, 1.1])
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3), textcoords="offset points",
                       ha='center', va='bottom', fontsize=7, rotation=90)
    
    plt.tight_layout()
    
    save_path = os.path.join(output_dir, 'per_class_metrics.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   ✓ Saved: per_class_metrics.png")
    
    return save_path


def plot_metrics_summary(metrics, output_dir):
    """Generate overall metrics summary bar chart"""
    metric_names = ['Accuracy', 'Precision\n(Macro)', 'Precision\n(Weighted)', 
                    'Recall\n(Macro)', 'Recall\n(Weighted)', 'F1\n(Macro)', 
                    'F1\n(Weighted)', 'ROC-AUC']
    values = [
        metrics['accuracy'],
        metrics['precision_macro'],
        metrics['precision_weighted'],
        metrics['recall_macro'],
        metrics['recall_weighted'],
        metrics['f1_macro'],
        metrics['f1_weighted'],
        metrics.get('roc_auc', 0) or 0
    ]
    
    colors = ['#e74c3c', '#2ecc71', '#27ae60', '#3498db', '#2980b9', 
              '#9b59b6', '#8e44ad', '#f39c12']
    
    fig, ax = plt.subplots(figsize=(14, 8))
    bars = ax.bar(metric_names, values, color=colors, alpha=0.85, edgecolor='black', linewidth=1.2)
    
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Model Performance Summary', fontsize=16, fontweight='bold')
    ax.set_ylim([0, 1.15])
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, value in zip(bars, values):
        ax.annotate(f'{value:.4f}',
                   xy=(bar.get_x() + bar.get_width() / 2, value),
                   xytext=(0, 5), textcoords="offset points",
                   ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    
    save_path = os.path.join(output_dir, 'metrics_summary.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   ✓ Saved: metrics_summary.png")
    
    return save_path


def plot_class_distribution(y_true, class_names, output_dir):
    """Plot class distribution in test set"""
    unique, counts = np.unique(y_true, return_counts=True)
    
    fig, ax = plt.subplots(figsize=(14, 8))
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(class_names)))
    
    bars = ax.bar(range(len(class_names)), [counts[unique.tolist().index(i)] if i in unique else 0 
                                             for i in range(len(class_names))],
                  color=colors, alpha=0.85, edgecolor='black', linewidth=1.2)
    
    ax.set_xlabel('Class', fontsize=12)
    ax.set_ylabel('Number of Samples', fontsize=12)
    ax.set_title('Test Set Class Distribution', fontsize=16, fontweight='bold')
    ax.set_xticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add count labels
    for bar, count in zip(bars, [counts[unique.tolist().index(i)] if i in unique else 0 
                                  for i in range(len(class_names))]):
        ax.annotate(f'{count}',
                   xy=(bar.get_x() + bar.get_width() / 2, count),
                   xytext=(0, 3), textcoords="offset points",
                   ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    
    save_path = os.path.join(output_dir, 'class_distribution.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   ✓ Saved: class_distribution.png")
    
    return save_path


def plot_prediction_confidence(y_proba, y_true, y_pred, output_dir):
    """Plot prediction confidence distribution"""
    max_proba = np.max(y_proba, axis=1)
    correct = y_true == y_pred
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Left: Histogram of confidence scores
    ax1 = axes[0]
    ax1.hist(max_proba[correct], bins=20, alpha=0.7, label='Correct', color='#2ecc71', edgecolor='black')
    ax1.hist(max_proba[~correct], bins=20, alpha=0.7, label='Incorrect', color='#e74c3c', edgecolor='black')
    ax1.set_xlabel('Prediction Confidence', fontsize=12)
    ax1.set_ylabel('Count', fontsize=12)
    ax1.set_title('Confidence Distribution', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Right: Box plot by correctness
    ax2 = axes[1]
    data = [max_proba[correct], max_proba[~correct]]
    bp = ax2.boxplot(data, labels=['Correct', 'Incorrect'], patch_artist=True)
    bp['boxes'][0].set_facecolor('#2ecc71')
    bp['boxes'][1].set_facecolor('#e74c3c')
    ax2.set_ylabel('Prediction Confidence', fontsize=12)
    ax2.set_title('Confidence by Prediction Correctness', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    save_path = os.path.join(output_dir, 'prediction_confidence.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   ✓ Saved: prediction_confidence.png")
    
    return save_path


def generate_text_report(best_info, metrics, class_names, output_dir):
    """Generate a comprehensive text report"""
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("BEST MODEL EVALUATION REPORT")
    report_lines.append("=" * 80)
    report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("")
    
    # Model Information
    report_lines.append("-" * 40)
    report_lines.append("MODEL INFORMATION")
    report_lines.append("-" * 40)
    report_lines.append(f"Model Name:     {best_info['model_name']}")
    report_lines.append(f"Model Path:     {best_info['model_path']}")
    report_lines.append(f"Num Classes:    {best_info['num_classes']}")
    report_lines.append(f"Sequence Length: {best_info['sequence_length']}")
    report_lines.append(f"Num Features:   {best_info['num_features']}")
    report_lines.append(f"Training Time:  {best_info.get('timestamp', 'N/A')}")
    report_lines.append("")
    
    # Overall Metrics
    report_lines.append("-" * 40)
    report_lines.append("OVERALL METRICS")
    report_lines.append("-" * 40)
    report_lines.append(f"Accuracy:           {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    report_lines.append(f"Precision (Macro):  {metrics['precision_macro']:.4f}")
    report_lines.append(f"Precision (Weighted): {metrics['precision_weighted']:.4f}")
    report_lines.append(f"Recall (Macro):     {metrics['recall_macro']:.4f}")
    report_lines.append(f"Recall (Weighted):  {metrics['recall_weighted']:.4f}")
    report_lines.append(f"F1-Score (Macro):   {metrics['f1_macro']:.4f}")
    report_lines.append(f"F1-Score (Weighted): {metrics['f1_weighted']:.4f}")
    if metrics.get('roc_auc'):
        report_lines.append(f"ROC-AUC:            {metrics['roc_auc']:.4f}")
    report_lines.append("")
    
    # Per-class Metrics
    report_lines.append("-" * 40)
    report_lines.append("PER-CLASS METRICS")
    report_lines.append("-" * 40)
    report_lines.append(f"{'Class':<15} {'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'Support':>10}")
    report_lines.append("-" * 55)
    
    for class_name in class_names:
        pc = metrics['per_class'][class_name]
        report_lines.append(f"{class_name:<15} {pc['precision']:>10.4f} {pc['recall']:>10.4f} {pc['f1']:>10.4f} {pc['support']:>10}")
    
    report_lines.append("")
    report_lines.append("=" * 80)
    report_lines.append("END OF REPORT")
    report_lines.append("=" * 80)
    
    # Write report
    report_text = "\n".join(report_lines)
    save_path = os.path.join(output_dir, 'evaluation_report.txt')
    with open(save_path, 'w') as f:
        f.write(report_text)
    
    print(f"   ✓ Saved: evaluation_report.txt")
    
    return save_path, report_text


def generate_model_comparison(output_dir):
    """Generate comparison chart with all trained models"""
    summary_path = os.path.join("results", "training_summary_20251220_162048.json")
    
    if not os.path.exists(summary_path):
        print("   [SKIP] training_summary not found for model comparison")
        return None
    
    with open(summary_path) as f:
        summary = json.load(f)
    
    results = summary.get("results", {})
    
    if not results:
        return None
    
    # Sort by accuracy
    sorted_models = sorted(results.items(), key=lambda x: x[1].get('accuracy', 0), reverse=True)
    model_names = [m[0] for m in sorted_models]
    accuracies = [m[1].get('accuracy', 0) for m in sorted_models]
    f1_scores = [m[1].get('f1_macro', 0) for m in sorted_models]
    roc_aucs = [m[1].get('roc_auc', 0) for m in sorted_models]
    
    # Create comparison plot
    fig, ax = plt.subplots(figsize=(14, 8))
    
    x = np.arange(len(model_names))
    width = 0.25
    
    bars1 = ax.bar(x - width, accuracies, width, label='Accuracy', color='#e74c3c', alpha=0.85)
    bars2 = ax.bar(x, f1_scores, width, label='F1 (Macro)', color='#3498db', alpha=0.85)
    bars3 = ax.bar(x + width, roc_aucs, width, label='ROC-AUC', color='#2ecc71', alpha=0.85)
    
    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Model Comparison (All Trained Models)', fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=45, ha='right')
    ax.legend()
    ax.set_ylim([0, 1.15])
    ax.grid(True, alpha=0.3, axis='y')
    
    # Highlight best model
    best_idx = 0  # Already sorted, first is best
    ax.axvline(x=best_idx, color='gold', linestyle='--', linewidth=2, alpha=0.7, label='Best Model')
    
    plt.tight_layout()
    
    save_path = os.path.join(output_dir, 'model_comparison.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   ✓ Saved: model_comparison.png")
    
    return save_path


def main():
    print("\n" + "=" * 70)
    print("BEST MODEL REPORT GENERATOR")
    print("DeepFake Detection using Gait Analysis")
    print("=" * 70)
    
    # Setup
    print("\n[1/7] Setting up output directory...")
    output_dir = setup_output_dir()
    print(f"   Output: {output_dir}")
    
    # Load best model info
    print("\n[2/7] Loading best model information...")
    best_info = load_best_model_info()
    print(f"   Best Model: {best_info['model_name']}")
    print(f"   Accuracy:   {best_info['accuracy']:.2%}")
    print(f"   F1 (Macro): {best_info['f1_macro']:.4f}")
    print(f"   ROC-AUC:    {best_info['roc_auc']:.4f}")
    
    # Load model
    print("\n[3/7] Loading trained model...")
    model = load_model(best_info['model_path'])
    print(f"   Model loaded successfully")
    
    # Load data
    print("\n[4/7] Loading preprocessed data...")
    X, y, labels, class_names = load_data()
    print(f"   Samples: {len(X)}")
    print(f"   Classes: {len(class_names)} ({', '.join(class_names)})")
    
    # Evaluate model
    print("\n[5/7] Evaluating model...")
    metrics, y_test, y_pred, y_proba, y_test_bin = evaluate_model(model, X, y, class_names)
    print(f"   Accuracy:   {metrics['accuracy']:.2%}")
    print(f"   F1 (Macro): {metrics['f1_macro']:.4f}")
    
    # Generate visualizations
    print("\n[6/7] Generating visualizations...")
    plot_confusion_matrix(y_test, y_pred, class_names, output_dir, normalize=False)
    plot_confusion_matrix(y_test, y_pred, class_names, output_dir, normalize=True)
    plot_roc_curves(y_test_bin, y_proba, class_names, output_dir)
    plot_precision_recall_curves(y_test_bin, y_proba, class_names, output_dir)
    plot_per_class_metrics(metrics, class_names, output_dir)
    plot_metrics_summary(metrics, output_dir)
    plot_class_distribution(y_test, class_names, output_dir)
    plot_prediction_confidence(y_proba, y_test, y_pred, output_dir)
    generate_model_comparison(output_dir)
    
    # Generate reports
    print("\n[7/7] Generating reports...")
    
    # Text report
    _, report_text = generate_text_report(best_info, metrics, class_names, output_dir)
    
    # Save metrics JSON
    metrics_with_info = {
        "model_info": best_info,
        "evaluation_metrics": metrics,
        "generated_at": datetime.now().isoformat()
    }
    
    metrics_path = os.path.join(output_dir, 'full_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics_with_info, f, indent=2)
    print(f"   ✓ Saved: full_metrics.json")
    
    # Save classification report
    from sklearn.metrics import classification_report
    clf_report = classification_report(y_test, y_pred, target_names=class_names)
    clf_report_path = os.path.join(output_dir, 'classification_report.txt')
    with open(clf_report_path, 'w') as f:
        f.write(clf_report)
    print(f"   ✓ Saved: classification_report.txt")
    
    # Summary
    print("\n" + "=" * 70)
    print("REPORT GENERATION COMPLETE")
    print("=" * 70)
    print(f"\n📁 All outputs saved to: {os.path.abspath(output_dir)}")
    print("\nGenerated files:")
    for f in sorted(os.listdir(output_dir)):
        fpath = os.path.join(output_dir, f)
        size_kb = os.path.getsize(fpath) / 1024
        print(f"   • {f} ({size_kb:.1f} KB)")
    
    # Print key metrics
    print("\n" + "-" * 40)
    print("KEY METRICS SUMMARY")
    print("-" * 40)
    print(report_text.split("OVERALL METRICS")[1].split("PER-CLASS METRICS")[0])
    
    print("\n✅ Done!")
    
    return output_dir


if __name__ == "__main__":
    main()
