"""
Generate Confusion Matrices for Existing Models
Creates confusion matrix visualizations from your trained models
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import tensorflow as tf

def load_data():
    """Load processed data"""
    print("📊 Loading data...")
    X = np.load('data/processed/X.npy')
    y = np.load('data/processed/y.npy')
    
    with open('data/processed/labels.json', 'r') as f:
        labels = json.load(f)
    
    # Reverse mapping (id -> name)
    label_names = {v: k for k, v in labels.items()}
    
    print(f"✅ Data loaded: {X.shape}, Labels: {y.shape}")
    return X, y, label_names

def create_confusion_matrix(y_true, y_pred, label_names, model_name, save_path):
    """Create and save confusion matrix visualization"""
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Create figure
    plt.figure(figsize=(10, 8))
    
    # Create heatmap
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=[label_names[i] for i in range(len(label_names))],
                yticklabels=[label_names[i] for i in range(len(label_names))],
                cbar_kws={'label': 'Count'})
    
    plt.title(f'Confusion Matrix - {model_name}', fontsize=16, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    
    # Save
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {save_path}")
    plt.close()
    
    # Calculate and print metrics
    accuracy = np.trace(cm) / np.sum(cm)
    print(f"\n📊 {model_name} Metrics:")
    print(f"   Accuracy: {accuracy * 100:.2f}%")
    print(f"   Confusion Matrix:\n{cm}")
    
    return cm

def generate_all_confusion_matrices():
    """Generate confusion matrices for all trained models"""
    
    print("=" * 80)
    print("🎯 GENERATING CONFUSION MATRICES")
    print("=" * 80)
    
    # Load data
    X, y, label_names = load_data()
    
    # Create output directory
    output_dir = 'data/visualizations'
    os.makedirs(output_dir, exist_ok=True)
    
    # Model files
    model_files = {
        'LSTM': 'models/LSTM_20250917_164301.h5',
        'CNN': 'models/CNN_20250917_164306.h5',
        'Hybrid': 'models/Hybrid_20250917_164314.h5'
    }
    
    results = {}
    
    for model_name, model_path in model_files.items():
        if not os.path.exists(model_path):
            print(f"⚠️  Model not found: {model_path}")
            continue
        
        print(f"\n{'='*80}")
        print(f"🔄 Processing {model_name} Model")
        print(f"{'='*80}")
        
        try:
            # Load model
            print(f"Loading model from: {model_path}")
            model = tf.keras.models.load_model(model_path)
            
            # Make predictions
            print("Making predictions...")
            y_pred_proba = model.predict(X, verbose=0)
            y_pred = np.argmax(y_pred_proba, axis=1)
            
            # Create confusion matrix
            cm_path = os.path.join(output_dir, f'confusion_matrix_{model_name.lower()}.png')
            cm = create_confusion_matrix(y, y_pred, label_names, model_name, cm_path)
            
            # Store results
            results[model_name] = {
                'confusion_matrix': cm.tolist(),
                'accuracy': float(np.trace(cm) / np.sum(cm)),
                'predictions': y_pred.tolist()
            }
            
            # Print classification report
            print(f"\n📋 Classification Report - {model_name}:")
            print(classification_report(y, y_pred, 
                                       target_names=[label_names[i] for i in range(len(label_names))]))
            
        except Exception as e:
            print(f"❌ Error processing {model_name}: {e}")
    
    # Create combined comparison
    if len(results) > 1:
        print(f"\n{'='*80}")
        print("📊 Creating Combined Comparison")
        print(f"{'='*80}")
        create_combined_confusion_matrix(results, label_names, output_dir)
    
    # Save results
    results_path = os.path.join(output_dir, 'confusion_matrix_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✅ Results saved to: {results_path}")
    
    return results

def create_combined_confusion_matrix(results, label_names, output_dir):
    """Create side-by-side comparison of all confusion matrices"""
    
    num_models = len(results)
    fig, axes = plt.subplots(1, num_models, figsize=(8 * num_models, 6))
    
    if num_models == 1:
        axes = [axes]
    
    for idx, (model_name, data) in enumerate(results.items()):
        cm = np.array(data['confusion_matrix'])
        accuracy = data['accuracy']
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                   xticklabels=[label_names[i] for i in range(len(label_names))],
                   yticklabels=[label_names[i] for i in range(len(label_names))],
                   cbar_kws={'label': 'Count'})
        
        axes[idx].set_title(f'{model_name}\nAccuracy: {accuracy * 100:.1f}%', 
                           fontsize=14, fontweight='bold')
        axes[idx].set_ylabel('True Label', fontsize=11)
        axes[idx].set_xlabel('Predicted Label', fontsize=11)
    
    plt.tight_layout()
    
    combined_path = os.path.join(output_dir, 'confusion_matrix_comparison.png')
    plt.savefig(combined_path, dpi=300, bbox_inches='tight')
    print(f"✅ Combined comparison saved: {combined_path}")
    plt.close()

def create_per_class_accuracy_plot(results, label_names, output_dir):
    """Create bar chart showing per-class accuracy for each model"""
    
    print("\n📊 Creating per-class accuracy visualization...")
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(label_names))
    width = 0.25
    
    for idx, (model_name, data) in enumerate(results.items()):
        cm = np.array(data['confusion_matrix'])
        # Calculate per-class accuracy
        per_class_acc = np.diag(cm) / cm.sum(axis=1) * 100
        
        offset = (idx - len(results)/2 + 0.5) * width
        ax.bar(x + offset, per_class_acc, width, label=model_name)
    
    ax.set_xlabel('Person', fontsize=12)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('Per-Class Accuracy by Model', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([label_names[i] for i in range(len(label_names))])
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim([0, 105])
    
    plt.tight_layout()
    
    acc_path = os.path.join(output_dir, 'per_class_accuracy.png')
    plt.savefig(acc_path, dpi=300, bbox_inches='tight')
    print(f"✅ Per-class accuracy plot saved: {acc_path}")
    plt.close()

def main():
    """Main function"""
    
    print("\n")
    print("█" * 80)
    print("█" + " " * 78 + "█")
    print("█" + " " * 20 + "CONFUSION MATRIX GENERATOR" + " " * 32 + "█")
    print("█" + " " * 78 + "█")
    print("█" * 80)
    print()
    
    # Check if data exists
    if not os.path.exists('data/processed/X.npy'):
        print("❌ Data not found! Please run training first.")
        return
    
    # Generate confusion matrices
    results = generate_all_confusion_matrices()
    
    if results:
        # Load data for additional visualizations
        _, _, label_names = load_data()
        output_dir = 'data/visualizations'
        
        # Create per-class accuracy plot
        create_per_class_accuracy_plot(results, label_names, output_dir)
        
        print("\n" + "=" * 80)
        print("✅ ALL CONFUSION MATRICES GENERATED!")
        print("=" * 80)
        
        print("\n📁 Generated Files:")
        print("   • confusion_matrix_lstm.png")
        print("   • confusion_matrix_cnn.png")
        print("   • confusion_matrix_hybrid.png")
        print("   • confusion_matrix_comparison.png (side-by-side)")
        print("   • per_class_accuracy.png")
        print("   • confusion_matrix_results.json")
        
        print(f"\n📂 Location: {os.path.abspath(output_dir)}")
        
        print("\n🎯 Summary:")
        for model_name, data in results.items():
            print(f"   {model_name}: {data['accuracy'] * 100:.2f}% accuracy")
        
        print("\n💡 To view:")
        print(f"   explorer {output_dir}")
        
        print("\n" + "=" * 80)
    else:
        print("\n❌ No models processed. Check if model files exist.")

if __name__ == "__main__":
    main()
