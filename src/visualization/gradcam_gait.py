#!/usr/bin/env python3
"""
GradCAM Visualization for Gait-based Deepfake Detection

Implements 1D GradCAM for temporal sequence models to visualize:
- Which frames in the gait sequence are most important
- Which body keypoints contribute most to predictions
- Interpretability reports for model decisions

Usage:
    python src/visualization/gradcam_gait.py --model models/best_model.keras --data data/processed_augmented
    python src/visualization/gradcam_gait.py --model models/best_model.keras --video path/to/video.mp4
"""

import os
import sys
import json
import argparse
import numpy as np
from typing import Optional, Tuple, List

# Suppress TensorFlow warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


class GaitGradCAM:
    """
    GradCAM implementation for 1D temporal sequences.
    Adapted for gait analysis sequences.
    """
    
    def __init__(self, model: keras.Model, layer_name: Optional[str] = None):
        """
        Initialize GradCAM.
        
        Args:
            model: Trained Keras model
            layer_name: Name of convolutional layer to use for GradCAM.
                       If None, automatically finds last conv layer.
        """
        self.model = model
        
        # Find target layer
        if layer_name:
            self.target_layer = model.get_layer(layer_name)
        else:
            self.target_layer = self._find_last_conv_layer()
        
        if self.target_layer is None:
            raise ValueError("No convolutional layer found in model. "
                           "GradCAM requires at least one Conv1D layer.")
        
        print(f"GradCAM target layer: {self.target_layer.name}")
        
        # Create gradient model
        self.grad_model = keras.Model(
            inputs=self.model.input,
            outputs=[self.target_layer.output, self.model.output]
        )
    
    def _find_last_conv_layer(self) -> Optional[keras.layers.Layer]:
        """Find the last Conv1D layer in the model"""
        last_conv = None
        for layer in self.model.layers:
            if isinstance(layer, keras.layers.Conv1D):
                last_conv = layer
            # Also check inside nested models
            if hasattr(layer, 'layers'):
                for sublayer in layer.layers:
                    if isinstance(sublayer, keras.layers.Conv1D):
                        last_conv = sublayer
        return last_conv
    
    def compute_gradcam(self, input_sequence: np.ndarray, 
                        class_idx: Optional[int] = None) -> np.ndarray:
        """
        Compute GradCAM heatmap for a single input sequence.
        
        Args:
            input_sequence: Shape (1, seq_len, features) or (seq_len, features)
            class_idx: Class index for gradient computation. 
                      If None, uses predicted class.
        
        Returns:
            heatmap: Shape (seq_len,) normalized activation values
        """
        # Ensure batch dimension
        if len(input_sequence.shape) == 2:
            input_sequence = np.expand_dims(input_sequence, axis=0)
        
        input_tensor = tf.convert_to_tensor(input_sequence, dtype=tf.float32)
        
        # Compute gradients
        with tf.GradientTape() as tape:
            tape.watch(input_tensor)
            conv_output, predictions = self.grad_model(input_tensor)
            
            if class_idx is None:
                class_idx = tf.argmax(predictions[0])
            
            class_score = predictions[0, class_idx]
        
        # Gradient of class score w.r.t. conv output
        grads = tape.gradient(class_score, conv_output)
        
        # Global average pooling of gradients (across time dimension)
        pooled_grads = tf.reduce_mean(grads, axis=1)
        
        # Weight conv outputs by gradients
        conv_output = conv_output[0]  # Remove batch dim
        pooled_grads = pooled_grads[0]
        
        # Weighted sum across channels
        heatmap = tf.reduce_sum(conv_output * pooled_grads, axis=-1)
        
        # ReLU and normalize
        heatmap = tf.maximum(heatmap, 0)
        max_val = tf.reduce_max(heatmap)
        if max_val > 0:
            heatmap = heatmap / max_val
        
        return heatmap.numpy()
    
    def compute_feature_importance(self, input_sequence: np.ndarray,
                                    class_idx: Optional[int] = None) -> np.ndarray:
        """
        Compute importance of each feature (body keypoint) using gradient.
        
        Args:
            input_sequence: Shape (1, seq_len, features) or (seq_len, features)
            class_idx: Target class index
        
        Returns:
            importance: Shape (features,) normalized importance values
        """
        if len(input_sequence.shape) == 2:
            input_sequence = np.expand_dims(input_sequence, axis=0)
        
        input_tensor = tf.convert_to_tensor(input_sequence, dtype=tf.float32)
        
        with tf.GradientTape() as tape:
            tape.watch(input_tensor)
            predictions = self.model(input_tensor)
            
            if class_idx is None:
                class_idx = tf.argmax(predictions[0])
            
            class_score = predictions[0, class_idx]
        
        grads = tape.gradient(class_score, input_tensor)
        
        # Average across time, take absolute value for importance
        feature_importance = tf.reduce_mean(tf.abs(grads), axis=[0, 1])
        
        # Normalize
        feature_importance = feature_importance / tf.reduce_max(feature_importance)
        
        return feature_importance.numpy()


class GaitGradCAMVisualizer:
    """
    Visualization utilities for gait GradCAM results.
    """
    
    # MediaPipe pose keypoint names (33 keypoints, x/y each = 66 features)
    KEYPOINT_NAMES = [
        'nose', 'left_eye_inner', 'left_eye', 'left_eye_outer',
        'right_eye_inner', 'right_eye', 'right_eye_outer',
        'left_ear', 'right_ear', 'mouth_left', 'mouth_right',
        'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
        'left_wrist', 'right_wrist', 'left_pinky', 'right_pinky',
        'left_index', 'right_index', 'left_thumb', 'right_thumb',
        'left_hip', 'right_hip', 'left_knee', 'right_knee',
        'left_ankle', 'right_ankle', 'left_heel', 'right_heel',
        'left_foot_index', 'right_foot_index'
    ]
    
    def __init__(self, output_dir: str = "results/gradcam"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def plot_temporal_heatmap(self, heatmap: np.ndarray, 
                               title: str = "GradCAM Temporal Attention",
                               save_name: Optional[str] = None) -> plt.Figure:
        """
        Plot temporal heatmap showing frame importance.
        """
        fig, axes = plt.subplots(2, 1, figsize=(14, 6), 
                                  gridspec_kw={'height_ratios': [1, 3]})
        
        # Top: Heatmap bar
        ax1 = axes[0]
        ax1.imshow([heatmap], aspect='auto', cmap='jet', 
                   extent=[0, len(heatmap), 0, 1])
        ax1.set_yticks([])
        ax1.set_xlabel('Frame')
        ax1.set_title(title)
        
        # Add colorbar
        sm = plt.cm.ScalarMappable(cmap='jet', 
                                   norm=mcolors.Normalize(vmin=0, vmax=1))
        plt.colorbar(sm, ax=ax1, orientation='horizontal', 
                    label='Attention Score', pad=0.3)
        
        # Bottom: Line plot
        ax2 = axes[1]
        frames = np.arange(len(heatmap))
        ax2.fill_between(frames, heatmap, alpha=0.3, color='red')
        ax2.plot(frames, heatmap, color='red', linewidth=2)
        ax2.set_xlabel('Frame Index')
        ax2.set_ylabel('Attention Score')
        ax2.set_xlim(0, len(heatmap))
        ax2.set_ylim(0, 1)
        ax2.grid(True, alpha=0.3)
        
        # Mark peak frames
        peak_threshold = 0.7
        peaks = np.where(heatmap > peak_threshold)[0]
        if len(peaks) > 0:
            ax2.scatter(peaks, heatmap[peaks], color='darkred', 
                       s=100, zorder=5, label=f'High attention (>{peak_threshold})')
            ax2.legend()
        
        plt.tight_layout()
        
        if save_name:
            path = os.path.join(self.output_dir, f"{save_name}.png")
            plt.savefig(path, dpi=150, bbox_inches='tight')
            print(f"💾 Saved: {path}")
        
        return fig
    
    def plot_feature_importance(self, importance: np.ndarray,
                                  title: str = "Feature (Keypoint) Importance",
                                  save_name: Optional[str] = None) -> plt.Figure:
        """
        Plot feature importance as bar chart.
        Shows which body keypoints are most important.
        """
        num_features = len(importance)
        
        # If we have coordinate pairs (x, y), group them by keypoint
        if num_features == 66:  # 33 keypoints * 2 coords
            # Average x and y importance for each keypoint
            keypoint_importance = []
            for i in range(33):
                x_imp = importance[i * 2] if i * 2 < num_features else 0
                y_imp = importance[i * 2 + 1] if i * 2 + 1 < num_features else 0
                keypoint_importance.append((x_imp + y_imp) / 2)
            
            keypoint_importance = np.array(keypoint_importance)
            labels = self.KEYPOINT_NAMES[:33]
        else:
            keypoint_importance = importance
            labels = [f"Feature {i}" for i in range(len(importance))]
        
        # Sort by importance
        sorted_idx = np.argsort(keypoint_importance)[::-1]
        
        # Plot top 15
        top_k = min(15, len(sorted_idx))
        top_idx = sorted_idx[:top_k]
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        y_pos = np.arange(top_k)
        bars = ax.barh(y_pos, keypoint_importance[top_idx], 
                       color=plt.cm.Reds(keypoint_importance[top_idx]))
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels([labels[i] for i in top_idx])
        ax.invert_yaxis()
        ax.set_xlabel('Importance Score')
        ax.set_title(title)
        ax.set_xlim(0, 1)
        
        # Add value labels
        for bar, val in zip(bars, keypoint_importance[top_idx]):
            ax.text(val + 0.02, bar.get_y() + bar.get_height()/2,
                   f'{val:.3f}', va='center', fontsize=9)
        
        plt.tight_layout()
        
        if save_name:
            path = os.path.join(self.output_dir, f"{save_name}.png")
            plt.savefig(path, dpi=150, bbox_inches='tight')
            print(f"💾 Saved: {path}")
        
        return fig
    
    def plot_sequence_with_attention(self, sequence: np.ndarray, 
                                      heatmap: np.ndarray,
                                      feature_idx: int = 0,
                                      title: str = "Sequence with Attention Overlay",
                                      save_name: Optional[str] = None) -> plt.Figure:
        """
        Plot a feature trajectory with attention overlay.
        """
        fig, ax = plt.subplots(figsize=(14, 5))
        
        frames = np.arange(len(sequence))
        feature_values = sequence[:, feature_idx] if len(sequence.shape) > 1 else sequence
        
        # Create scatter plot with colors based on attention
        scatter = ax.scatter(frames, feature_values, c=heatmap, cmap='jet',
                            s=50, alpha=0.8, edgecolors='black', linewidth=0.5)
        ax.plot(frames, feature_values, 'k-', alpha=0.3, linewidth=1)
        
        ax.set_xlabel('Frame Index')
        ax.set_ylabel('Feature Value')
        ax.set_title(title)
        
        plt.colorbar(scatter, ax=ax, label='Attention Score')
        plt.tight_layout()
        
        if save_name:
            path = os.path.join(self.output_dir, f"{save_name}.png")
            plt.savefig(path, dpi=150, bbox_inches='tight')
            print(f"💾 Saved: {path}")
        
        return fig
    
    def generate_report(self, model: keras.Model, X: np.ndarray, y: np.ndarray,
                        num_samples: int = 5, save_name: str = "gradcam_report"):
        """
        Generate a comprehensive GradCAM report for multiple samples.
        """
        gradcam = GaitGradCAM(model)
        
        # Get predictions
        predictions = model.predict(X[:num_samples], verbose=0)
        pred_classes = np.argmax(predictions, axis=1)
        true_classes = np.argmax(y[:num_samples], axis=1) if len(y.shape) > 1 else y[:num_samples]
        
        report = {
            'samples': [],
            'average_attention': None,
            'average_feature_importance': None
        }
        
        all_heatmaps = []
        all_importances = []
        
        for i in range(num_samples):
            sample = X[i:i+1]
            
            # Compute GradCAM
            heatmap = gradcam.compute_gradcam(sample, pred_classes[i])
            importance = gradcam.compute_feature_importance(sample, pred_classes[i])
            
            all_heatmaps.append(heatmap)
            all_importances.append(importance)
            
            # Plot
            self.plot_temporal_heatmap(
                heatmap,
                title=f"Sample {i+1}: Pred={pred_classes[i]}, True={true_classes[i]}",
                save_name=f"{save_name}_sample_{i+1}_temporal"
            )
            
            # Find peak attention frames
            peak_frames = np.where(heatmap > 0.5)[0].tolist()
            
            report['samples'].append({
                'sample_index': i,
                'predicted_class': int(pred_classes[i]),
                'true_class': int(true_classes[i]),
                'correct': bool(pred_classes[i] == true_classes[i]),
                'confidence': float(predictions[i, pred_classes[i]]),
                'peak_attention_frames': peak_frames,
                'max_attention': float(np.max(heatmap))
            })
        
        # Average attention across samples
        avg_heatmap = np.mean(all_heatmaps, axis=0)
        avg_importance = np.mean(all_importances, axis=0)
        
        self.plot_temporal_heatmap(
            avg_heatmap,
            title="Average Temporal Attention Across Samples",
            save_name=f"{save_name}_avg_temporal"
        )
        
        self.plot_feature_importance(
            avg_importance,
            title="Average Feature (Keypoint) Importance",
            save_name=f"{save_name}_avg_features"
        )
        
        report['average_attention'] = avg_heatmap.tolist()
        report['average_feature_importance'] = avg_importance.tolist()
        
        # Save report JSON
        report_path = os.path.join(self.output_dir, f"{save_name}.json")
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"📄 Report saved: {report_path}")
        
        plt.close('all')
        
        return report


def main():
    parser = argparse.ArgumentParser(
        description="GradCAM Visualization for Gait Deepfake Detection"
    )
    parser.add_argument("--model", type=str, required=True,
                       help="Path to trained model (.keras file)")
    parser.add_argument("--data_dir", type=str, default="data/processed_augmented",
                       help="Directory with preprocessed data")
    parser.add_argument("--output_dir", type=str, default="results/gradcam",
                       help="Output directory for visualizations")
    parser.add_argument("--num_samples", type=int, default=5,
                       help="Number of samples to visualize")
    parser.add_argument("--layer", type=str, default=None,
                       help="Target layer name for GradCAM (default: last conv)")
    
    args = parser.parse_args()
    
    # Load model
    print(f"Loading model: {args.model}")
    model = keras.models.load_model(args.model)
    model.summary()
    
    # Load data
    X_path = os.path.join(args.data_dir, "X.npy")
    y_path = os.path.join(args.data_dir, "y.npy")
    
    if not os.path.exists(X_path):
        print(f"❌ Data not found: {X_path}")
        return
    
    X = np.load(X_path)
    y = np.load(y_path)
    
    # One-hot encode if needed
    if len(y.shape) == 1:
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        num_classes = len(np.unique(y))
        y = keras.utils.to_categorical(y_encoded, num_classes=num_classes)
    
    print(f"\n✅ Loaded data: X{X.shape}, y{y.shape}")
    
    # Generate report
    visualizer = GaitGradCAMVisualizer(output_dir=args.output_dir)
    
    print(f"\n🔍 Generating GradCAM visualizations...")
    report = visualizer.generate_report(
        model, X, y,
        num_samples=args.num_samples,
        save_name="gradcam_analysis"
    )
    
    # Print summary
    print("\n" + "="*50)
    print("GRADCAM ANALYSIS SUMMARY")
    print("="*50)
    
    correct = sum(1 for s in report['samples'] if s['correct'])
    print(f"Samples analyzed: {len(report['samples'])}")
    print(f"Correct predictions: {correct}/{len(report['samples'])}")
    
    print(f"\nVisualizations saved to: {args.output_dir}/")
    print("  - gradcam_analysis.json")
    print("  - gradcam_analysis_sample_*_temporal.png")
    print("  - gradcam_analysis_avg_temporal.png")
    print("  - gradcam_analysis_avg_features.png")


if __name__ == "__main__":
    main()
