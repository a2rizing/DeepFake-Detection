"""
Regenerate ALL visualizations for the updated 9-person dataset
Creates individual gait patterns, stride analysis, trajectories, etc.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from sklearn.decomposition import PCA
import cv2
import mediapipe as mp

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

def load_data():
    """Load processed data"""
    X = np.load('data/processed/X.npy')
    y = np.load('data/processed/y.npy')
    
    with open('data/processed/labels.json', 'r') as f:
        labels = json.load(f)
    
    # Reverse mapping: ID to name
    id_to_name = {v: k for k, v in labels.items()}
    
    return X, y, labels, id_to_name

def create_individual_gait_patterns(X, y, id_to_name):
    """Create gait pattern visualization for each person"""
    
    print("\n" + "="*80)
    print("Creating Individual Gait Patterns")
    print("="*80)
    
    os.makedirs('data/visualizations', exist_ok=True)
    
    for person_id in sorted(id_to_name.keys()):
        person_name = id_to_name[person_id]
        
        # Get all samples for this person
        person_samples = X[y == person_id]
        
        if len(person_samples) == 0:
            continue
        
        # Average across all samples
        avg_features = np.mean(person_samples, axis=0)
        
        # Create visualization
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'Gait Signature: {person_name.capitalize()}', 
                    fontsize=16, fontweight='bold')
        
        # Feature groups (assuming 528 features = 4 * 132)
        feature_size = 132
        
        # Plot 1: Mean features
        ax = axes[0, 0]
        ax.plot(avg_features[:feature_size], 'b-', linewidth=1, alpha=0.7)
        ax.set_title('Mean Temporal Features', fontsize=12, fontweight='bold')
        ax.set_xlabel('Feature Index')
        ax.set_ylabel('Value')
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Std features
        ax = axes[0, 1]
        ax.plot(avg_features[feature_size:2*feature_size], 'r-', linewidth=1, alpha=0.7)
        ax.set_title('Std Deviation Features', fontsize=12, fontweight='bold')
        ax.set_xlabel('Feature Index')
        ax.set_ylabel('Value')
        ax.grid(True, alpha=0.3)
        
        # Plot 3: Min features
        ax = axes[1, 0]
        ax.plot(avg_features[2*feature_size:3*feature_size], 'g-', linewidth=1, alpha=0.7)
        ax.set_title('Minimum Features', fontsize=12, fontweight='bold')
        ax.set_xlabel('Feature Index')
        ax.set_ylabel('Value')
        ax.grid(True, alpha=0.3)
        
        # Plot 4: Max features
        ax = axes[1, 1]
        ax.plot(avg_features[3*feature_size:], 'm-', linewidth=1, alpha=0.7)
        ax.set_title('Maximum Features', fontsize=12, fontweight='bold')
        ax.set_xlabel('Feature Index')
        ax.set_ylabel('Value')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        filename = f'data/visualizations/gait_{person_name.lower()}.png'
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Created: gait_{person_name.lower()}.png ({len(person_samples)} sample(s))")

def create_stride_analysis(X, y, id_to_name):
    """Create stride analysis comparing all people"""
    
    print("\nCreating Stride Analysis...")
    
    plt.figure(figsize=(14, 8))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(id_to_name)))
    
    for person_id, color in zip(sorted(id_to_name.keys()), colors):
        person_name = id_to_name[person_id]
        person_samples = X[y == person_id]
        
        if len(person_samples) == 0:
            continue
        
        # Use first 50 features as proxy for stride pattern
        avg_stride = np.mean(person_samples, axis=0)[:50]
        
        plt.plot(avg_stride, label=person_name.capitalize(), 
                color=color, linewidth=2, alpha=0.8)
    
    plt.title('Stride Analysis - Gait Patterns Across Individuals', 
             fontsize=14, fontweight='bold')
    plt.xlabel('Temporal Feature Index', fontsize=12)
    plt.ylabel('Feature Value', fontsize=12)
    plt.legend(loc='upper right', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig('data/visualizations/stride_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("✅ Created: stride_analysis.png")

def create_joint_trajectories():
    """Create joint trajectory visualization from a sample video"""
    
    print("\nCreating Joint Trajectories...")
    
    # Find a sample video
    video_files = [f for f in os.listdir('data') if f.endswith('.mp4')]
    
    if not video_files:
        print("⚠️  No videos found, skipping joint trajectories")
        return
    
    video_path = os.path.join('data', video_files[0])
    
    # Extract keypoints
    mp_pose = mp.solutions.pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=0.5
    )
    
    cap = cv2.VideoCapture(video_path)
    
    trajectories = {'left_ankle': [], 'right_ankle': [], 
                   'left_knee': [], 'right_knee': [],
                   'left_hip': [], 'right_hip': []}
    
    frame_count = 0
    max_frames = 60
    
    while frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_count % 2 != 0:  # Process every other frame
            frame_count += 1
            continue
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = mp_pose.process(rgb_frame)
        
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            
            # Key joint indices in MediaPipe
            trajectories['left_ankle'].append((landmarks[27].x, landmarks[27].y))
            trajectories['right_ankle'].append((landmarks[28].x, landmarks[28].y))
            trajectories['left_knee'].append((landmarks[25].x, landmarks[25].y))
            trajectories['right_knee'].append((landmarks[26].x, landmarks[26].y))
            trajectories['left_hip'].append((landmarks[23].x, landmarks[23].y))
            trajectories['right_hip'].append((landmarks[24].x, landmarks[24].y))
        
        frame_count += 1
    
    cap.release()
    
    # Plot trajectories
    plt.figure(figsize=(14, 8))
    
    for joint_name, positions in trajectories.items():
        if positions:
            x_coords = [p[0] for p in positions]
            y_coords = [1 - p[1] for p in positions]  # Invert y for proper orientation
            plt.plot(x_coords, y_coords, '-o', label=joint_name.replace('_', ' ').title(),
                    linewidth=2, markersize=3, alpha=0.7)
    
    plt.title('Joint Trajectories During Walking', fontsize=14, fontweight='bold')
    plt.xlabel('Horizontal Position (normalized)', fontsize=12)
    plt.ylabel('Vertical Position (normalized)', fontsize=12)
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.tight_layout()
    
    plt.savefig('data/visualizations/joint_trajectories.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("✅ Created: joint_trajectories.png")

def create_pca_visualization(X, y, id_to_name):
    """Create PCA visualization"""
    
    print("\nCreating PCA Visualization...")
    
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    
    plt.figure(figsize=(12, 8))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(id_to_name)))
    
    for person_id, color in zip(sorted(id_to_name.keys()), colors):
        person_name = id_to_name[person_id]
        mask = y == person_id
        
        plt.scatter(X_pca[mask, 0], X_pca[mask, 1], 
                   c=[color], label=person_name.capitalize(),
                   s=100, alpha=0.7, edgecolors='black', linewidth=0.5)
    
    plt.title('PCA: Gait Feature Space (2D Projection)', fontsize=14, fontweight='bold')
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)', fontsize=12)
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)', fontsize=12)
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig('data/visualizations/gait_pca.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("✅ Created: gait_pca.png")

def create_correlation_heatmap(X):
    """Create feature correlation heatmap"""
    
    print("\nCreating Correlation Heatmap...")
    
    # Sample features for visualization (all 528 would be too dense)
    sample_features = X[:, ::10]  # Every 10th feature
    
    corr_matrix = np.corrcoef(sample_features.T)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, cmap='coolwarm', center=0, 
                square=True, linewidths=0, cbar_kws={"shrink": 0.8})
    
    plt.title('Gait Feature Correlation Matrix (Sampled)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    plt.savefig('data/visualizations/gait_correlation.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("✅ Created: gait_correlation.png")

def remove_old_files():
    """Remove old gait pattern files"""
    
    print("\n" + "="*80)
    print("Removing Old Visualization Files")
    print("="*80)
    
    old_files = [
        'data/visualizations/gait_krees.png',
        'data/visualizations/gait_prax.png',
        'data/visualizations/gait_vastal.png',
        'data/visualizations/gait_bubbly.png'
    ]
    
    for old_file in old_files:
        if os.path.exists(old_file):
            os.remove(old_file)
            print(f"🗑️  Removed: {old_file}")

def main():
    """Main regeneration pipeline"""
    
    print("\n")
    print("█" * 80)
    print("█" + " " * 20 + "REGENERATE ALL VISUALIZATIONS" + " " * 29 + "█")
    print("█" * 80)
    
    # Remove old files
    remove_old_files()
    
    # Load data
    print("\n📊 Loading data...")
    X, y, labels, id_to_name = load_data()
    print(f"✅ Loaded {len(X)} samples for {len(labels)} people")
    
    # Create all visualizations
    create_individual_gait_patterns(X, y, id_to_name)
    create_stride_analysis(X, y, id_to_name)
    create_joint_trajectories()
    create_pca_visualization(X, y, id_to_name)
    create_correlation_heatmap(X)
    
    print("\n" + "="*80)
    print("✅ ALL VISUALIZATIONS REGENERATED!")
    print("="*80)
    
    print("\n📁 Check: data/visualizations/")
    print("\n📋 Generated files:")
    print("   • Individual gait patterns (9 people)")
    print("   • stride_analysis.png")
    print("   • joint_trajectories.png")
    print("   • gait_pca.png")
    print("   • gait_correlation.png")
    print("\n" + "="*80)

if __name__ == "__main__":
    main()
