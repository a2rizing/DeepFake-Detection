#!/usr/bin/env python3
"""
Visualization Tools for DeepFake Detection
Interactive menu for visualizing gait patterns and model results

Usage:
    python visualize.py           # Interactive menu
    python visualize.py --gait    # Gait animation
    python visualize.py --pca     # PCA analysis
"""

import argparse
import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.animation import FuncAnimation
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)


def load_data():
    """Load processed data"""
    data_path = "data/processed"
    
    if not os.path.exists(os.path.join(data_path, "X.npy")):
        print("❌ Processed data not found!")
        print("   Run: python train.py first")
        return None, None, None
    
    X = np.load(os.path.join(data_path, "X.npy"))
    y = np.load(os.path.join(data_path, "y.npy"))
    
    with open(os.path.join(data_path, "labels.json")) as f:
        labels = json.load(f)
    
    id_to_name = {v: k for k, v in labels.items()}
    
    return X, y, id_to_name


def visualize_gait_animation():
    """Interactive gait skeleton animation"""
    csv_path = "data/gait_keypoints.csv"
    
    if not os.path.exists(csv_path):
        print("❌ Keypoints CSV not found!")
        print("   Run: python src/preprocessing/extract_gait.py")
        return
    
    print("Loading gait keypoints...")
    df = pd.read_csv(csv_path)
    
    # Get unique video names
    video_names = df['video_name'].unique()
    
    print(f"\n📹 Found {len(video_names)} videos:")
    for i, name in enumerate(video_names[:20]):  # Show first 20
        frame_count = len(df[df['video_name'] == name])
        print(f"  {i}: {name} ({frame_count} frames)")
    
    if len(video_names) > 20:
        print(f"  ... and {len(video_names) - 20} more")
    
    # Get user choice
    try:
        video_idx = int(input(f"\nChoose video (0-{len(video_names)-1}): "))
        if video_idx < 0 or video_idx >= len(video_names):
            video_idx = 0
    except ValueError:
        video_idx = 0
    
    selected_video = video_names[video_idx]
    print(f"\n🎬 Visualizing: {selected_video}")
    
    # Extract frames
    video_df = df[df['video_name'] == selected_video].sort_values('frame')
    n_frames = len(video_df)
    
    # Extract coordinates
    n_landmarks = 33
    x_coords = np.zeros((n_frames, n_landmarks))
    y_coords = np.zeros((n_frames, n_landmarks))
    
    for frame_idx, row in enumerate(video_df.itertuples()):
        for lm_idx in range(n_landmarks):
            x_coords[frame_idx, lm_idx] = getattr(row, f'x_{lm_idx}')
            y_coords[frame_idx, lm_idx] = getattr(row, f'y_{lm_idx}')
    
    # MediaPipe pose connections
    pose_connections = [
        # Face
        (0, 1), (1, 2), (2, 3), (3, 7),
        (0, 4), (4, 5), (5, 6), (6, 8),
        # Torso
        (9, 10),
        (11, 12), (11, 23), (12, 24), (23, 24),
        # Arms
        (11, 13), (13, 15),  # left arm
        (12, 14), (14, 16),  # right arm
        # Legs
        (23, 25), (25, 27), (27, 29), (27, 31),  # left leg
        (24, 26), (26, 28), (28, 30), (28, 32),  # right leg
    ]
    
    # Create animation
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_xlim(0, 1)
    ax.set_ylim(1, 0)  # Inverted for image coordinates
    ax.set_aspect('equal')
    ax.set_title(f"Gait Visualization: {selected_video}")
    ax.grid(True, alpha=0.3)
    
    # Initialize lines and points
    lines = []
    for _ in pose_connections:
        line, = ax.plot([], [], 'b-', lw=2, alpha=0.7)
        lines.append(line)
    
    points, = ax.plot([], [], 'ro', markersize=6)
    
    def init():
        for line in lines:
            line.set_data([], [])
        points.set_data([], [])
        return lines + [points]
    
    def animate(frame_num):
        x = x_coords[frame_num]
        y = y_coords[frame_num]
        
        for i, (start, end) in enumerate(pose_connections):
            lines[i].set_data([x[start], x[end]], [y[start], y[end]])
        
        points.set_data(x, y)
        ax.set_title(f"Gait: {selected_video} - Frame {frame_num+1}/{n_frames}")
        
        return lines + [points]
    
    print("\n▶️ Starting animation...")
    print("   Close the window when done.\n")
    
    anim = FuncAnimation(
        fig, animate, init_func=init,
        frames=n_frames, interval=50, blit=True, repeat=True
    )
    
    plt.tight_layout()
    plt.show()
    print("Animation closed.")


def visualize_pca():
    """PCA visualization of gait signatures"""
    X, y, id_to_name = load_data()
    
    if X is None:
        return
    
    print("\n📊 Generating PCA visualization...")
    
    # Flatten sequences
    X_flat = X.reshape(X.shape[0], -1)
    
    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_flat)
    
    # Apply PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    # Plot
    plt.figure(figsize=(12, 10))
    
    # Get unique labels and colors
    unique_labels = np.unique(y)
    colors = plt.cm.Set3(np.linspace(0, 1, len(unique_labels)))
    
    for i, label in enumerate(unique_labels):
        mask = y == label
        name = id_to_name.get(label, f"Person_{label}")
        plt.scatter(X_pca[mask, 0], X_pca[mask, 1], 
                   c=[colors[i]], s=100, label=name, alpha=0.7)
        
        # Add annotation
        for j, (is_match) in enumerate(mask):
            if is_match:
                plt.annotate(name, (X_pca[j, 0], X_pca[j, 1]),
                           xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
    plt.title('Gait Signature Analysis (PCA)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save
    output_path = "data/visualizations/gait_pca.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✅ Saved: {output_path}")
    
    plt.show()


def visualize_correlation():
    """Correlation heatmap between gait patterns"""
    X, y, id_to_name = load_data()
    
    if X is None:
        return
    
    print("\n📊 Generating correlation heatmap...")
    
    # Flatten sequences
    X_flat = X.reshape(X.shape[0], -1)
    
    # Group samples by person and calculate mean
    unique_labels = np.unique(y)
    person_features = {}
    
    for label in unique_labels:
        mask = y == label
        name = id_to_name.get(label, f"Person_{label}")
        person_features[name] = np.mean(X_flat[mask], axis=0)
    
    # Calculate correlation matrix
    names = list(person_features.keys())
    n_people = len(names)
    correlation_matrix = np.zeros((n_people, n_people))
    
    for i, name1 in enumerate(names):
        for j, name2 in enumerate(names):
            corr = np.corrcoef(person_features[name1], person_features[name2])[0, 1]
            correlation_matrix[i, j] = corr
    
    # Plot
    plt.figure(figsize=(12, 10))
    sns.heatmap(correlation_matrix, 
                xticklabels=names,
                yticklabels=names,
                annot=True, 
                cmap='coolwarm', 
                center=0,
                fmt='.2f',
                square=True)
    
    plt.title('Gait Pattern Similarity Matrix')
    plt.tight_layout()
    
    # Save
    output_path = "data/visualizations/gait_correlation.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✅ Saved: {output_path}")
    
    plt.show()


def visualize_individual_gait():
    """Individual gait pattern visualization"""
    X, y, id_to_name = load_data()
    
    if X is None:
        return
    
    print("\n📊 Generating individual gait patterns...")
    
    output_dir = "data/visualizations"
    os.makedirs(output_dir, exist_ok=True)
    
    unique_labels = np.unique(y)
    
    for label in unique_labels:
        mask = y == label
        name = id_to_name.get(label, f"Person_{label}")
        
        # Get first sample for this person
        idx = np.where(mask)[0][0]
        sequence = X[idx]
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'Gait Pattern: {name}', fontsize=14)
        
        # Plot 1: X coordinates over time
        ax1 = axes[0, 0]
        for i in range(0, 66, 2):  # Every x coordinate
            ax1.plot(sequence[:, i], alpha=0.5)
        ax1.set_title('X Coordinates Over Time')
        ax1.set_xlabel('Frame')
        ax1.set_ylabel('X Position')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Y coordinates over time
        ax2 = axes[0, 1]
        for i in range(1, 66, 2):  # Every y coordinate
            ax2.plot(sequence[:, i], alpha=0.5)
        ax2.set_title('Y Coordinates Over Time')
        ax2.set_xlabel('Frame')
        ax2.set_ylabel('Y Position')
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Ankle movement (key for gait)
        ax3 = axes[1, 0]
        left_ankle_x = sequence[:, 27*2]
        right_ankle_x = sequence[:, 28*2]
        ax3.plot(left_ankle_x, label='Left Ankle', color='blue')
        ax3.plot(right_ankle_x, label='Right Ankle', color='red')
        ax3.set_title('Ankle Movement (X-axis)')
        ax3.set_xlabel('Frame')
        ax3.set_ylabel('X Position')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Joint angles
        ax4 = axes[1, 1]
        if sequence.shape[1] > 66:
            angles = sequence[:, 66:]
            for i, angle_name in enumerate(['Left Knee', 'Right Knee', 'Left Elbow', 'Right Elbow']):
                if i < angles.shape[1]:
                    ax4.plot(angles[:, i], label=angle_name, alpha=0.7)
            ax4.set_title('Joint Angles Over Time')
            ax4.set_xlabel('Frame')
            ax4.set_ylabel('Angle (radians)')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        else:
            ax4.text(0.5, 0.5, 'No angle data', ha='center', va='center')
        
        plt.tight_layout()
        
        output_path = os.path.join(output_dir, f'gait_{name.lower()}.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"   ✅ Saved: gait_{name.lower()}.png")
    
    print(f"\n✅ All patterns saved to: {output_dir}")


def show_menu():
    """Interactive visualization menu"""
    print("\n" + "=" * 60)
    print("🎨 GAIT VISUALIZATION TOOLS")
    print("=" * 60)
    print("\n📊 Available visualizations:")
    print("   1. Gait skeleton animation")
    print("   2. PCA analysis (gait signatures)")
    print("   3. Correlation heatmap")
    print("   4. Individual gait patterns")
    print("   5. Generate all visualizations")
    print("   0. Exit")
    
    try:
        choice = input("\nEnter choice (0-5): ").strip()
    except KeyboardInterrupt:
        return
    
    if choice == '1':
        visualize_gait_animation()
    elif choice == '2':
        visualize_pca()
    elif choice == '3':
        visualize_correlation()
    elif choice == '4':
        visualize_individual_gait()
    elif choice == '5':
        visualize_pca()
        visualize_correlation()
        visualize_individual_gait()
        print("\n✅ All visualizations generated!")
    elif choice == '0':
        print("Goodbye!")
        return
    else:
        print("Invalid choice!")
    
    # Show menu again
    show_menu()


def main():
    parser = argparse.ArgumentParser(
        description="Visualization Tools for DeepFake Detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python visualize.py               # Interactive menu
  python visualize.py --gait        # Gait animation
  python visualize.py --pca         # PCA analysis
  python visualize.py --all         # Generate all visualizations
        """
    )
    
    parser.add_argument('--gait', '-g', action='store_true',
                        help='Show gait skeleton animation')
    parser.add_argument('--pca', '-p', action='store_true',
                        help='Generate PCA visualization')
    parser.add_argument('--correlation', '-c', action='store_true',
                        help='Generate correlation heatmap')
    parser.add_argument('--individual', '-i', action='store_true',
                        help='Generate individual gait patterns')
    parser.add_argument('--all', '-a', action='store_true',
                        help='Generate all visualizations')
    
    args = parser.parse_args()
    
    # Check for specific visualizations
    if args.gait:
        visualize_gait_animation()
    elif args.pca:
        visualize_pca()
    elif args.correlation:
        visualize_correlation()
    elif args.individual:
        visualize_individual_gait()
    elif args.all:
        visualize_pca()
        visualize_correlation()
        visualize_individual_gait()
        print("\n✅ All visualizations generated!")
    else:
        # Show interactive menu
        show_menu()


if __name__ == "__main__":
    main()
