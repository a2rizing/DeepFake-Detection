import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def analyze_gait_patterns():
    """Analyze and compare gait patterns between different people"""
    
    # Load data
    X = np.load("data/processed/X.npy")  # Shape: (N_samples, N_frames, N_features)
    y = np.load("data/processed/y.npy")
    
    with open("data/processed/labels.json") as f:
        labels_json = json.load(f)
    
    label_map = {v: k for k, v in labels_json.items()}
    
    print("=== Gait Analysis Report ===")
    print(f"Dataset shape: {X.shape}")
    print(f"Number of people: {len(label_map)}")
    print(f"Frames per sequence: {X.shape[1]}")
    print(f"Features per frame: {X.shape[2]}")
    print()
    
    # 1. Basic statistics
    print("--- Basic Statistics ---")
    for person_id, person_name in label_map.items():
        sequence = X[person_id]
        coords = sequence[:, :66]  # x,y coordinates only
        angles = sequence[:, 66:70]  # joint angles
        
        print(f"{person_name}:")
        print(f"  Avg movement range (x): {np.std(coords[:, 0::2]):.4f}")
        print(f"  Avg movement range (y): {np.std(coords[:, 1::2]):.4f}")
        print(f"  Avg joint angle variation: {np.std(angles):.4f}")
        print()
    
    # 2. Stride analysis (using ankle positions)
    print("--- Stride Analysis ---")
    left_ankle_idx = 27 * 2  # x coordinate of left ankle (27th landmark * 2 for x)
    right_ankle_idx = 28 * 2  # x coordinate of right ankle
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle('Ankle Movement Patterns (X-axis)', fontsize=14)
    
    for person_id, person_name in label_map.items():
        sequence = X[person_id]
        left_ankle_x = sequence[:, left_ankle_idx]
        right_ankle_x = sequence[:, right_ankle_idx]
        
        # Calculate stride frequency (simplified)
        left_peaks = find_peaks_simple(left_ankle_x)
        right_peaks = find_peaks_simple(right_ankle_x)
        
        print(f"{person_name}:")
        print(f"  Left ankle peaks: {len(left_peaks)}")
        print(f"  Right ankle peaks: {len(right_peaks)}")
        print(f"  Estimated stride frequency: {(len(left_peaks) + len(right_peaks)) / 2:.1f} steps")
        
        # Plot ankle movements
        row = person_id // 3
        col = person_id % 3
        ax = axes[row, col]
        
        frames = np.arange(len(left_ankle_x))
        ax.plot(frames, left_ankle_x, label='Left Ankle', color='blue', alpha=0.7)
        ax.plot(frames, right_ankle_x, label='Right Ankle', color='red', alpha=0.7)
        ax.scatter(left_peaks, left_ankle_x[left_peaks], color='blue', s=50, zorder=5)
        ax.scatter(right_peaks, right_ankle_x[right_peaks], color='red', s=50, zorder=5)
        
        ax.set_title(f'{person_name}')
        ax.set_xlabel('Frame')
        ax.set_ylabel('X Position')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("data/visualizations/stride_analysis.png", dpi=150, bbox_inches='tight')
    print("\nStride analysis plot saved: data/visualizations/stride_analysis.png")
    plt.close()
    
    # 3. PCA Analysis for gait signature comparison
    print("\n--- Gait Signature Analysis (PCA) ---")
    
    # Flatten sequences for PCA
    X_flat = X.reshape(X.shape[0], -1)  # (n_people, n_frames * n_features)
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_flat)
    
    # Apply PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    # Plot PCA results
    plt.figure(figsize=(10, 8))
    colors = plt.cm.Set3(np.linspace(0, 1, len(label_map)))
    
    for i, (person_id, person_name) in enumerate(label_map.items()):
        plt.scatter(X_pca[person_id, 0], X_pca[person_id, 1], 
                   c=[colors[i]], s=200, label=person_name, alpha=0.8)
        plt.annotate(person_name, (X_pca[person_id, 0], X_pca[person_id, 1]),
                    xytext=(10, 10), textcoords='offset points', fontsize=10)
    
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
    plt.title('Gait Signature Comparison (PCA)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.savefig("data/visualizations/gait_pca.png", dpi=150, bbox_inches='tight')
    print("PCA analysis plot saved: data/visualizations/gait_pca.png")
    plt.close()
    
    # 4. Feature correlation heatmap
    print("\n--- Feature Correlation Analysis ---")
    
    # Calculate correlation between people based on average features
    n_people = len(label_map)
    correlation_matrix = np.zeros((n_people, n_people))
    
    for i in range(n_people):
        for j in range(n_people):
            # Calculate correlation between flattened sequences
            corr = np.corrcoef(X_flat[i], X_flat[j])[0, 1]
            correlation_matrix[i, j] = corr
    
    # Plot correlation heatmap
    plt.figure(figsize=(8, 6))
    person_names = [label_map[i] for i in range(n_people)]
    
    sns.heatmap(correlation_matrix, 
                xticklabels=person_names,
                yticklabels=person_names,
                annot=True, 
                cmap='coolwarm', 
                center=0,
                fmt='.2f')
    
    plt.title('Gait Pattern Similarity Matrix')
    plt.tight_layout()
    plt.savefig("data/visualizations/gait_correlation.png", dpi=150, bbox_inches='tight')
    print("Correlation heatmap saved: data/visualizations/gait_correlation.png")
    plt.close()
    
    print("\n=== Analysis Complete ===")
    print("Check 'data/visualizations/' folder for all generated plots:")
    print("- Individual gait patterns: gait_[person].png")
    print("- Joint trajectories: joint_trajectories.png") 
    print("- Stride analysis: stride_analysis.png")
    print("- PCA comparison: gait_pca.png")
    print("- Similarity matrix: gait_correlation.png")

def find_peaks_simple(signal, threshold=None):
    """Simple peak detection"""
    if threshold is None:
        threshold = np.mean(signal)
    
    peaks = []
    for i in range(1, len(signal) - 1):
        if (signal[i] > signal[i-1] and 
            signal[i] > signal[i+1] and 
            signal[i] > threshold):
            peaks.append(i)
    
    return np.array(peaks)

if __name__ == "__main__":
    analyze_gait_patterns()