"""
Gait Visualization Menu
Run this script to choose what kind of gait visualization you want to see.
"""

import os
import subprocess
import sys

def main():
    print("=" * 50)
    print("    GAIT VISUALIZATION MENU")
    print("=" * 50)
    print()
    print("Choose what you want to visualize:")
    print()
    print("1. Animated gait visualization (interactive)")
    print("2. Static gait patterns (saved as images)")
    print("3. Comprehensive gait analysis")
    print("4. View existing visualizations")
    print("5. Generate all visualizations")
    print("0. Exit")
    print()
    
    choice = input("Enter your choice (0-5): ").strip()
    
    if choice == "1":
        print("\nStarting animated gait visualization...")
        print("You'll be able to choose which person to visualize.")
        run_script("src/utils/visualize_gait.py")
        
    elif choice == "2":
        print("\nGenerating static gait visualizations...")
        generate_static_visualizations()
        
    elif choice == "3":
        print("\nRunning comprehensive gait analysis...")
        run_script("src/utils/analyze_gait.py")
        
    elif choice == "4":
        print("\nListing existing visualizations:")
        list_visualizations()
        
    elif choice == "5":
        print("\nGenerating all visualizations...")
        generate_static_visualizations()
        run_script("src/utils/analyze_gait.py")
        print("\nAll visualizations complete!")
        list_visualizations()
        
    elif choice == "0":
        print("Goodbye!")
        return
        
    else:
        print("Invalid choice. Please try again.")
        main()

def generate_static_visualizations():
    """Generate static visualization images directly in this script"""
    try:
        import numpy as np
        import json
        import matplotlib.pyplot as plt
        
        # Load data
        X = np.load("data/processed/X.npy")
        y = np.load("data/processed/y.npy")
        
        with open("data/processed/labels.json") as f:
            labels_json = json.load(f)
        
        label_map = {v: k for k, v in labels_json.items()}
        
        print(f"Data shape: {X.shape}")
        print(f"Available people: {list(labels_json.keys())}")
        
        # Create visualizations for all people
        for person_id, person_name in label_map.items():
            create_person_visualization(X, y, person_id, person_name, label_map)
        
        create_trajectory_plot(X, y, label_map)
        print("Static visualizations complete!")
        
    except Exception as e:
        print(f"Error generating static visualizations: {e}")

def create_person_visualization(X, y, person_id, person_name, label_map):
    """Create visualization for a single person"""
    import numpy as np
    import matplotlib.pyplot as plt
    
    sample_sequence = X[person_id]
    n_frames = sample_sequence.shape[0]
    
    # Extract x,y coordinates (first 66 features)
    coords_data = sample_sequence[:, :66]
    x_coords = coords_data[:, 0::2]
    y_coords = coords_data[:, 1::2]
    
    # MediaPipe pose connections
    pose_connections = [
        (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8),
        (9, 10), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
        (11, 23), (12, 24), (23, 24), (23, 25), (25, 27), (27, 29), (27, 31),
        (24, 26), (26, 28), (28, 30), (28, 32),
    ]
    
    # Create subplots for multiple frames
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle(f'Gait Sequence: {person_name}', fontsize=16)
    
    # Select 8 evenly spaced frames
    frame_indices = np.linspace(0, n_frames-1, 8, dtype=int)
    
    for idx, frame_idx in enumerate(frame_indices):
        row = idx // 4
        col = idx % 4
        ax = axes[row, col]
        
        x_frame = x_coords[frame_idx]
        y_frame = -y_coords[frame_idx]  # Flip y-axis
        
        # Plot joints and skeleton
        ax.scatter(x_frame, y_frame, s=30, c='red', alpha=0.7, zorder=3)
        
        for i, j in pose_connections:
            if i < len(x_frame) and j < len(x_frame):
                ax.plot([x_frame[i], x_frame[j]], [y_frame[i], y_frame[j]], 
                       'b-', alpha=0.6, linewidth=1.5, zorder=2)
        
        ax.set_xlim(-2, 2)
        ax.set_ylim(-2, 2)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_title(f'Frame {frame_idx + 1}')
    
    plt.tight_layout()
    os.makedirs("data/visualizations", exist_ok=True)
    output_path = f"data/visualizations/gait_{person_name}.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()

def create_trajectory_plot(X, y, label_map):
    """Create trajectory plot for key joints"""
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Joint Trajectories Over Time', fontsize=16)
    
    key_joints = {
        'Left Hip': 23, 'Right Hip': 24, 'Left Knee': 25, 
        'Right Knee': 26, 'Left Ankle': 27, 'Right Ankle': 28
    }
    
    for person_id, person_name in label_map.items():
        sample_sequence = X[person_id]
        coords_data = sample_sequence[:, :66]
        x_coords = coords_data[:, 0::2]
        y_coords = coords_data[:, 1::2]
        
        for idx, (joint_name, joint_idx) in enumerate(key_joints.items()):
            row = idx // 3
            col = idx % 3
            ax = axes[row, col]
            
            ax.plot(x_coords[:, joint_idx], -y_coords[:, joint_idx], 
                   label=person_name, marker='o', markersize=2, alpha=0.7)
            
            ax.set_title(f'{joint_name} Trajectory')
            ax.set_xlabel('X Coordinate')
            ax.set_ylabel('Y Coordinate')
            ax.grid(True, alpha=0.3)
            ax.legend()
    
    plt.tight_layout()
    os.makedirs("data/visualizations", exist_ok=True)
    plt.savefig("data/visualizations/joint_trajectories.png", dpi=150, bbox_inches='tight')
    print("Saved: data/visualizations/joint_trajectories.png")
    plt.close()

def run_script(script_path):
    """Run a Python script"""
    try:
        result = subprocess.run([sys.executable, script_path], 
                              capture_output=False, 
                              text=True, 
                              cwd=os.getcwd())
        if result.returncode != 0:
            print(f"Error running {script_path}")
    except Exception as e:
        print(f"Error: {e}")

def list_visualizations():
    """List all available visualization files"""
    viz_dir = "data/visualizations"
    if os.path.exists(viz_dir):
        files = os.listdir(viz_dir)
        if files:
            print(f"\nVisualization files in {viz_dir}:")
            for file in sorted(files):
                file_path = os.path.join(viz_dir, file)
                file_size = os.path.getsize(file_path) / 1024  # KB
                print(f"  📊 {file} ({file_size:.1f} KB)")
            print(f"\nTotal: {len(files)} visualization files")
            print("\nYou can open these PNG files with any image viewer.")
        else:
            print(f"\nNo visualizations found in {viz_dir}")
            print("Run option 2 or 5 to generate visualizations.")
    else:
        print(f"\nVisualization directory {viz_dir} doesn't exist yet.")
        print("Run option 2 or 5 to generate visualizations.")

if __name__ == "__main__":
    # Change to project directory if not already there
    if not os.path.exists("data/processed/X.npy"):
        print("Error: Could not find processed gait data.")
        print("Make sure you're running this from the project root directory.")
        print("Current directory:", os.getcwd())
        sys.exit(1)
    
    main()