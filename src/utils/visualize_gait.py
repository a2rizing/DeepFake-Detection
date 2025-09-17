import numpy as np
import json
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os

# Paths
X_path = "data/processed/X.npy"
y_path = "data/processed/y.npy"
labels_path = "data/processed/labels.json"

# Load data
X = np.load(X_path)  # Shape: (N_samples, N_frames, N_features)
y = np.load(y_path)

with open(labels_path) as f:
    labels_json = json.load(f)

print(f"Data shape: {X.shape}")
print(f"Labels: {labels_json}")

# Build reverse mapping: numeric ID → name
label_map = {v: k for k, v in labels_json.items()}

# Let user choose which person to visualize
print("\nAvailable people:")
for person_id, person_name in label_map.items():
    print(f"  {person_id}: {person_name}")

sample_idx = int(input(f"\nChoose person to visualize (0-{len(label_map)-1}): "))
if sample_idx < 0 or sample_idx >= len(X):
    sample_idx = 0
    print(f"Invalid choice, using person {sample_idx}")

sample_sequence = X[sample_idx]  # Shape: (N_frames, N_features)
label_id = y[sample_idx]
label_name = label_map.get(label_id, f"Person_{label_id}")

print(f"Visualizing: {label_name}")
print(f"Sequence shape: {sample_sequence.shape}")

# Extract x,y coordinates from features
# Feature format: [x_0, y_0, x_1, y_1, ..., x_32, y_32, angle_0, angle_1, angle_2, angle_3]
# So we have 33 landmarks * 2 = 66 coordinate features, plus 4 angle features = 70 total
n_landmarks = 33
n_frames = sample_sequence.shape[0]

# Extract x,y coordinates (first 66 features)
coords_data = sample_sequence[:, :66]  # (n_frames, 66)
x_coords = coords_data[:, 0::2]  # (n_frames, 33) - every other starting from 0
y_coords = coords_data[:, 1::2]  # (n_frames, 33) - every other starting from 1

# MediaPipe pose connections for skeleton visualization
pose_connections = [
    # Face connections
    (0, 1), (1, 2), (2, 3), (3, 7),
    (0, 4), (4, 5), (5, 6), (6, 8),
    # Body connections
    (9, 10),  # mouth to mouth
    (11, 12),  # shoulder to shoulder
    (11, 13), (13, 15),  # left arm
    (12, 14), (14, 16),  # right arm
    (11, 23), (12, 24),  # shoulders to hips
    (23, 24),  # hip to hip
    (23, 25), (25, 27), (27, 29), (27, 31),  # left leg
    (24, 26), (26, 28), (28, 30), (28, 32),  # right leg
]

# Plot setup
fig, ax = plt.subplots(figsize=(10, 8))
ax.set_title(f"Gait Visualization: {label_name}")
ax.set_xlim(-2, 2)  # Adjusted for normalized coordinates
ax.set_ylim(-2, 2)
ax.set_aspect('equal')
ax.grid(True, alpha=0.3)

# Initialize scatter plot for joints
scat = ax.scatter([], [], s=50, c='red', alpha=0.7)

# Initialize lines for skeleton connections
lines = [ax.plot([], [], 'b-', alpha=0.6, linewidth=2)[0] for _ in pose_connections]

# Text to show current frame
frame_text = ax.text(0.02, 0.98, '', transform=ax.transAxes, fontsize=12,
                     verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

def update(frame_idx):
    # Get coordinates for current frame
    x_frame = x_coords[frame_idx]  # (33,)
    y_frame = y_coords[frame_idx]  # (33,)
    
    # Flip y-axis for proper visualization (MediaPipe uses top-left origin)
    y_frame_flipped = -y_frame
    
    # Update scatter plot
    scat.set_offsets(np.column_stack([x_frame, y_frame_flipped]))
    
    # Update skeleton lines
    for line_idx, (i, j) in enumerate(pose_connections):
        if i < len(x_frame) and j < len(x_frame):  # Safety check
            lines[line_idx].set_data([x_frame[i], x_frame[j]], 
                                   [y_frame_flipped[i], y_frame_flipped[j]])
    
    # Update frame counter
    frame_text.set_text(f'Frame: {frame_idx + 1}/{n_frames}')
    
    return [scat] + lines + [frame_text]

print(f"Creating animation with {n_frames} frames...")
anim = FuncAnimation(fig, update, frames=n_frames, interval=100, blit=True, repeat=True)

# Show the plot
plt.tight_layout()
plt.show()

# Optionally save as GIF (uncomment if needed)
# print("Saving animation as GIF...")
# anim.save(f'gait_animation_{label_name}.gif', writer='pillow', fps=10)
