import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os

# Load gait keypoints CSV
csv_path = "data/gait_keypoints.csv"

if not os.path.exists(csv_path):
    print(f"Error: {csv_path} not found!")
    print("Run regenerate_keypoints_csv.py first to generate keypoints.")
    exit(1)

print("Loading gait keypoints from CSV...")
df = pd.read_csv(csv_path)

# Get unique video names
video_names = df['video_name'].unique()

print(f"\nFound {len(video_names)} videos:")
for i, name in enumerate(video_names):
    frame_count = len(df[df['video_name'] == name])
    print(f"  {i}: {name} ({frame_count} frames)")

# Let user choose which video to visualize
video_idx = int(input(f"\nChoose video to visualize (0-{len(video_names)-1}): "))
if video_idx < 0 or video_idx >= len(video_names):
    video_idx = 0
    print(f"Invalid choice, using video {video_idx}")

selected_video = video_names[video_idx]
print(f"\nVisualizing: {selected_video}")

# Extract frames for selected video
video_df = df[df['video_name'] == selected_video].sort_values('frame')
n_frames = len(video_df)

print(f"Total frames: {n_frames}")

# Extract x,y coordinates for all frames
n_landmarks = 33
x_coords = np.zeros((n_frames, n_landmarks))
y_coords = np.zeros((n_frames, n_landmarks))

for frame_idx, row in enumerate(video_df.itertuples()):
    for lm_idx in range(n_landmarks):
        x_coords[frame_idx, lm_idx] = getattr(row, f'x_{lm_idx}')
        y_coords[frame_idx, lm_idx] = getattr(row, f'y_{lm_idx}')

print(f"Extracted coordinates: {x_coords.shape}")

# MediaPipe pose connections for skeleton visualization
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

# Create figure and axis
fig, ax = plt.subplots(figsize=(10, 10))
ax.set_xlim(0, 1)
ax.set_ylim(1, 0)  # Inverted for image coordinates
ax.set_aspect('equal')
ax.set_title(f"Gait Visualization: {selected_video}")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.grid(True, alpha=0.3)

# Plot lines for skeleton
lines = []
for connection in pose_connections:
    line, = ax.plot([], [], 'b-', lw=2, alpha=0.7)
    lines.append(line)

# Plot points for landmarks
points, = ax.plot([], [], 'ro', markersize=6)

def init():
    """Initialize animation"""
    for line in lines:
        line.set_data([], [])
    points.set_data([], [])
    return lines + [points]

def animate(frame_num):
    """Update animation for each frame"""
    # Get coordinates for this frame
    x = x_coords[frame_num]
    y = y_coords[frame_num]
    
    # Update skeleton lines
    for i, (start, end) in enumerate(pose_connections):
        lines[i].set_data([x[start], x[end]], [y[start], y[end]])
    
    # Update landmark points
    points.set_data(x, y)
    
    ax.set_title(f"Gait: {selected_video} - Frame {frame_num+1}/{n_frames}")
    
    return lines + [points]

# Create animation
print("\nCreating animation...")
print("Close the window when done.\n")

anim = FuncAnimation(
    fig, animate, init_func=init,
    frames=n_frames, interval=50, blit=True, repeat=True
)

plt.tight_layout()
plt.show()

print("Animation closed.")
