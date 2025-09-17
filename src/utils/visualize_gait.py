import numpy as np
import json
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Paths
X_path = "data/processed/X.npy"
y_path = "data/processed/y.npy"
labels_path = "data/processed/labels.json"

# Load data
X = np.load(X_path)
y = np.load(y_path)

with open(labels_path) as f:
    labels_json = json.load(f)

# Build reverse mapping: numeric ID → name
# This works even if labels.json has { "gait": 0, "other": 1 }
label_map = {v: k for k, v in labels_json.items()}

# Pick a sample to visualize
sample_idx = 0
sample = X[sample_idx]
label_id = y[sample_idx]
label_name = label_map.get(label_id, f"Person_{label_id}")

# Assuming 33 keypoints (MediaPipe Pose)
num_joints = int(len(sample) / 3)
x_coords = sample[0::3]
y_coords = sample[1::3]

# Plot setup
fig, ax = plt.subplots()
ax.set_title(f"Gait Visualization: {label_name}")
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
scat = ax.scatter(x_coords, y_coords)

# Example connections between joints
connections = [
    (0, 1), (1, 2), (2, 3), (0, 4), (4, 5), (5, 6),
    (11, 12), (12, 13), (13, 14), (11, 15), (15, 16)
]
lines = [ax.plot([], [], 'b-')[0] for _ in connections]

def update(frame):
    # For now we have only 1 sample; later could loop multiple samples
    sample_frame = X[frame] if len(X.shape) > 1 else sample
    x = sample_frame[0::3]
    y_ = sample_frame[1::3]
    scat.set_offsets(np.c_[x, y_])
    for idx, (i, j) in enumerate(connections):
        lines[idx].set_data([x[i], x[j]], [y_[i], y_[j]])
    return scat, *lines

anim = FuncAnimation(fig, update, frames=min(50, X.shape[0]), interval=100, blit=True)
plt.show()
