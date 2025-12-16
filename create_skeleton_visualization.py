"""
Create Skeleton/Pose Visualization
Shows MediaPipe keypoint connections overlaid on walking sequences
"""

import cv2
import numpy as np
import mediapipe as mp
import os
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import matplotlib.lines as mlines

def create_skeleton_visualization():
    """Create skeleton overlay visualization from sample video"""
    
    print("\n" + "="*80)
    print("CREATING SKELETON/POSE VISUALIZATION")
    print("="*80)
    
    # Find a sample video
    video_files = [f for f in os.listdir('data') if f.endswith('.mp4')]
    
    if not video_files:
        print("❌ No videos found in data/ folder")
        return
    
    video_path = os.path.join('data', video_files[0])
    print(f"\n📹 Using video: {video_files[0]}")
    
    # Initialize MediaPipe
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"❌ Cannot open video: {video_path}")
        return
    
    # Extract frames with skeleton overlay
    frames_with_skeleton = []
    frame_count = 0
    max_frames = 6  # Get 6 frames for visualization
    
    print("\n🎬 Extracting frames with pose overlay...")
    
    while len(frames_with_skeleton) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Skip frames for better variety
        if frame_count % 10 != 0:
            frame_count += 1
            continue
        
        # Convert to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process with MediaPipe
        results = pose.process(rgb_frame)
        
        if results.pose_landmarks:
            # Draw skeleton on frame
            annotated_frame = rgb_frame.copy()
            mp_drawing.draw_landmarks(
                annotated_frame,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
            )
            
            frames_with_skeleton.append(annotated_frame)
            print(f"   ✅ Frame {len(frames_with_skeleton)}/6 extracted")
        
        frame_count += 1
    
    cap.release()
    
    if len(frames_with_skeleton) == 0:
        print("❌ No pose detected in video")
        return
    
    # Create visualization
    print("\n📊 Creating skeleton visualization...")
    
    # Determine grid layout
    n_frames = len(frames_with_skeleton)
    if n_frames <= 3:
        rows, cols = 1, n_frames
    else:
        rows, cols = 2, 3
    
    fig, axes = plt.subplots(rows, cols, figsize=(16, 10))
    fig.suptitle('MediaPipe Pose Estimation: Skeleton Overlay on Walking Sequence', 
                 fontsize=16, fontweight='bold')
    
    # Flatten axes for easier iteration
    if n_frames == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if hasattr(axes, 'flatten') else axes
    
    for idx, (ax, frame) in enumerate(zip(axes, frames_with_skeleton)):
        ax.imshow(frame)
        ax.set_title(f'Frame {idx+1}', fontsize=12, fontweight='bold')
        ax.axis('off')
    
    # Hide unused subplots
    for idx in range(n_frames, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    
    # Save
    os.makedirs('data/visualizations', exist_ok=True)
    output_path = 'data/visualizations/skeleton_pose_overlay.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n✅ Saved: {output_path}")
    
    # Create detailed landmark diagram
    create_landmark_diagram()
    
    print("\n" + "="*80)
    print("✅ SKELETON VISUALIZATIONS COMPLETE!")
    print("="*80)
    print("\n📁 Created files:")
    print("   • skeleton_pose_overlay.png - Actual pose detection on video frames")
    print("   • landmark_diagram.png - MediaPipe 33-point landmark reference")
    print("\n" + "="*80)

def create_landmark_diagram():
    """Create a diagram showing MediaPipe's 33 landmarks"""
    
    print("\n📊 Creating landmark reference diagram...")
    
    # MediaPipe landmark connections
    connections = [
        # Face
        (0, 1), (1, 2), (2, 3), (3, 7),
        (0, 4), (4, 5), (5, 6), (6, 8),
        (9, 10),
        # Torso
        (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
        (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
        (11, 23), (12, 24), (23, 24),
        # Legs
        (23, 25), (25, 27), (27, 29), (27, 31), (29, 31),
        (24, 26), (26, 28), (28, 30), (28, 32), (30, 32)
    ]
    
    # Approximate landmark positions for diagram (normalized coordinates)
    landmark_positions = {
        # Face
        0: (0.5, 0.95), 1: (0.52, 0.94), 2: (0.54, 0.93), 3: (0.56, 0.92),
        4: (0.48, 0.94), 5: (0.46, 0.93), 6: (0.44, 0.92), 7: (0.58, 0.90),
        8: (0.42, 0.90), 9: (0.51, 0.91), 10: (0.49, 0.91),
        # Shoulders and arms
        11: (0.55, 0.80), 12: (0.45, 0.80),
        13: (0.58, 0.70), 14: (0.42, 0.70),
        15: (0.60, 0.60), 16: (0.40, 0.60),
        17: (0.62, 0.58), 18: (0.38, 0.58),
        19: (0.63, 0.56), 20: (0.37, 0.56),
        21: (0.62, 0.57), 22: (0.38, 0.57),
        # Hips and legs
        23: (0.54, 0.55), 24: (0.46, 0.55),
        25: (0.54, 0.35), 26: (0.46, 0.35),
        27: (0.54, 0.15), 28: (0.46, 0.15),
        29: (0.56, 0.08), 30: (0.44, 0.08),
        31: (0.52, 0.05), 32: (0.48, 0.05)
    }
    
    # Landmark names
    landmark_names = {
        0: "Nose", 1: "Left Eye Inner", 2: "Left Eye", 3: "Left Eye Outer",
        4: "Right Eye Inner", 5: "Right Eye", 6: "Right Eye Outer",
        7: "Left Ear", 8: "Right Ear", 9: "Mouth Left", 10: "Mouth Right",
        11: "Left Shoulder", 12: "Right Shoulder",
        13: "Left Elbow", 14: "Right Elbow",
        15: "Left Wrist", 16: "Right Wrist",
        17: "Left Pinky", 18: "Right Pinky",
        19: "Left Index", 20: "Right Index",
        21: "Left Thumb", 22: "Right Thumb",
        23: "Left Hip", 24: "Right Hip",
        25: "Left Knee", 26: "Right Knee",
        27: "Left Ankle", 28: "Right Ankle",
        29: "Left Heel", 30: "Right Heel",
        31: "Left Foot Index", 32: "Right Foot Index"
    }
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 14))
    
    # Draw connections
    for start, end in connections:
        if start in landmark_positions and end in landmark_positions:
            x_coords = [landmark_positions[start][0], landmark_positions[end][0]]
            y_coords = [landmark_positions[start][1], landmark_positions[end][1]]
            ax.plot(x_coords, y_coords, 'b-', linewidth=2, alpha=0.6)
    
    # Draw landmarks
    for idx, (x, y) in landmark_positions.items():
        # Color code by body part
        if idx < 11:
            color = 'red'  # Face
        elif idx < 23:
            color = 'green'  # Upper body
        else:
            color = 'blue'  # Lower body
        
        circle = Circle((x, y), 0.015, color=color, alpha=0.7, zorder=3)
        ax.add_patch(circle)
        
        # Add landmark number
        ax.text(x, y, str(idx), fontsize=7, ha='center', va='center', 
               color='white', fontweight='bold', zorder=4)
    
    # Add title and legend
    ax.set_title('MediaPipe Pose: 33 Landmark Points', fontsize=14, fontweight='bold', pad=20)
    ax.set_xlim(0.3, 0.7)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.axis('off')
    
    # Legend
    legend_elements = [
        mlines.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', 
                     markersize=10, label='Face (0-10)'),
        mlines.Line2D([0], [0], marker='o', color='w', markerfacecolor='green', 
                     markersize=10, label='Upper Body (11-22)'),
        mlines.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', 
                     markersize=10, label='Lower Body (23-32)')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
    
    # Add text annotation
    text = "33 Body Landmarks tracked for gait analysis\n" + \
           "Lower body (blue) most important for walking patterns"
    ax.text(0.5, 0.02, text, fontsize=10, ha='center', 
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    output_path = 'data/visualizations/landmark_diagram.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Saved: {output_path}")

if __name__ == "__main__":
    print("\n")
    print("█" * 80)
    print("█" + " " * 22 + "SKELETON POSE VISUALIZATION" + " " * 29 + "█")
    print("█" * 80)
    
    create_skeleton_visualization()
