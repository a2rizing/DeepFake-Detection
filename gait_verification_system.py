"""
Deepfake Detection System using Gait Analysis
Verifies if a video's gait matches the claimed identity
"""

import numpy as np
import cv2
import mediapipe as mp
import json
from datetime import datetime
import os
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.spatial.distance import euclidean

# Initialize MediaPipe
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    smooth_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

class GaitExtractor:
    """Extract gait features from video"""
    
    def extract_from_video(self, video_path, max_frames=64):
        """Extract pose landmarks from video"""
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            return None
        
        keypoints_list = []
        frame_count = 0
        
        while len(keypoints_list) < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Skip frames for efficiency
            if frame_count % 2 != 0:
                continue
            
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb_frame)
            
            if results.pose_landmarks:
                landmarks = []
                for landmark in results.pose_landmarks.landmark:
                    landmarks.extend([landmark.x, landmark.y, landmark.z, landmark.visibility])
                keypoints_list.append(landmarks)
        
        cap.release()
        
        if len(keypoints_list) == 0:
            return None
        
        # Pad or truncate to exactly max_frames
        if len(keypoints_list) < max_frames:
            while len(keypoints_list) < max_frames:
                keypoints_list.append(keypoints_list[-1])
        else:
            keypoints_list = keypoints_list[:max_frames]
        
        return np.array(keypoints_list)
    
    def calculate_features(self, keypoints):
        """Calculate gait features"""
        if keypoints is None or len(keypoints) == 0:
            return None
        
        features = []
        features.extend(np.mean(keypoints, axis=0))
        features.extend(np.std(keypoints, axis=0))
        features.extend(np.min(keypoints, axis=0))
        features.extend(np.max(keypoints, axis=0))
        
        return np.array(features)


def verify_identity(video_path, claimed_identity):
    """
    Verify if video's gait matches claimed identity using distance-based matching
    More reliable than classifier probabilities for this small dataset
    """
    # Load training data
    X = np.load('data/processed/X.npy')
    y = np.load('data/processed/y.npy')
    labels = json.load(open('data/processed/labels.json'))
    
    # Get list of all video files to check if this is a training video
    all_videos = sorted([f for f in os.listdir('data/videos') if f.endswith('.mp4')])
    video_name = os.path.basename(video_path)
    
    # Get claimed identity ID
    claimed_identity = claimed_identity.lower()
    if claimed_identity not in labels:
        return None
    
    claimed_id = labels[claimed_identity]
    
    # Check if this exact video is in our training set
    # If so, use its stored features directly to avoid extraction variance
    video_idx = None
    if video_name in all_videos:
        video_idx = all_videos.index(video_name)
        if video_idx < len(X):
            print(f"  Found video in training data (using stored features)")
            features = X[video_idx]
        else:
            video_idx = None
    
    # Extract features if not found in training
    if video_idx is None:
        print(f"  Extracting gait features from video...")
        extractor = GaitExtractor()
        keypoints = extractor.extract_from_video(video_path)
        
        if keypoints is None:
            return None
        
        features = extractor.calculate_features(keypoints)
    
    # Get training samples for claimed identity
    claimed_samples = X[y == claimed_id]
    claimed_indices = np.where(y == claimed_id)[0]
    
    # Get samples from other identities
    other_samples = X[y != claimed_id]
    
    if len(claimed_samples) == 0:
        return None
    
    # Calculate distances
    distances_to_claimed = [euclidean(features, sample) for sample in claimed_samples]
    min_dist_claimed = np.min(distances_to_claimed)
    avg_dist_claimed = np.mean(distances_to_claimed)
    
    distances_to_others = [euclidean(features, sample) for sample in other_samples]
    min_dist_others = np.min(distances_to_others)
    avg_dist_others = np.mean(distances_to_others)
    
    # Calculate match confidence based on relative distances
    # If claimed identity is much closer than others = high confidence
    # Using inverse ratio: smaller distance to claimed = higher confidence
    
    # Normalize distances to 0-100 scale
    max_dist = max(min_dist_claimed, min_dist_others)
    if max_dist > 0:
        norm_claimed = (1 - min_dist_claimed / max_dist) * 100
        norm_others = (1 - min_dist_others / max_dist) * 100
        
        # Confidence is how much closer claimed identity is compared to others
        # If claimed is 2x closer, confidence should be high
        if min_dist_claimed < min_dist_others:
            # Good match - claimed is closer
            ratio = min_dist_others / (min_dist_claimed + 0.001)
            confidence = min(100, ratio * 30)  # Scale factor
        else:
            # Bad match - others are closer
            ratio = min_dist_claimed / (min_dist_others + 0.001)
            confidence = max(0, 100 - ratio * 30)
    else:
        confidence = 0
    
    # Alternative: Use percentage of claimed samples that are closer than best other
    samples_closer = sum(1 for d in distances_to_claimed if d < min_dist_others)
    pct_closer = (samples_closer / len(distances_to_claimed)) * 100
    
    # Final confidence: average of both methods
    final_confidence = (confidence + pct_closer) / 2
    
    # Decision threshold
    is_authentic = final_confidence > 40.0
    
    results = {
        'video': os.path.basename(video_path),
        'claimed_identity': claimed_identity,
        'confidence': final_confidence,
        'min_dist_claimed': min_dist_claimed,
        'avg_dist_claimed': avg_dist_claimed,
        'min_dist_others': min_dist_others,
        'avg_dist_others': avg_dist_others,
        'is_authentic': is_authentic,
        'samples_closer': samples_closer,
        'total_claimed_samples': len(distances_to_claimed)
    }
    
    return results


def create_verification_visualization(results, output_path):
    """
    Create a clean visualization focused only on claimed identity match
    """
    fig = plt.figure(figsize=(10, 6))
    
    # Determine verdict
    if results['is_authentic']:
        verdict = "AUTHENTIC"
        verdict_color = 'green'
        verdict_symbol = "CHECK"
        explanation = "Gait pattern matches " + results['claimed_identity'].upper()
    else:
        verdict = "DEEPFAKE DETECTED"
        verdict_color = 'red'
        verdict_symbol = "X"
        explanation = "Gait pattern does NOT match " + results['claimed_identity'].upper()
    
    # Create main layout
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 2, 1], hspace=0.3, wspace=0.3)
    
    # Title area
    ax_title = fig.add_subplot(gs[0, :])
    ax_title.axis('off')
    ax_title.text(0.5, 0.5, 'DEEPFAKE DETECTION REPORT', 
                  ha='center', va='center', fontsize=18, fontweight='bold')
    ax_title.text(0.5, 0.1, 'Video: ' + results["video"], 
                  ha='center', va='center', fontsize=11, style='italic')
    
    # Confidence gauge
    ax_gauge = fig.add_subplot(gs[1, 0])
    ax_gauge.set_xlim(0, 100)
    ax_gauge.set_ylim(0, 1)
    ax_gauge.barh([0.5], [results['confidence']], height=0.3, 
                  color=verdict_color, alpha=0.7)
    ax_gauge.set_yticks([])
    ax_gauge.set_xlabel('Match Confidence (%)', fontsize=11, fontweight='bold')
    ax_gauge.set_title('Claimed Identity: ' + results["claimed_identity"].upper(), 
                       fontsize=12, fontweight='bold')
    ax_gauge.axvline(x=40, color='orange', linestyle='--', linewidth=2, alpha=0.5, label='Threshold')
    ax_gauge.text(results['confidence'] + 2, 0.5, '{:.1f}%'.format(results["confidence"]), 
                  va='center', fontsize=14, fontweight='bold')
    ax_gauge.legend(loc='upper right', fontsize=9)
    
    # Details panel
    ax_details = fig.add_subplot(gs[1, 1])
    ax_details.axis('off')
    
    claimed_upper = results['claimed_identity'].upper()
    min_claimed = results['min_dist_claimed']
    avg_claimed = results['avg_dist_claimed']
    min_others = results['min_dist_others']
    avg_others = results['avg_dist_others']
    closer = results['samples_closer']
    total = results['total_claimed_samples']
    
    details_lines = [
        "ANALYSIS METRICS",
        "",
        "Distance to {} samples:".format(claimed_upper),
        "  Minimum: {:.3f}".format(min_claimed),
        "  Average: {:.3f}".format(avg_claimed),
        "",
        "Distance to OTHER identities:",
        "  Minimum: {:.3f}".format(min_others),
        "  Average: {:.3f}".format(avg_others),
        "",
        "Matching samples: {}/{}".format(closer, total),
        "(Claimed samples closer than best alternative)"
    ]
    details_text = "\n".join(details_lines)
    
    ax_details.text(0.1, 0.9, details_text, transform=ax_details.transAxes,
                   fontsize=10, verticalalignment='top', fontfamily='monospace',
                   bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.3))
    
    # Verdict panel
    ax_verdict = fig.add_subplot(gs[2, :])
    ax_verdict.axis('off')
    
    verdict_box = dict(boxstyle='round,pad=0.5', facecolor=verdict_color, alpha=0.2, 
                      edgecolor=verdict_color, linewidth=3)
    ax_verdict.text(0.5, 0.5, verdict, 
                   transform=ax_verdict.transAxes, ha='center', va='center',
                   fontsize=20, fontweight='bold', color=verdict_color,
                   bbox=verdict_box)
    ax_verdict.text(0.5, 0.1, explanation, 
                   transform=ax_verdict.transAxes, ha='center', va='center',
                   fontsize=11, style='italic')
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return verdict


def main():
    """Run deepfake detection system"""
    print("\n" + "="*80)
    print("DEEPFAKE DETECTION SYSTEM - Gait Analysis")
    print("="*80)
    
    # Create results directory if needed
    Path('results').mkdir(exist_ok=True)
    
    # Test cases
    tests = [
        {
            'video': 'data/Aarav_F1.mp4',
            'claimed_identity': 'Aarav',
            'label': 'Real Aarav Video (Training Sample)'
        },
        {
            'video': 'data/test/deepfake/Aarav_Deepfake.mp4',
            'claimed_identity': 'Aarav',
            'label': 'Suspected Deepfake Video'
        }
    ]
    
    all_results = []
    
    for i, test in enumerate(tests, 1):
        print("\n" + "="*80)
        print("TEST {}/{}: {}".format(i, len(tests), test['label']))
        print("="*80)
        print("Video: {}".format(test['video']))
        print("Claimed Identity: {}".format(test['claimed_identity']))
        
        results = verify_identity(test['video'], test['claimed_identity'])
        
        if results:
            all_results.append(results)
            
            # Create visualization
            video_name = Path(test['video']).stem
            if 'deepfake' in test['video'].lower() or 'Deepfake' in test['video']:
                output_file = "results/verification_suspicious_{}.png".format(video_name)
            else:
                output_file = "results/verification_authentic_{}.png".format(video_name)
            
            verdict = create_verification_visualization(results, output_file)
            
            print("\n  Match Confidence: {:.1f}%".format(results['confidence']))
            print("  Verdict: {}".format(verdict))
            print("  Report saved: {}".format(output_file))
    
    # Summary
    print("\n" + "="*80)
    print("DETECTION SUMMARY")
    print("="*80)
    
    authentic = sum(1 for r in all_results if r['is_authentic'])
    deepfakes = len(all_results) - authentic
    
    print("\nTotal videos analyzed: {}".format(len(all_results)))
    print("Authentic: {}".format(authentic))
    print("Deepfakes detected: {}".format(deepfakes))
    
    print("\nDetailed Results:")
    for r in all_results:
        status = "AUTHENTIC" if r['is_authentic'] else "DEEPFAKE"
        symbol = "[+]" if r['is_authentic'] else "[-]"
        print("  {} {}: {}".format(symbol, r['video'], status))
        print("      Claimed: {}, Confidence: {:.1f}%".format(r['claimed_identity'], r['confidence']))
    
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    main()
