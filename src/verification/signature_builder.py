"""Signature building for gait verification."""

import json
import os
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np

from .data_structures import GaitSignature
from .keypoint_extractor import KeypointExtractor


class SignatureBuilder:
    """
    Builds gait signatures from training videos.
    
    A gait signature captures the unique walking pattern of a person,
    computed from multiple training videos. The signature includes
    statistical summaries of keypoint positions, joint angles, and
    stride characteristics.
    """
    
    def __init__(self, keypoint_extractor: Optional[KeypointExtractor] = None):
        """
        Initialize SignatureBuilder with a KeypointExtractor.
        
        Args:
            keypoint_extractor: KeypointExtractor instance for processing videos.
                               If None, a default extractor will be created.
        """
        self.keypoint_extractor = keypoint_extractor or KeypointExtractor()
    
    def build_signature(self, identity: str, video_paths: List[str]) -> GaitSignature:
        """
        Build a gait signature from multiple training videos.
        
        Args:
            identity: Person name (e.g., "Aarav")
            video_paths: List of paths to training videos
            
        Returns:
            GaitSignature: The computed gait signature
            
        Raises:
            ValueError: If no valid keypoints could be extracted from any video
        """
        if not video_paths:
            raise ValueError("At least one video path is required")
        
        all_keypoints = []
        all_normalized = []
        all_joint_angles = []
        all_stride_features = []
        sample_sequences = []
        
        for video_path in video_paths:
            try:
                # Extract keypoints from video
                keypoints = self.keypoint_extractor.extract_from_video(video_path)
                
                if keypoints.size == 0:
                    continue
                
                # Normalize the sequence
                normalized = self.keypoint_extractor.normalize_sequence(keypoints)
                
                # Compute joint angles
                joint_angles = self.keypoint_extractor.compute_joint_angles(normalized)
                
                # Compute stride features
                stride_features = self.keypoint_extractor.compute_stride_features(normalized)
                
                all_keypoints.append(keypoints)
                all_normalized.append(normalized)
                all_joint_angles.append(joint_angles)
                all_stride_features.append(stride_features)
                
                # Store sample sequence for DTW comparison (use normalized)
                sample_sequences.append(normalized)
                
            except (FileNotFoundError, ValueError) as e:
                # Log warning but continue with other videos
                print(f"Warning: Could not process {video_path}: {e}")
                continue
        
        if not all_normalized:
            raise ValueError(f"No valid keypoints could be extracted from any video for {identity}")
        
        # Compute mean keypoints and variance across all videos
        mean_keypoints, keypoint_variance = self._compute_keypoint_statistics(all_normalized)
        
        # Compute joint angle statistics
        joint_angle_stats = self._compute_joint_angle_statistics(all_joint_angles)
        
        # Compute average stride features
        stride_features = self._compute_average_stride_features(all_stride_features)
        
        return GaitSignature(
            identity=identity,
            mean_keypoints=mean_keypoints,
            keypoint_variance=keypoint_variance,
            joint_angle_stats=joint_angle_stats,
            stride_features=stride_features,
            sample_sequences=sample_sequences,
            num_training_samples=len(all_normalized),
            created_at=datetime.now().isoformat()
        )
    
    def _compute_keypoint_statistics(
        self, 
        normalized_sequences: List[np.ndarray]
    ) -> tuple:
        """
        Compute mean and variance of keypoints across all sequences.
        
        Args:
            normalized_sequences: List of normalized keypoint arrays
            
        Returns:
            Tuple of (mean_keypoints, keypoint_variance) each of shape (33, 2)
        """
        # Concatenate all frames from all sequences
        all_frames = []
        for seq in normalized_sequences:
            for frame in seq:
                all_frames.append(frame)
        
        all_frames = np.array(all_frames)  # Shape: (total_frames, 33, 2)
        
        # Compute mean and variance across all frames
        mean_keypoints = np.mean(all_frames, axis=0)  # Shape: (33, 2)
        keypoint_variance = np.var(all_frames, axis=0)  # Shape: (33, 2)
        
        return mean_keypoints, keypoint_variance
    
    def _compute_joint_angle_statistics(
        self, 
        all_joint_angles: List[Dict[str, np.ndarray]]
    ) -> Dict[str, Dict[str, float]]:
        """
        Compute mean and std for each joint angle across all videos.
        
        Args:
            all_joint_angles: List of joint angle dictionaries from each video
            
        Returns:
            Dictionary mapping joint names to {"mean": float, "std": float}
        """
        joint_names = ["left_knee", "right_knee", "left_elbow", "right_elbow", "left_hip", "right_hip"]
        stats = {}
        
        for joint in joint_names:
            # Concatenate all angle values for this joint across all videos
            all_angles = []
            for angles_dict in all_joint_angles:
                if joint in angles_dict and len(angles_dict[joint]) > 0:
                    all_angles.extend(angles_dict[joint].tolist())
            
            if all_angles:
                stats[joint] = {
                    "mean": float(np.mean(all_angles)),
                    "std": float(np.std(all_angles))
                }
            else:
                stats[joint] = {"mean": 0.0, "std": 0.0}
        
        return stats
    
    def _compute_average_stride_features(
        self, 
        all_stride_features: List[Dict[str, float]]
    ) -> Dict[str, float]:
        """
        Compute average stride features across all videos.
        
        Args:
            all_stride_features: List of stride feature dictionaries
            
        Returns:
            Dictionary with averaged stride features
        """
        feature_names = [
            "stride_length_mean", 
            "stride_length_std", 
            "stride_frequency",
            "left_right_symmetry", 
            "arm_swing_amplitude"
        ]
        
        averaged = {}
        for feature in feature_names:
            values = [sf[feature] for sf in all_stride_features if feature in sf]
            averaged[feature] = float(np.mean(values)) if values else 0.0
        
        return averaged

    
    def save_signature(self, signature: GaitSignature, output_dir: str) -> str:
        """
        Save a gait signature to a JSON file.
        
        Args:
            signature: The GaitSignature to save
            output_dir: Directory to save the signature file
            
        Returns:
            str: Path to the saved signature file
        """
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Create filename based on identity
        filename = f"{signature.identity}_signature.json"
        filepath = os.path.join(output_dir, filename)
        
        # Serialize and save
        with open(filepath, 'w') as f:
            json.dump(signature.to_dict(), f, indent=2)
        
        return filepath
    
    def load_signature(self, identity: str, signatures_dir: str) -> GaitSignature:
        """
        Load a single signature by identity name.
        
        Args:
            identity: Person name to load signature for
            signatures_dir: Directory containing signature files
            
        Returns:
            GaitSignature: The loaded signature
            
        Raises:
            FileNotFoundError: If signature file does not exist
            ValueError: If signature file is corrupted or invalid
        """
        filename = f"{identity}_signature.json"
        filepath = os.path.join(signatures_dir, filename)
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Signature file not found for identity: {identity}")
        
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            return GaitSignature.from_dict(data)
        except (json.JSONDecodeError, KeyError) as e:
            raise ValueError(f"Corrupted signature file for {identity}: {e}")
    
    def load_all_signatures(self, signatures_dir: str) -> Dict[str, GaitSignature]:
        """
        Load all signatures from a directory.
        
        Args:
            signatures_dir: Directory containing signature files
            
        Returns:
            Dict[str, GaitSignature]: Dictionary mapping identity names to signatures
        """
        signatures = {}
        
        if not os.path.exists(signatures_dir):
            return signatures
        
        for filename in os.listdir(signatures_dir):
            if filename.endswith('_signature.json'):
                # Extract identity from filename
                identity = filename.replace('_signature.json', '')
                filepath = os.path.join(signatures_dir, filename)
                
                try:
                    with open(filepath, 'r') as f:
                        data = json.load(f)
                    signatures[identity] = GaitSignature.from_dict(data)
                except (json.JSONDecodeError, KeyError, ValueError) as e:
                    # Log warning but continue loading other signatures
                    print(f"Warning: Could not load signature from {filename}: {e}")
                    continue
        
        return signatures
