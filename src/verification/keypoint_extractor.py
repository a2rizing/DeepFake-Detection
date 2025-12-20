"""Keypoint extraction and processing for gait verification."""

import cv2
import mediapipe as mp
import numpy as np
from typing import Dict, Optional, Tuple
from scipy.signal import find_peaks


class KeypointExtractor:
    """
    Extracts and normalizes pose keypoints from video for gait verification.
    
    Uses MediaPipe Pose with consistent configuration to ensure reproducible
    keypoint extraction between signature building and verification.
    """
    
    # MediaPipe landmark indices for joint angle calculations
    # Reference: https://developers.google.com/mediapipe/solutions/vision/pose_landmarker
    LEFT_HIP = 23
    RIGHT_HIP = 24
    LEFT_KNEE = 25
    RIGHT_KNEE = 26
    LEFT_ANKLE = 27
    RIGHT_ANKLE = 28
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    LEFT_ELBOW = 13
    RIGHT_ELBOW = 14
    LEFT_WRIST = 15
    RIGHT_WRIST = 16
    
    def __init__(
        self,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        static_image_mode: bool = False
    ):
        """
        Initialize KeypointExtractor with MediaPipe Pose.
        
        Args:
            min_detection_confidence: Minimum confidence for pose detection
            min_tracking_confidence: Minimum confidence for pose tracking
            static_image_mode: If True, treats each frame independently
        """
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.static_image_mode = static_image_mode
        
        # Initialize MediaPipe Pose
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=static_image_mode,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
    
    def extract_from_video(self, video_path: str) -> np.ndarray:
        """
        Extract keypoints from a video file.
        
        Args:
            video_path: Path to the video file
            
        Returns:
            np.ndarray: Keypoint array of shape (T, 33, 2) where T is number of frames
                       with detected poses. Returns empty array if no poses detected.
                       
        Raises:
            FileNotFoundError: If video file does not exist
            ValueError: If video cannot be opened or read
        """
        import os
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video file: {video_path}")
        
        keypoints_list = []
        
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Convert BGR to RGB for MediaPipe
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.pose.process(rgb_frame)
                
                if results.pose_landmarks:
                    # Extract x, y coordinates for all 33 landmarks
                    frame_keypoints = []
                    for landmark in results.pose_landmarks.landmark:
                        frame_keypoints.append([landmark.x, landmark.y])
                    keypoints_list.append(frame_keypoints)
        finally:
            cap.release()
        
        if not keypoints_list:
            return np.array([]).reshape(0, 33, 2)
        
        return np.array(keypoints_list)  # Shape: (T, 33, 2)

    
    def normalize_sequence(self, keypoints: np.ndarray) -> np.ndarray:
        """
        Normalize keypoint sequence by centering on mid-hip and scaling by torso length.
        
        Args:
            keypoints: Keypoint array of shape (T, 33, 2)
            
        Returns:
            np.ndarray: Normalized keypoints of shape (T, 33, 2)
        """
        if keypoints.size == 0:
            return keypoints
        
        normalized = np.zeros_like(keypoints, dtype=np.float64)
        
        for t in range(keypoints.shape[0]):
            frame = keypoints[t].copy()
            
            # Calculate mid-hip point (center between left and right hip)
            left_hip = frame[self.LEFT_HIP]
            right_hip = frame[self.RIGHT_HIP]
            mid_hip = (left_hip + right_hip) / 2
            
            # Center on mid-hip
            centered = frame - mid_hip
            
            # Calculate torso length (average of left and right shoulder-hip distances)
            left_shoulder = centered[self.LEFT_SHOULDER]
            right_shoulder = centered[self.RIGHT_SHOULDER]
            left_hip_centered = centered[self.LEFT_HIP]
            right_hip_centered = centered[self.RIGHT_HIP]
            
            left_torso = np.linalg.norm(left_shoulder - left_hip_centered)
            right_torso = np.linalg.norm(right_shoulder - right_hip_centered)
            torso_length = (left_torso + right_torso) / 2
            
            # Scale by torso length (avoid division by zero)
            if torso_length > 1e-6:
                normalized[t] = centered / torso_length
            else:
                normalized[t] = centered
        
        return normalized
    
    def compute_joint_angles(self, keypoints: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Compute joint angles over time for knees, elbows, and hips.
        
        Args:
            keypoints: Keypoint array of shape (T, 33, 2)
            
        Returns:
            Dict[str, np.ndarray]: Dictionary mapping joint names to angle time series
        """
        if keypoints.size == 0:
            return {
                "left_knee": np.array([]),
                "right_knee": np.array([]),
                "left_elbow": np.array([]),
                "right_elbow": np.array([]),
                "left_hip": np.array([]),
                "right_hip": np.array([])
            }
        
        T = keypoints.shape[0]
        angles = {
            "left_knee": np.zeros(T),
            "right_knee": np.zeros(T),
            "left_elbow": np.zeros(T),
            "right_elbow": np.zeros(T),
            "left_hip": np.zeros(T),
            "right_hip": np.zeros(T)
        }
        
        for t in range(T):
            frame = keypoints[t]
            
            # Left knee: Hip - Knee - Ankle (indices 23 - 25 - 27)
            angles["left_knee"][t] = self._compute_angle(
                frame[self.LEFT_HIP],
                frame[self.LEFT_KNEE],
                frame[self.LEFT_ANKLE]
            )
            
            # Right knee: Hip - Knee - Ankle (indices 24 - 26 - 28)
            angles["right_knee"][t] = self._compute_angle(
                frame[self.RIGHT_HIP],
                frame[self.RIGHT_KNEE],
                frame[self.RIGHT_ANKLE]
            )
            
            # Left elbow: Shoulder - Elbow - Wrist (indices 11 - 13 - 15)
            angles["left_elbow"][t] = self._compute_angle(
                frame[self.LEFT_SHOULDER],
                frame[self.LEFT_ELBOW],
                frame[self.LEFT_WRIST]
            )
            
            # Right elbow: Shoulder - Elbow - Wrist (indices 12 - 14 - 16)
            angles["right_elbow"][t] = self._compute_angle(
                frame[self.RIGHT_SHOULDER],
                frame[self.RIGHT_ELBOW],
                frame[self.RIGHT_WRIST]
            )
            
            # Left hip: Shoulder - Hip - Knee (indices 11 - 23 - 25)
            angles["left_hip"][t] = self._compute_angle(
                frame[self.LEFT_SHOULDER],
                frame[self.LEFT_HIP],
                frame[self.LEFT_KNEE]
            )
            
            # Right hip: Shoulder - Hip - Knee (indices 12 - 24 - 26)
            angles["right_hip"][t] = self._compute_angle(
                frame[self.RIGHT_SHOULDER],
                frame[self.RIGHT_HIP],
                frame[self.RIGHT_KNEE]
            )
        
        return angles
    
    def _compute_angle(self, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
        """
        Compute angle at p2 formed by points p1-p2-p3.
        
        Args:
            p1: First point (x, y)
            p2: Vertex point (x, y)
            p3: Third point (x, y)
            
        Returns:
            float: Angle in radians
        """
        v1 = p1 - p2
        v2 = p3 - p2
        
        # Compute dot product and magnitudes
        dot = np.dot(v1, v2)
        mag1 = np.linalg.norm(v1)
        mag2 = np.linalg.norm(v2)
        
        # Avoid division by zero
        if mag1 < 1e-6 or mag2 < 1e-6:
            return 0.0
        
        # Clamp to [-1, 1] to handle numerical errors
        cos_angle = np.clip(dot / (mag1 * mag2), -1.0, 1.0)
        return np.arccos(cos_angle)

    
    def compute_stride_features(self, keypoints: np.ndarray) -> Dict[str, float]:
        """
        Compute stride-related features from keypoint sequence.
        
        Args:
            keypoints: Keypoint array of shape (T, 33, 2)
            
        Returns:
            Dict[str, float]: Dictionary containing stride features:
                - stride_length_mean: Average step length in normalized units
                - stride_length_std: Variability in step length
                - stride_frequency: Steps per second (assuming 30 fps)
                - left_right_symmetry: Ratio of left/right stride lengths
                - arm_swing_amplitude: Average arm swing range
        """
        if keypoints.size == 0 or keypoints.shape[0] < 10:
            return {
                "stride_length_mean": 0.0,
                "stride_length_std": 0.0,
                "stride_frequency": 0.0,
                "left_right_symmetry": 1.0,
                "arm_swing_amplitude": 0.0
            }
        
        T = keypoints.shape[0]
        
        # Extract ankle positions over time
        left_ankle = keypoints[:, self.LEFT_ANKLE, :]
        right_ankle = keypoints[:, self.RIGHT_ANKLE, :]
        
        # Compute stride lengths from ankle positions
        left_strides = self._compute_stride_lengths(left_ankle)
        right_strides = self._compute_stride_lengths(right_ankle)
        
        all_strides = np.concatenate([left_strides, right_strides]) if len(left_strides) > 0 or len(right_strides) > 0 else np.array([0.0])
        
        stride_length_mean = float(np.mean(all_strides)) if len(all_strides) > 0 else 0.0
        stride_length_std = float(np.std(all_strides)) if len(all_strides) > 0 else 0.0
        
        # Compute stride frequency (steps per second, assuming 30 fps)
        num_steps = len(left_strides) + len(right_strides)
        duration_seconds = T / 30.0  # Assuming 30 fps
        stride_frequency = num_steps / duration_seconds if duration_seconds > 0 else 0.0
        
        # Compute left-right symmetry
        left_mean = np.mean(left_strides) if len(left_strides) > 0 else 0.0
        right_mean = np.mean(right_strides) if len(right_strides) > 0 else 0.0
        if right_mean > 1e-6:
            left_right_symmetry = float(left_mean / right_mean)
        elif left_mean > 1e-6:
            left_right_symmetry = float(right_mean / left_mean)
        else:
            left_right_symmetry = 1.0
        
        # Compute arm swing amplitude
        arm_swing_amplitude = self._compute_arm_swing_amplitude(keypoints)
        
        return {
            "stride_length_mean": stride_length_mean,
            "stride_length_std": stride_length_std,
            "stride_frequency": stride_frequency,
            "left_right_symmetry": left_right_symmetry,
            "arm_swing_amplitude": arm_swing_amplitude
        }
    
    def _compute_stride_lengths(self, ankle_positions: np.ndarray) -> np.ndarray:
        """
        Compute stride lengths from ankle position time series.
        
        Uses peak detection on the x-coordinate to find stride events.
        
        Args:
            ankle_positions: Array of shape (T, 2) with ankle x, y positions
            
        Returns:
            np.ndarray: Array of stride lengths
        """
        if len(ankle_positions) < 10:
            return np.array([])
        
        # Use x-coordinate for stride detection (horizontal movement)
        x_positions = ankle_positions[:, 0]
        
        # Find peaks (forward-most positions of the foot)
        # Use a minimum distance between peaks to avoid noise
        min_distance = max(5, len(x_positions) // 20)
        peaks, _ = find_peaks(x_positions, distance=min_distance)
        
        if len(peaks) < 2:
            return np.array([])
        
        # Compute distances between consecutive peaks
        stride_lengths = []
        for i in range(1, len(peaks)):
            # Stride length is the horizontal distance traveled
            stride_length = abs(x_positions[peaks[i]] - x_positions[peaks[i-1]])
            stride_lengths.append(stride_length)
        
        return np.array(stride_lengths)
    
    def _compute_arm_swing_amplitude(self, keypoints: np.ndarray) -> float:
        """
        Compute average arm swing amplitude.
        
        Args:
            keypoints: Keypoint array of shape (T, 33, 2)
            
        Returns:
            float: Average arm swing amplitude
        """
        if keypoints.shape[0] < 2:
            return 0.0
        
        # Get wrist positions
        left_wrist = keypoints[:, self.LEFT_WRIST, :]
        right_wrist = keypoints[:, self.RIGHT_WRIST, :]
        
        # Compute range of motion for each wrist
        left_range_x = np.max(left_wrist[:, 0]) - np.min(left_wrist[:, 0])
        left_range_y = np.max(left_wrist[:, 1]) - np.min(left_wrist[:, 1])
        right_range_x = np.max(right_wrist[:, 0]) - np.min(right_wrist[:, 0])
        right_range_y = np.max(right_wrist[:, 1]) - np.min(right_wrist[:, 1])
        
        # Average amplitude (using Euclidean distance of ranges)
        left_amplitude = np.sqrt(left_range_x**2 + left_range_y**2)
        right_amplitude = np.sqrt(right_range_x**2 + right_range_y**2)
        
        return float((left_amplitude + right_amplitude) / 2)
    
    def __del__(self):
        """Clean up MediaPipe resources."""
        if hasattr(self, 'pose'):
            self.pose.close()
