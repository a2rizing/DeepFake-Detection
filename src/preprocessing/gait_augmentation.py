"""
Gait-specific data augmentation module for gait recognition training.

This module implements various augmentation techniques that preserve
biomechanical validity of gait patterns while increasing training data diversity.

Requirements: 4.1, 4.2, 4.3, 4.4
"""
import numpy as np
from typing import Tuple, Optional, List
from scipy.interpolate import interp1d


class GaitAugmentation:
    """
    Gait-specific data augmentation for sequence data.
    
    All augmentation methods preserve the input shape (sequence_length, num_features)
    and maintain biomechanical validity of gait patterns.
    
    Expected input shape: (64, 70) where:
    - 64 = number of frames (sequence_length)
    - 70 = 66 normalized coordinates (33 landmarks * 2) + 4 joint angles
    """
    
    # Feature indices for joint angles (last 4 features)
    ANGLE_START_IDX = 66
    NUM_ANGLES = 4
    
    def __init__(self, random_state: Optional[int] = None):
        """
        Initialize the augmentation module.
        
        Args:
            random_state: Random seed for reproducibility
        """
        self.rng = np.random.default_rng(random_state)
    
    def temporal_warp(self, 
                      sequence: np.ndarray, 
                      warp_factor: float = 0.2,
                      num_knots: int = 4) -> np.ndarray:
        """
        Apply non-uniform temporal warping to simulate speed variations.
        
        This creates realistic variations in walking speed by warping the
        time axis non-uniformly while preserving the sequence length.
        
        Args:
            sequence: Input sequence of shape (T, D)
            warp_factor: Maximum deviation from uniform timing (0.0-1.0)
            num_knots: Number of control points for the warping curve
            
        Returns:
            Warped sequence of same shape (T, D)
            
        Requirements: 4.1 - Temporal warping with non-uniform speed variations
        """
        if sequence.ndim != 2:
            raise ValueError(f"Expected 2D array, got shape {sequence.shape}")
        
        T, D = sequence.shape
        
        # Create original time indices
        original_time = np.linspace(0, 1, T)
        
        # Create random knot points for warping
        # Knots define how time is stretched/compressed at different points
        knot_positions = np.linspace(0, 1, num_knots + 2)  # Include endpoints
        
        # Random deviations at each knot (except endpoints which stay fixed)
        deviations = np.zeros(num_knots + 2)
        deviations[1:-1] = self.rng.uniform(-warp_factor, warp_factor, num_knots)
        
        # Cumulative sum to create monotonic warping function
        # This ensures time always moves forward
        warped_knots = knot_positions + deviations
        warped_knots = np.clip(warped_knots, 0, 1)
        warped_knots[0] = 0  # Fix start
        warped_knots[-1] = 1  # Fix end
        
        # Ensure monotonicity
        for i in range(1, len(warped_knots)):
            if warped_knots[i] <= warped_knots[i-1]:
                warped_knots[i] = warped_knots[i-1] + 1e-6
        
        # Create smooth warping function using cubic interpolation
        warp_func = interp1d(knot_positions, warped_knots, kind='cubic')
        warped_time = warp_func(original_time)
        
        # Normalize warped time to [0, 1]
        warped_time = (warped_time - warped_time.min()) / (warped_time.max() - warped_time.min() + 1e-8)
        
        # Resample sequence at warped time points
        warped_sequence = np.zeros_like(sequence)
        for d in range(D):
            # Create interpolator for this feature
            feature_interp = interp1d(original_time, sequence[:, d], 
                                      kind='linear', fill_value='extrapolate')
            warped_sequence[:, d] = feature_interp(warped_time)
        
        return warped_sequence
    
    def magnitude_scale(self, 
                        sequence: np.ndarray,
                        scale_range: Tuple[float, float] = (0.9, 1.1)) -> np.ndarray:
        """
        Scale joint angles by a random factor within the specified range.
        
        This simulates natural variations in joint flexibility and movement
        amplitude while preserving the overall gait pattern structure.
        
        Args:
            sequence: Input sequence of shape (T, D) where D >= 70
            scale_range: (min_scale, max_scale) for angle scaling
            
        Returns:
            Scaled sequence of same shape (T, D)
            
        Requirements: 4.2 - Magnitude scaling of joint angles (0.9-1.1 factor)
        """
        if sequence.ndim != 2:
            raise ValueError(f"Expected 2D array, got shape {sequence.shape}")
        
        T, D = sequence.shape
        
        # Validate scale range
        if scale_range[0] > scale_range[1]:
            raise ValueError("scale_range[0] must be <= scale_range[1]")
        
        # Create output copy
        scaled_sequence = sequence.copy()
        
        # Only scale joint angles (last 4 features if D >= 70)
        if D >= self.ANGLE_START_IDX + self.NUM_ANGLES:
            # Generate random scale factor
            scale_factor = self.rng.uniform(scale_range[0], scale_range[1])
            
            # Scale the angle features
            angle_indices = slice(self.ANGLE_START_IDX, self.ANGLE_START_IDX + self.NUM_ANGLES)
            scaled_sequence[:, angle_indices] = sequence[:, angle_indices] * scale_factor
        
        return scaled_sequence

    
    def add_gaussian_noise(self, 
                           sequence: np.ndarray,
                           std: float = 0.01,
                           noise_coords_only: bool = True) -> np.ndarray:
        """
        Add Gaussian noise to normalized coordinates.
        
        This simulates measurement noise and small variations in pose
        estimation while preserving the overall gait pattern.
        
        Args:
            sequence: Input sequence of shape (T, D)
            std: Standard deviation of Gaussian noise
            noise_coords_only: If True, only add noise to coordinates (not angles)
            
        Returns:
            Noisy sequence of same shape (T, D)
            
        Requirements: 4.3 - Gaussian noise injection on normalized coordinates
        """
        if sequence.ndim != 2:
            raise ValueError(f"Expected 2D array, got shape {sequence.shape}")
        
        T, D = sequence.shape
        noisy_sequence = sequence.copy()
        
        if noise_coords_only and D >= self.ANGLE_START_IDX + self.NUM_ANGLES:
            # Only add noise to coordinate features (first 66 features)
            noise = self.rng.normal(0, std, (T, self.ANGLE_START_IDX))
            noisy_sequence[:, :self.ANGLE_START_IDX] += noise
        else:
            # Add noise to all features
            noise = self.rng.normal(0, std, (T, D))
            noisy_sequence += noise
        
        return noisy_sequence
    
    def mixup(self, 
              seq1: np.ndarray, 
              seq2: np.ndarray,
              alpha: float = 0.2) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform mixup augmentation between two sequences.
        
        Mixup creates new training samples by linearly interpolating
        between two sequences and their labels. This helps regularization
        and improves generalization.
        
        Args:
            seq1: First sequence of shape (T, D)
            seq2: Second sequence of shape (T, D)
            alpha: Beta distribution parameter for mixing coefficient
            
        Returns:
            Tuple of (mixed_sequence, mixing_coefficient) where:
            - mixed_sequence: Interpolated sequence of shape (T, D)
            - mixing_coefficient: Lambda value used for mixing (for label interpolation)
            
        Requirements: 4.4 - Sequence mixup combining features from different samples
        """
        if seq1.shape != seq2.shape:
            raise ValueError(f"Sequences must have same shape: {seq1.shape} vs {seq2.shape}")
        
        if seq1.ndim != 2:
            raise ValueError(f"Expected 2D arrays, got shape {seq1.shape}")
        
        # Sample mixing coefficient from Beta distribution
        # When alpha is small, lambda tends to be close to 0 or 1
        # When alpha is larger, lambda tends to be closer to 0.5
        lam = self.rng.beta(alpha, alpha)
        
        # Ensure lambda is at least 0.5 to keep seq1 as the dominant sample
        lam = max(lam, 1 - lam)
        
        # Linear interpolation
        mixed_sequence = lam * seq1 + (1 - lam) * seq2
        
        # Return mixing coefficient as array for label interpolation
        mixing_coef = np.array([lam, 1 - lam], dtype=np.float32)
        
        return mixed_sequence, mixing_coef
    
    def augment_single(self, 
                       sequence: np.ndarray,
                       apply_temporal_warp: bool = True,
                       apply_magnitude_scale: bool = True,
                       apply_noise: bool = True,
                       warp_factor: float = 0.2,
                       scale_range: Tuple[float, float] = (0.9, 1.1),
                       noise_std: float = 0.01) -> np.ndarray:
        """
        Apply a combination of augmentations to a single sequence.
        
        Args:
            sequence: Input sequence of shape (T, D)
            apply_temporal_warp: Whether to apply temporal warping
            apply_magnitude_scale: Whether to apply magnitude scaling
            apply_noise: Whether to add Gaussian noise
            warp_factor: Temporal warp factor
            scale_range: Range for magnitude scaling
            noise_std: Standard deviation for Gaussian noise
            
        Returns:
            Augmented sequence of same shape (T, D)
        """
        augmented = sequence.copy()
        
        if apply_temporal_warp:
            augmented = self.temporal_warp(augmented, warp_factor=warp_factor)
        
        if apply_magnitude_scale:
            augmented = self.magnitude_scale(augmented, scale_range=scale_range)
        
        if apply_noise:
            augmented = self.add_gaussian_noise(augmented, std=noise_std)
        
        return augmented
    
    def augment_batch(self, 
                      X: np.ndarray, 
                      y: np.ndarray,
                      augment_factor: int = 2,
                      use_mixup: bool = False,
                      mixup_alpha: float = 0.2) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply augmentations to increase dataset size.
        
        Args:
            X: Input data of shape (N, T, D)
            y: Labels of shape (N,) - integer class labels
            augment_factor: Number of augmented copies per original sample
            use_mixup: Whether to include mixup augmentation
            mixup_alpha: Alpha parameter for mixup
            
        Returns:
            Tuple of (X_augmented, y_augmented) where:
            - X_augmented: Shape (N * (1 + augment_factor), T, D)
            - y_augmented: Shape (N * (1 + augment_factor),) for hard labels
                          or (N * (1 + augment_factor), num_classes) for soft labels with mixup
        """
        if X.ndim != 3:
            raise ValueError(f"Expected 3D array for X, got shape {X.shape}")
        
        N, T, D = X.shape
        
        # Start with original data
        X_list = [X]
        y_list = [y]
        
        for _ in range(augment_factor):
            # Create augmented batch
            X_aug = np.zeros_like(X)
            
            for i in range(N):
                # Randomly select which augmentations to apply
                apply_warp = self.rng.random() > 0.3
                apply_scale = self.rng.random() > 0.3
                apply_noise = self.rng.random() > 0.3
                
                X_aug[i] = self.augment_single(
                    X[i],
                    apply_temporal_warp=apply_warp,
                    apply_magnitude_scale=apply_scale,
                    apply_noise=apply_noise
                )
            
            X_list.append(X_aug)
            y_list.append(y.copy())
        
        # Optionally add mixup samples
        if use_mixup and N > 1:
            X_mixup = np.zeros_like(X)
            y_mixup = y.copy()  # For simplicity, use dominant class label
            
            for i in range(N):
                # Select random partner for mixup
                j = self.rng.integers(0, N)
                while j == i and N > 1:
                    j = self.rng.integers(0, N)
                
                mixed_seq, mix_coef = self.mixup(X[i], X[j], alpha=mixup_alpha)
                X_mixup[i] = mixed_seq
                
                # Use dominant class label (the one with higher weight)
                if mix_coef[0] >= 0.5:
                    y_mixup[i] = y[i]
                else:
                    y_mixup[i] = y[j]
            
            X_list.append(X_mixup)
            y_list.append(y_mixup)
        
        # Concatenate all augmented data
        X_augmented = np.concatenate(X_list, axis=0)
        y_augmented = np.concatenate(y_list, axis=0)
        
        return X_augmented, y_augmented


def create_augmenter(random_state: int = 42) -> GaitAugmentation:
    """
    Factory function to create a GaitAugmentation instance.
    
    Args:
        random_state: Random seed for reproducibility
        
    Returns:
        Configured GaitAugmentation instance
    """
    return GaitAugmentation(random_state=random_state)
