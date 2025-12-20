"""Gait verification for deepfake detection."""

import numpy as np
from typing import Dict, List, Optional, Tuple
from scipy.spatial.distance import cdist
from scipy.signal import savgol_filter

from .data_structures import GaitSignature, VerificationResult
from .keypoint_extractor import KeypointExtractor
from .signature_builder import SignatureBuilder


class GaitVerifier:
    """
    Performs gait verification by comparing test video gait against stored signatures.
    
    The verifier extracts gait features from a test video and compares them against
    the stored authentic signature for the claimed identity. It combines multiple
    similarity metrics (DTW, joint angles, stride patterns) to compute an overall
    authenticity score.
    """
    
    DEFAULT_THRESHOLD = 0.7
    
    def __init__(
        self, 
        signatures_dir: str, 
        threshold: float = DEFAULT_THRESHOLD,
        keypoint_extractor: Optional[KeypointExtractor] = None
    ):
        """
        Initialize GaitVerifier with signatures directory and threshold.
        
        Args:
            signatures_dir: Directory containing signature JSON files
            threshold: Authenticity threshold (default 0.7). Scores >= threshold
                      are classified as authentic.
            keypoint_extractor: Optional KeypointExtractor instance. If None,
                               a default extractor will be created.
        """
        self.signatures_dir = signatures_dir
        self.threshold = threshold
        self.keypoint_extractor = keypoint_extractor or KeypointExtractor()
        
        # Load all signatures on initialization
        self._signature_builder = SignatureBuilder(self.keypoint_extractor)
        self.signatures: Dict[str, GaitSignature] = self._signature_builder.load_all_signatures(
            signatures_dir
        )

    
    def compute_dtw_similarity(self, seq1: np.ndarray, seq2: np.ndarray) -> float:
        """
        Compute DTW-based similarity between two keypoint sequences.
        
        Uses Dynamic Time Warping to measure similarity between sequences
        that may vary in speed/length. The distance is normalized to a
        0-1 similarity score.
        
        Args:
            seq1: First keypoint sequence of shape (T1, 33, 2)
            seq2: Second keypoint sequence of shape (T2, 33, 2)
            
        Returns:
            float: Similarity score in range [0.0, 1.0], where 1.0 is identical
        """
        if seq1.size == 0 or seq2.size == 0:
            return 0.0
        
        # Flatten keypoints for each frame: (T, 33, 2) -> (T, 66)
        seq1_flat = seq1.reshape(seq1.shape[0], -1)
        seq2_flat = seq2.reshape(seq2.shape[0], -1)
        
        # Compute DTW distance using dynamic programming
        dtw_distance = self._compute_dtw_distance(seq1_flat, seq2_flat)
        
        # Normalize distance to similarity score
        # Use exponential decay to map distance to [0, 1]
        # The scaling factor is tuned for typical keypoint distances
        similarity = np.exp(-dtw_distance / 10.0)
        
        return float(np.clip(similarity, 0.0, 1.0))
    
    def _compute_dtw_distance(self, seq1: np.ndarray, seq2: np.ndarray) -> float:
        """
        Compute DTW distance between two sequences using dynamic programming.
        
        Args:
            seq1: First sequence of shape (T1, D)
            seq2: Second sequence of shape (T2, D)
            
        Returns:
            float: DTW distance (unnormalized)
        """
        n, m = len(seq1), len(seq2)
        
        # Compute pairwise distances
        cost_matrix = cdist(seq1, seq2, metric='euclidean')
        
        # Initialize DTW matrix
        dtw = np.full((n + 1, m + 1), np.inf)
        dtw[0, 0] = 0
        
        # Fill DTW matrix
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                cost = cost_matrix[i - 1, j - 1]
                dtw[i, j] = cost + min(
                    dtw[i - 1, j],      # insertion
                    dtw[i, j - 1],      # deletion
                    dtw[i - 1, j - 1]   # match
                )
        
        # Normalize by path length
        path_length = n + m
        return dtw[n, m] / path_length if path_length > 0 else 0.0

    
    def compute_joint_angle_correlation(
        self, 
        test_angles: Dict[str, np.ndarray], 
        signature: GaitSignature
    ) -> float:
        """
        Compare test joint angles against signature statistics.
        
        Computes how well the test joint angles match the expected
        distribution from the signature (mean and std).
        
        Args:
            test_angles: Dictionary mapping joint names to angle time series
            signature: The reference GaitSignature
            
        Returns:
            float: Correlation-based similarity score in range [0.0, 1.0]
        """
        if not test_angles or not signature.joint_angle_stats:
            return 0.0
        
        joint_scores = []
        
        for joint_name, angles in test_angles.items():
            if joint_name not in signature.joint_angle_stats:
                continue
            
            if len(angles) == 0:
                continue
            
            stats = signature.joint_angle_stats[joint_name]
            sig_mean = stats.get("mean", 0.0)
            sig_std = stats.get("std", 0.1)  # Default to small std to avoid division by zero
            
            # Compute test statistics
            test_mean = float(np.mean(angles))
            
            # Score based on how close test mean is to signature mean
            # Using a Gaussian-like scoring where being within 2 std gives high score
            if sig_std > 1e-6:
                z_score = abs(test_mean - sig_mean) / sig_std
            else:
                z_score = abs(test_mean - sig_mean) * 10  # Penalize if std is very small
            
            # Convert z-score to similarity (z=0 -> 1.0, z=2 -> ~0.37, z=4 -> ~0.02)
            joint_score = np.exp(-z_score / 2.0)
            joint_scores.append(joint_score)
        
        if not joint_scores:
            return 0.0
        
        return float(np.mean(joint_scores))

    
    def compute_stride_similarity(
        self, 
        test_strides: Dict[str, float], 
        signature: GaitSignature
    ) -> float:
        """
        Compare test stride features against signature.
        
        Args:
            test_strides: Dictionary of stride features from test video
            signature: The reference GaitSignature
            
        Returns:
            float: Similarity score based on feature distances in range [0.0, 1.0]
        """
        if not test_strides or not signature.stride_features:
            return 0.0
        
        feature_scores = []
        
        # Features to compare and their typical ranges for normalization
        feature_weights = {
            "stride_length_mean": 1.0,
            "stride_frequency": 1.0,
            "left_right_symmetry": 0.5,
            "arm_swing_amplitude": 0.8
        }
        
        for feature_name, weight in feature_weights.items():
            if feature_name not in test_strides or feature_name not in signature.stride_features:
                continue
            
            test_val = test_strides[feature_name]
            sig_val = signature.stride_features[feature_name]
            
            # Compute relative difference
            if abs(sig_val) > 1e-6:
                rel_diff = abs(test_val - sig_val) / abs(sig_val)
            else:
                rel_diff = abs(test_val - sig_val)
            
            # Convert to similarity score (0 diff -> 1.0, large diff -> 0.0)
            feature_score = np.exp(-rel_diff * 2.0)
            feature_scores.append(feature_score * weight)
        
        if not feature_scores:
            return 0.0
        
        # Weighted average
        total_weight = sum(feature_weights.get(f, 1.0) for f in feature_weights if f in test_strides)
        if total_weight > 0:
            return float(sum(feature_scores) / total_weight)
        return 0.0

    
    def compute_temporal_consistency(self, sequence: np.ndarray) -> Tuple[float, List[str]]:
        """
        Detect AI-generated video artifacts through temporal analysis.
        
        AI-generated videos often have:
        - Unnatural frame-to-frame jitter (high-frequency noise)
        - Too smooth motion (lack of natural micro-movements)
        - Inconsistent velocity patterns
        
        Args:
            sequence: Normalized keypoint sequence (T, 33, 2)
            
        Returns:
            Tuple of (naturalness_score, list of detected issues)
        """
        issues = []
        scores = []
        
        if sequence.shape[0] < 10:
            return 0.5, ["Insufficient frames for temporal analysis"]
        
        # Flatten sequence for analysis: (T, 33, 2) -> (T, 66)
        flat_seq = sequence.reshape(sequence.shape[0], -1)
        
        # 1. Compute frame-to-frame velocity
        velocity = np.diff(flat_seq, axis=0)
        
        # 2. Compute acceleration (second derivative)
        acceleration = np.diff(velocity, axis=0)
        
        # 3. Jitter detection - AI videos often have high-frequency noise
        # Real human motion is smooth; AI motion can have micro-jitters
        jitter_magnitude = np.std(acceleration)
        
        # Typical human motion has acceleration std in range [0.01, 0.1]
        # AI-generated often has either very low (too smooth) or very high (jittery)
        if jitter_magnitude < 0.005:
            issues.append("Motion is unnaturally smooth (possible AI generation)")
            scores.append(0.3)
        elif jitter_magnitude > 0.15:
            issues.append("High motion jitter detected (possible AI artifacts)")
            scores.append(0.4)
        else:
            scores.append(0.9)
        
        # 4. Velocity consistency - real walking has periodic velocity patterns
        velocity_magnitude = np.linalg.norm(velocity, axis=1)
        velocity_std = np.std(velocity_magnitude)
        velocity_mean = np.mean(velocity_magnitude)
        
        # Coefficient of variation for velocity
        if velocity_mean > 1e-6:
            cv = velocity_std / velocity_mean
            if cv < 0.1:
                issues.append("Velocity too consistent (unnatural uniformity)")
                scores.append(0.4)
            elif cv > 2.0:
                issues.append("Velocity highly erratic (unnatural variation)")
                scores.append(0.4)
            else:
                scores.append(0.85)
        else:
            scores.append(0.5)
        
        # 5. Check for temporal discontinuities (sudden jumps)
        velocity_jumps = np.abs(np.diff(velocity_magnitude))
        max_jump = np.max(velocity_jumps) if len(velocity_jumps) > 0 else 0
        mean_velocity = np.mean(velocity_magnitude) if len(velocity_magnitude) > 0 else 1
        
        if mean_velocity > 1e-6 and max_jump > 5 * mean_velocity:
            issues.append("Temporal discontinuity detected (sudden motion jump)")
            scores.append(0.3)
        else:
            scores.append(0.9)
        
        # 6. Check motion periodicity (walking should be periodic)
        # Use autocorrelation to detect periodicity
        if len(velocity_magnitude) > 20:
            # Normalize velocity for autocorrelation
            v_norm = velocity_magnitude - np.mean(velocity_magnitude)
            v_std = np.std(v_norm)
            if v_std > 1e-6:
                v_norm = v_norm / v_std
                autocorr = np.correlate(v_norm, v_norm, mode='full')
                autocorr = autocorr[len(autocorr)//2:]  # Take positive lags
                autocorr = autocorr / autocorr[0]  # Normalize
                
                # Look for periodic peaks (walking cycle ~0.5-1.5 seconds)
                # Assuming ~30fps, look for peaks at 15-45 frames
                if len(autocorr) > 45:
                    periodic_region = autocorr[15:45]
                    max_periodic_corr = np.max(periodic_region)
                    
                    if max_periodic_corr < 0.3:
                        issues.append("Weak gait periodicity (unnatural walking rhythm)")
                        scores.append(0.5)
                    else:
                        scores.append(0.9)
                else:
                    scores.append(0.7)
            else:
                scores.append(0.5)
        else:
            scores.append(0.7)
        
        # Combine scores
        naturalness_score = float(np.mean(scores)) if scores else 0.5
        
        return naturalness_score, issues
    
    def compute_biomechanical_validity(self, sequence: np.ndarray) -> Tuple[float, List[str]]:
        """
        Check if joint movements obey biomechanical constraints.
        
        AI-generated videos may violate physical constraints:
        - Joint angles exceeding human limits
        - Impossible limb length changes
        - Unnatural joint velocity
        
        Args:
            sequence: Normalized keypoint sequence (T, 33, 2)
            
        Returns:
            Tuple of (validity_score, list of violations)
        """
        violations = []
        scores = []
        
        if sequence.shape[0] < 5:
            return 0.5, ["Insufficient frames for biomechanical analysis"]
        
        # Compute joint angles over time
        joint_angles = self.keypoint_extractor.compute_joint_angles(sequence)
        
        # Human joint angle limits (in radians) - more permissive for 2D projection
        # Note: 2D projection can make angles appear outside normal 3D limits
        joint_limits = {
            "left_knee": (0.0, 3.14),      # 0° to 180° (full range for 2D)
            "right_knee": (0.0, 3.14),
            "left_elbow": (0.0, 3.14),     # 0° to 180°
            "right_elbow": (0.0, 3.14),
            "left_hip": (0.3, 3.14),       # ~17° to 180°
            "right_hip": (0.3, 3.14),
        }
        
        for joint_name, angles in joint_angles.items():
            if len(angles) == 0:
                continue
            
            if joint_name in joint_limits:
                min_limit, max_limit = joint_limits[joint_name]
                
                # Check for violations - only flag extreme violations
                below_min = np.sum(angles < min_limit - 0.3)  # Allow tolerance
                above_max = np.sum(angles > max_limit + 0.3)
                
                violation_rate = (below_min + above_max) / len(angles)
                
                if violation_rate > 0.2:  # More than 20% frames violate limits
                    violations.append(f"{joint_name} exceeds human limits ({violation_rate:.0%} of frames)")
                    scores.append(0.3)
                elif violation_rate > 0.05:
                    scores.append(0.7)
                else:
                    scores.append(0.95)
                
                # Check for unnatural angle velocity (too fast changes)
                if len(angles) > 1:
                    angle_velocity = np.abs(np.diff(angles))
                    max_velocity = np.max(angle_velocity)
                    mean_velocity = np.mean(angle_velocity)
                    
                    # AI videos often have sudden jumps - check for outliers
                    # Human joints typically don't change more than ~0.8 rad/frame at 30fps
                    if max_velocity > 0.8 and max_velocity > 5 * mean_velocity:
                        violations.append(f"{joint_name} has sudden unnatural movement")
                        scores.append(0.4)
        
        # Check limb length consistency
        # In real videos, limb lengths should be relatively constant
        # AI videos may have limbs that stretch/shrink significantly
        
        # Left arm length (shoulder to wrist)
        left_arm_lengths = []
        for t in range(sequence.shape[0]):
            shoulder = sequence[t, 11]  # Left shoulder
            wrist = sequence[t, 15]     # Left wrist
            length = np.linalg.norm(wrist - shoulder)
            if length > 0.01:  # Filter out invalid measurements
                left_arm_lengths.append(length)
        
        if len(left_arm_lengths) > 5:
            arm_length_std = np.std(left_arm_lengths)
            arm_length_mean = np.mean(left_arm_lengths)
            
            if arm_length_mean > 1e-6:
                arm_cv = arm_length_std / arm_length_mean
                if arm_cv > 0.25:  # More than 25% variation is suspicious
                    violations.append(f"Limb length varies unnaturally (CV={arm_cv:.2f})")
                    scores.append(0.4)
                elif arm_cv > 0.18:
                    scores.append(0.7)
                else:
                    scores.append(0.9)
        
        # Check for temporal smoothness of keypoint positions
        # AI videos often have micro-jitters or unnatural smoothness
        flat_seq = sequence.reshape(sequence.shape[0], -1)
        velocity = np.diff(flat_seq, axis=0)
        acceleration = np.diff(velocity, axis=0)
        
        # Compute jerk (third derivative) - AI videos often have high jerk
        if len(acceleration) > 1:
            jerk = np.diff(acceleration, axis=0)
            jerk_magnitude = np.mean(np.abs(jerk))
            
            # Very high jerk indicates unnatural motion
            if jerk_magnitude > 0.05:
                violations.append("High motion jerk detected (unnatural acceleration changes)")
                scores.append(0.4)
            elif jerk_magnitude > 0.03:
                scores.append(0.7)
            else:
                scores.append(0.9)
        
        # Combine scores
        validity_score = float(np.mean(scores)) if scores else 0.7
        
        return validity_score, violations

    
    def compute_authenticity_score(
        self, 
        test_sequence: np.ndarray, 
        signature: GaitSignature
    ) -> Tuple[float, List[str], Dict[str, float]]:
        """
        Compute overall authenticity score combining multiple metrics.
        
        This includes both identity matching (does gait match the person?)
        and deepfake detection (is this real human motion?).
        
        Args:
            test_sequence: Normalized keypoint sequence from test video (T, 33, 2)
            signature: The reference GaitSignature
            
        Returns:
            Tuple containing:
                - float: Overall authenticity score in range [0.0, 1.0]
                - List[str]: Reasons explaining the score components
                - Dict[str, float]: Individual component scores
        """
        reasons = []
        component_scores = {}
        
        # === IDENTITY MATCHING METRICS ===
        
        # 1. DTW similarity against sample sequences
        dtw_scores = []
        for sample_seq in signature.sample_sequences:
            dtw_score = self.compute_dtw_similarity(test_sequence, sample_seq)
            dtw_scores.append(dtw_score)
        
        dtw_similarity = float(np.max(dtw_scores)) if dtw_scores else 0.0
        component_scores["dtw_similarity"] = dtw_similarity
        
        if dtw_similarity < 0.5:
            reasons.append(f"Low DTW similarity ({dtw_similarity:.2f}): movement pattern differs significantly from reference")
        elif dtw_similarity >= 0.7:
            reasons.append(f"Good DTW similarity ({dtw_similarity:.2f}): movement pattern matches reference well")
        
        # 2. Joint angle correlation
        test_angles = self.keypoint_extractor.compute_joint_angles(test_sequence)
        angle_similarity = self.compute_joint_angle_correlation(test_angles, signature)
        component_scores["joint_angle_similarity"] = angle_similarity
        
        if angle_similarity < 0.5:
            reasons.append(f"Low joint angle correlation ({angle_similarity:.2f}): joint movements differ from reference")
        elif angle_similarity >= 0.7:
            reasons.append(f"Good joint angle correlation ({angle_similarity:.2f}): joint movements match reference")
        
        # 3. Stride pattern similarity
        test_strides = self.keypoint_extractor.compute_stride_features(test_sequence)
        stride_similarity = self.compute_stride_similarity(test_strides, signature)
        component_scores["stride_similarity"] = stride_similarity
        
        if stride_similarity < 0.5:
            reasons.append(f"Low stride similarity ({stride_similarity:.2f}): walking rhythm differs from reference")
        elif stride_similarity >= 0.7:
            reasons.append(f"Good stride similarity ({stride_similarity:.2f}): walking rhythm matches reference")
        
        # === DEEPFAKE DETECTION METRICS ===
        
        # 4. Temporal consistency (detects AI-generated motion artifacts)
        temporal_score, temporal_issues = self.compute_temporal_consistency(test_sequence)
        component_scores["temporal_naturalness"] = temporal_score
        
        if temporal_score < 0.6:
            reasons.append(f"Low temporal naturalness ({temporal_score:.2f}): motion appears artificial")
            for issue in temporal_issues:
                reasons.append(f"  - {issue}")
        elif temporal_score >= 0.8:
            reasons.append(f"Good temporal naturalness ({temporal_score:.2f}): motion appears natural")
        
        # 5. Biomechanical validity (detects physically impossible movements)
        biomech_score, biomech_violations = self.compute_biomechanical_validity(test_sequence)
        component_scores["biomechanical_validity"] = biomech_score
        
        if biomech_score < 0.6:
            reasons.append(f"Low biomechanical validity ({biomech_score:.2f}): motion violates physical constraints")
            for violation in biomech_violations:
                reasons.append(f"  - {violation}")
        elif biomech_score >= 0.8:
            reasons.append(f"Good biomechanical validity ({biomech_score:.2f}): motion obeys physical constraints")
        
        # === COMBINE SCORES ===
        # New weights that emphasize deepfake detection
        weights = {
            "dtw_similarity": 0.20,           # Identity matching
            "joint_angle_similarity": 0.15,   # Identity matching
            "stride_similarity": 0.10,        # Identity matching
            "temporal_naturalness": 0.30,     # Deepfake detection (most important)
            "biomechanical_validity": 0.25,   # Deepfake detection
        }
        
        authenticity_score = sum(
            component_scores.get(k, 0.0) * weights[k] for k in weights
        )
        
        # Ensure score is in valid range
        authenticity_score = float(np.clip(authenticity_score, 0.0, 1.0))
        
        return authenticity_score, reasons, component_scores

    
    def verify(self, video_path: str, claimed_identity: str) -> VerificationResult:
        """
        Verify if a video's gait matches the claimed identity's signature.
        
        Args:
            video_path: Path to the test video file
            claimed_identity: The person identity that the video claims to represent
            
        Returns:
            VerificationResult: Complete verification result with score and reasons
        """
        # Check if claimed identity exists in signature store
        if claimed_identity not in self.signatures:
            return VerificationResult.error(
                video_path=video_path,
                claimed_identity=claimed_identity,
                error_message=f"Unknown identity: {claimed_identity}. No signature found.",
                threshold=self.threshold
            )
        
        signature = self.signatures[claimed_identity]
        
        # Extract keypoints from test video
        try:
            keypoints = self.keypoint_extractor.extract_from_video(video_path)
        except FileNotFoundError:
            return VerificationResult.error(
                video_path=video_path,
                claimed_identity=claimed_identity,
                error_message=f"Video file not found: {video_path}",
                threshold=self.threshold
            )
        except ValueError as e:
            return VerificationResult.error(
                video_path=video_path,
                claimed_identity=claimed_identity,
                error_message=str(e),
                threshold=self.threshold
            )
        
        # Check if we got enough frames
        if keypoints.size == 0:
            return VerificationResult.error(
                video_path=video_path,
                claimed_identity=claimed_identity,
                error_message="No pose landmarks detected in video",
                threshold=self.threshold
            )
        
        if keypoints.shape[0] < 10:
            return VerificationResult.error(
                video_path=video_path,
                claimed_identity=claimed_identity,
                error_message=f"Insufficient frames for analysis (minimum 10 required, got {keypoints.shape[0]})",
                threshold=self.threshold
            )
        
        # Normalize the keypoint sequence
        normalized = self.keypoint_extractor.normalize_sequence(keypoints)
        
        # Compute authenticity score
        authenticity_score, reasons, component_scores = self.compute_authenticity_score(
            normalized, signature
        )
        
        # Create success result
        return VerificationResult.success(
            video_path=video_path,
            claimed_identity=claimed_identity,
            authenticity_score=authenticity_score,
            threshold=self.threshold,
            reasons=reasons,
            component_scores=component_scores,
            frames_analyzed=keypoints.shape[0]
        )
    
    def reload_signatures(self) -> None:
        """Reload all signatures from the signatures directory."""
        self.signatures = self._signature_builder.load_all_signatures(self.signatures_dir)
    
    def get_known_identities(self) -> List[str]:
        """Get list of all known identities with stored signatures."""
        return list(self.signatures.keys())
