"""Core data structures for gait verification."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from datetime import datetime
import numpy as np
import json


@dataclass
class GaitSignature:
    """
    Stores the authentic gait pattern for a person, built from multiple training videos.
    
    Attributes:
        identity: Person name (e.g., "Aarav")
        mean_keypoints: Mean normalized keypoint positions (33, 2)
        keypoint_variance: Variance per keypoint (33, 2)
        joint_angle_stats: Mean/std for each joint angle
        stride_features: Stride length, frequency, symmetry, etc.
        sample_sequences: Representative gait cycles for DTW comparison
        num_training_samples: Number of videos used to build signature
        created_at: ISO timestamp when signature was created
    """
    identity: str
    mean_keypoints: np.ndarray
    keypoint_variance: np.ndarray
    joint_angle_stats: Dict[str, Dict[str, float]]
    stride_features: Dict[str, float]
    sample_sequences: List[np.ndarray]
    num_training_samples: int
    created_at: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize GaitSignature to a dictionary for JSON storage."""
        return {
            "identity": self.identity,
            "mean_keypoints": self.mean_keypoints.tolist(),
            "keypoint_variance": self.keypoint_variance.tolist(),
            "joint_angle_stats": self.joint_angle_stats,
            "stride_features": self.stride_features,
            "sample_sequences": [seq.tolist() for seq in self.sample_sequences],
            "num_training_samples": self.num_training_samples,
            "created_at": self.created_at
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GaitSignature":
        """Deserialize a dictionary back into a GaitSignature object."""
        return cls(
            identity=data["identity"],
            mean_keypoints=np.array(data["mean_keypoints"]),
            keypoint_variance=np.array(data["keypoint_variance"]),
            joint_angle_stats=data["joint_angle_stats"],
            stride_features=data["stride_features"],
            sample_sequences=[np.array(seq) for seq in data["sample_sequences"]],
            num_training_samples=data["num_training_samples"],
            created_at=data["created_at"]
        )
    



@dataclass
class VerificationResult:
    """
    Structured result from gait verification.
    
    Attributes:
        video_path: Path to the video that was verified
        claimed_identity: The person identity that the video claims to represent
        authenticity_score: Score between 0.0 and 1.0 indicating gait match
        is_authentic: True if score >= threshold
        threshold: The threshold used for classification
        reasons: List of reasons explaining the decision
        component_scores: Individual scores for DTW, angle, stride components
        frames_analyzed: Number of frames processed
        timestamp: ISO timestamp of verification
        status: "success" or "error"
        error_message: Error details if status is "error"
    """
    video_path: str
    claimed_identity: str
    authenticity_score: float
    is_authentic: bool
    threshold: float
    reasons: List[str]
    component_scores: Dict[str, float]
    frames_analyzed: int
    timestamp: str
    status: str
    error_message: Optional[str] = None
    
    @classmethod
    def success(
        cls,
        video_path: str,
        claimed_identity: str,
        authenticity_score: float,
        threshold: float,
        reasons: List[str],
        component_scores: Dict[str, float],
        frames_analyzed: int
    ) -> "VerificationResult":
        """Create a successful verification result."""
        return cls(
            video_path=video_path,
            claimed_identity=claimed_identity,
            authenticity_score=authenticity_score,
            is_authentic=authenticity_score >= threshold,
            threshold=threshold,
            reasons=reasons,
            component_scores=component_scores,
            frames_analyzed=frames_analyzed,
            timestamp=datetime.now().isoformat(),
            status="success",
            error_message=None
        )
    
    @classmethod
    def error(
        cls,
        video_path: str,
        claimed_identity: str,
        error_message: str,
        threshold: float = 0.7
    ) -> "VerificationResult":
        """Create an error verification result."""
        return cls(
            video_path=video_path,
            claimed_identity=claimed_identity,
            authenticity_score=0.0,
            is_authentic=False,
            threshold=threshold,
            reasons=[],
            component_scores={},
            frames_analyzed=0,
            timestamp=datetime.now().isoformat(),
            status="error",
            error_message=error_message
        )
