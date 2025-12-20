"""
Checkpoint validation tests for gait verification extraction components.

This script validates that Tasks 1 and 2 are working correctly:
- GaitSignature and VerificationResult data structures
- KeypointExtractor with all its methods
"""

import sys
import os
import numpy as np
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.verification import GaitSignature, VerificationResult, KeypointExtractor


def test_gait_signature_creation():
    """Test GaitSignature dataclass creation."""
    print("Testing GaitSignature creation...")
    
    signature = GaitSignature(
        identity="TestPerson",
        mean_keypoints=np.random.rand(33, 2),
        keypoint_variance=np.random.rand(33, 2),
        joint_angle_stats={
            "left_knee": {"mean": 2.1, "std": 0.15},
            "right_knee": {"mean": 2.0, "std": 0.14}
        },
        stride_features={
            "stride_length_mean": 0.45,
            "stride_frequency": 1.8
        },
        sample_sequences=[np.random.rand(30, 33, 2)],
        num_training_samples=5,
        created_at=datetime.now().isoformat()
    )
    
    assert signature.identity == "TestPerson"
    assert signature.mean_keypoints.shape == (33, 2)
    assert signature.num_training_samples == 5
    print("  ✓ GaitSignature creation works correctly")
    return True


def test_gait_signature_serialization():
    """Test GaitSignature to_dict and from_dict methods."""
    print("Testing GaitSignature serialization...")
    
    original = GaitSignature(
        identity="TestPerson",
        mean_keypoints=np.random.rand(33, 2),
        keypoint_variance=np.random.rand(33, 2),
        joint_angle_stats={
            "left_knee": {"mean": 2.1, "std": 0.15},
            "right_knee": {"mean": 2.0, "std": 0.14}
        },
        stride_features={
            "stride_length_mean": 0.45,
            "stride_frequency": 1.8
        },
        sample_sequences=[np.random.rand(30, 33, 2)],
        num_training_samples=5,
        created_at="2025-12-20T10:30:00"
    )
    
    # Serialize to dict
    data = original.to_dict()
    assert isinstance(data, dict)
    assert data["identity"] == "TestPerson"
    
    # Deserialize back
    restored = GaitSignature.from_dict(data)
    assert restored.identity == original.identity
    assert np.allclose(restored.mean_keypoints, original.mean_keypoints)
    assert restored.num_training_samples == original.num_training_samples
    print("  ✓ GaitSignature serialization round-trip works correctly")
    return True


def test_verification_result_success():
    """Test VerificationResult success factory method."""
    print("Testing VerificationResult success creation...")
    
    result = VerificationResult.success(
        video_path="test.mp4",
        claimed_identity="TestPerson",
        authenticity_score=0.85,
        threshold=0.7,
        reasons=["High DTW similarity", "Joint angles match"],
        component_scores={"dtw": 0.9, "angles": 0.8, "stride": 0.85},
        frames_analyzed=100
    )
    
    assert result.status == "success"
    assert result.is_authentic == True  # 0.85 >= 0.7
    assert result.authenticity_score == 0.85
    assert result.error_message is None
    print("  ✓ VerificationResult.success() works correctly")
    return True


def test_verification_result_error():
    """Test VerificationResult error factory method."""
    print("Testing VerificationResult error creation...")
    
    result = VerificationResult.error(
        video_path="test.mp4",
        claimed_identity="UnknownPerson",
        error_message="Unknown identity: UnknownPerson. No signature found."
    )
    
    assert result.status == "error"
    assert result.is_authentic == False
    assert result.error_message is not None
    assert "Unknown identity" in result.error_message
    print("  ✓ VerificationResult.error() works correctly")
    return True


def test_keypoint_extractor_initialization():
    """Test KeypointExtractor initialization."""
    print("Testing KeypointExtractor initialization...")
    
    extractor = KeypointExtractor()
    assert extractor.min_detection_confidence == 0.5
    assert extractor.min_tracking_confidence == 0.5
    assert extractor.pose is not None
    print("  ✓ KeypointExtractor initialization works correctly")
    return True


def test_keypoint_extractor_normalize_sequence():
    """Test KeypointExtractor normalize_sequence method."""
    print("Testing KeypointExtractor normalize_sequence...")
    
    extractor = KeypointExtractor()
    
    # Create synthetic keypoints (T=10 frames, 33 landmarks, 2 coords)
    keypoints = np.random.rand(10, 33, 2) * 0.5 + 0.25  # Values in [0.25, 0.75]
    
    normalized = extractor.normalize_sequence(keypoints)
    
    assert normalized.shape == keypoints.shape
    
    # Check that mid-hip is approximately at origin for each frame
    for t in range(normalized.shape[0]):
        left_hip = normalized[t, KeypointExtractor.LEFT_HIP]
        right_hip = normalized[t, KeypointExtractor.RIGHT_HIP]
        mid_hip = (left_hip + right_hip) / 2
        assert np.allclose(mid_hip, [0, 0], atol=1e-6), f"Mid-hip not centered at frame {t}"
    
    print("  ✓ KeypointExtractor normalize_sequence works correctly")
    return True


def test_keypoint_extractor_compute_joint_angles():
    """Test KeypointExtractor compute_joint_angles method."""
    print("Testing KeypointExtractor compute_joint_angles...")
    
    extractor = KeypointExtractor()
    
    # Create synthetic keypoints
    keypoints = np.random.rand(10, 33, 2)
    
    angles = extractor.compute_joint_angles(keypoints)
    
    assert isinstance(angles, dict)
    expected_joints = ["left_knee", "right_knee", "left_elbow", "right_elbow", "left_hip", "right_hip"]
    for joint in expected_joints:
        assert joint in angles, f"Missing joint: {joint}"
        assert len(angles[joint]) == 10, f"Wrong length for {joint}"
        # Angles should be in valid range [0, pi]
        assert np.all(angles[joint] >= 0), f"Negative angle for {joint}"
        assert np.all(angles[joint] <= np.pi), f"Angle > pi for {joint}"
    
    print("  ✓ KeypointExtractor compute_joint_angles works correctly")
    return True


def test_keypoint_extractor_compute_stride_features():
    """Test KeypointExtractor compute_stride_features method."""
    print("Testing KeypointExtractor compute_stride_features...")
    
    extractor = KeypointExtractor()
    
    # Create synthetic keypoints with enough frames
    keypoints = np.random.rand(60, 33, 2)
    
    features = extractor.compute_stride_features(keypoints)
    
    assert isinstance(features, dict)
    expected_features = ["stride_length_mean", "stride_length_std", "stride_frequency", 
                        "left_right_symmetry", "arm_swing_amplitude"]
    for feature in expected_features:
        assert feature in features, f"Missing feature: {feature}"
        assert isinstance(features[feature], float), f"Feature {feature} is not float"
    
    print("  ✓ KeypointExtractor compute_stride_features works correctly")
    return True


def test_keypoint_extractor_empty_sequence():
    """Test KeypointExtractor handles empty sequences gracefully."""
    print("Testing KeypointExtractor with empty sequences...")
    
    extractor = KeypointExtractor()
    
    # Empty keypoints
    empty = np.array([]).reshape(0, 33, 2)
    
    normalized = extractor.normalize_sequence(empty)
    assert normalized.shape == (0, 33, 2)
    
    angles = extractor.compute_joint_angles(empty)
    assert all(len(v) == 0 for v in angles.values())
    
    features = extractor.compute_stride_features(empty)
    assert features["stride_length_mean"] == 0.0
    
    print("  ✓ KeypointExtractor handles empty sequences correctly")
    return True


def test_keypoint_extractor_video_extraction():
    """Test KeypointExtractor extract_from_video with a real video."""
    print("Testing KeypointExtractor extract_from_video...")
    
    # Check if test video exists (using augmented videos directory)
    test_video = "data/videos_augmented/Aarav/S/Aarav_S1_original.mp4"
    if not os.path.exists(test_video):
        print(f"  ⚠ Skipping video extraction test - {test_video} not found")
        return True
    
    extractor = KeypointExtractor()
    
    try:
        keypoints = extractor.extract_from_video(test_video)
        
        assert keypoints.ndim == 3, "Keypoints should be 3D array"
        assert keypoints.shape[1] == 33, "Should have 33 landmarks"
        assert keypoints.shape[2] == 2, "Should have x, y coordinates"
        assert keypoints.shape[0] > 0, "Should have at least one frame"
        
        print(f"  ✓ Extracted {keypoints.shape[0]} frames from video")
        
        # Test full pipeline
        normalized = extractor.normalize_sequence(keypoints)
        angles = extractor.compute_joint_angles(normalized)
        features = extractor.compute_stride_features(normalized)
        
        print(f"  ✓ Full extraction pipeline works: {len(angles)} joint angles, {len(features)} stride features")
        
    except Exception as e:
        print(f"  ✗ Video extraction failed: {e}")
        return False
    
    return True


def run_all_tests():
    """Run all checkpoint validation tests."""
    print("=" * 60)
    print("CHECKPOINT 3: Extraction Validation Tests")
    print("=" * 60)
    print()
    
    tests = [
        test_gait_signature_creation,
        test_gait_signature_serialization,
        test_verification_result_success,
        test_verification_result_error,
        test_keypoint_extractor_initialization,
        test_keypoint_extractor_normalize_sequence,
        test_keypoint_extractor_compute_joint_angles,
        test_keypoint_extractor_compute_stride_features,
        test_keypoint_extractor_empty_sequence,
        test_keypoint_extractor_video_extraction,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"  ✗ {test.__name__} failed with exception: {e}")
            failed += 1
        print()
    
    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
