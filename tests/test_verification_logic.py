"""
Checkpoint 6 validation tests for gait verification logic.

This script validates that Tasks 4 and 5 are working correctly:
- SignatureBuilder with all its methods
- GaitVerifier with all its methods
"""

import sys
import os
import tempfile
import shutil
import numpy as np
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.verification import (
    GaitSignature, 
    VerificationResult, 
    KeypointExtractor,
    SignatureBuilder,
    GaitVerifier
)


def create_test_signature(identity: str = "TestPerson") -> GaitSignature:
    """Create a test GaitSignature for testing."""
    return GaitSignature(
        identity=identity,
        mean_keypoints=np.random.rand(33, 2),
        keypoint_variance=np.random.rand(33, 2) * 0.1,
        joint_angle_stats={
            "left_knee": {"mean": 2.1, "std": 0.15},
            "right_knee": {"mean": 2.0, "std": 0.14},
            "left_elbow": {"mean": 2.5, "std": 0.2},
            "right_elbow": {"mean": 2.4, "std": 0.18},
            "left_hip": {"mean": 2.8, "std": 0.25},
            "right_hip": {"mean": 2.7, "std": 0.22}
        },
        stride_features={
            "stride_length_mean": 0.45,
            "stride_length_std": 0.05,
            "stride_frequency": 1.8,
            "left_right_symmetry": 0.98,
            "arm_swing_amplitude": 0.3
        },
        sample_sequences=[np.random.rand(30, 33, 2) for _ in range(3)],
        num_training_samples=5,
        created_at=datetime.now().isoformat()
    )


class TestSignatureBuilder:
    """Tests for SignatureBuilder class."""
    
    def test_initialization(self):
        """Test SignatureBuilder initialization."""
        print("Testing SignatureBuilder initialization...")
        
        builder = SignatureBuilder()
        assert builder.keypoint_extractor is not None
        
        # Test with custom extractor
        extractor = KeypointExtractor()
        builder2 = SignatureBuilder(keypoint_extractor=extractor)
        assert builder2.keypoint_extractor is extractor
        
        print("  ✓ SignatureBuilder initialization works correctly")
        return True
    
    def test_save_and_load_signature(self):
        """Test saving and loading signatures."""
        print("Testing SignatureBuilder save/load signature...")
        
        builder = SignatureBuilder()
        signature = create_test_signature("SaveLoadTest")
        
        # Create temp directory
        temp_dir = tempfile.mkdtemp()
        try:
            # Save signature
            filepath = builder.save_signature(signature, temp_dir)
            assert os.path.exists(filepath)
            assert filepath.endswith("SaveLoadTest_signature.json")
            
            # Load signature
            loaded = builder.load_signature("SaveLoadTest", temp_dir)
            assert loaded.identity == signature.identity
            assert np.allclose(loaded.mean_keypoints, signature.mean_keypoints)
            assert loaded.num_training_samples == signature.num_training_samples
            
            print("  ✓ SignatureBuilder save/load signature works correctly")
            return True
        finally:
            shutil.rmtree(temp_dir)
    
    def test_load_all_signatures(self):
        """Test loading all signatures from directory."""
        print("Testing SignatureBuilder load_all_signatures...")
        
        builder = SignatureBuilder()
        
        # Create temp directory with multiple signatures
        temp_dir = tempfile.mkdtemp()
        try:
            # Save multiple signatures
            for name in ["Person1", "Person2", "Person3"]:
                sig = create_test_signature(name)
                builder.save_signature(sig, temp_dir)
            
            # Load all
            signatures = builder.load_all_signatures(temp_dir)
            assert len(signatures) == 3
            assert "Person1" in signatures
            assert "Person2" in signatures
            assert "Person3" in signatures
            
            print("  ✓ SignatureBuilder load_all_signatures works correctly")
            return True
        finally:
            shutil.rmtree(temp_dir)
    
    def test_load_nonexistent_signature(self):
        """Test loading a signature that doesn't exist."""
        print("Testing SignatureBuilder load nonexistent signature...")
        
        builder = SignatureBuilder()
        temp_dir = tempfile.mkdtemp()
        
        try:
            try:
                builder.load_signature("NonExistent", temp_dir)
                assert False, "Should have raised FileNotFoundError"
            except FileNotFoundError:
                pass
            
            print("  ✓ SignatureBuilder handles nonexistent signature correctly")
            return True
        finally:
            shutil.rmtree(temp_dir)
    
    def test_load_from_empty_directory(self):
        """Test loading signatures from empty directory."""
        print("Testing SignatureBuilder load from empty directory...")
        
        builder = SignatureBuilder()
        temp_dir = tempfile.mkdtemp()
        
        try:
            signatures = builder.load_all_signatures(temp_dir)
            assert len(signatures) == 0
            
            print("  ✓ SignatureBuilder handles empty directory correctly")
            return True
        finally:
            shutil.rmtree(temp_dir)


class TestGaitVerifier:
    """Tests for GaitVerifier class."""
    
    def setup_temp_signatures(self):
        """Create temp directory with test signatures."""
        temp_dir = tempfile.mkdtemp()
        builder = SignatureBuilder()
        
        # Create and save test signatures
        for name in ["Aarav", "Bharti", "Devika"]:
            sig = create_test_signature(name)
            builder.save_signature(sig, temp_dir)
        
        return temp_dir
    
    def test_initialization(self):
        """Test GaitVerifier initialization."""
        print("Testing GaitVerifier initialization...")
        
        temp_dir = self.setup_temp_signatures()
        try:
            verifier = GaitVerifier(temp_dir)
            
            assert verifier.threshold == 0.7  # Default threshold
            assert len(verifier.signatures) == 3
            assert "Aarav" in verifier.signatures
            
            # Test with custom threshold
            verifier2 = GaitVerifier(temp_dir, threshold=0.8)
            assert verifier2.threshold == 0.8
            
            print("  ✓ GaitVerifier initialization works correctly")
            return True
        finally:
            shutil.rmtree(temp_dir)
    
    def test_get_known_identities(self):
        """Test getting list of known identities."""
        print("Testing GaitVerifier get_known_identities...")
        
        temp_dir = self.setup_temp_signatures()
        try:
            verifier = GaitVerifier(temp_dir)
            identities = verifier.get_known_identities()
            
            assert len(identities) == 3
            assert "Aarav" in identities
            assert "Bharti" in identities
            assert "Devika" in identities
            
            print("  ✓ GaitVerifier get_known_identities works correctly")
            return True
        finally:
            shutil.rmtree(temp_dir)
    
    def test_compute_dtw_similarity(self):
        """Test DTW similarity computation."""
        print("Testing GaitVerifier compute_dtw_similarity...")
        
        temp_dir = self.setup_temp_signatures()
        try:
            verifier = GaitVerifier(temp_dir)
            
            # Test with identical sequences
            seq1 = np.random.rand(30, 33, 2)
            similarity = verifier.compute_dtw_similarity(seq1, seq1)
            assert 0.0 <= similarity <= 1.0
            assert similarity > 0.9  # Should be very high for identical sequences
            
            # Test with different sequences
            seq2 = np.random.rand(30, 33, 2) * 10  # Very different
            similarity2 = verifier.compute_dtw_similarity(seq1, seq2)
            assert 0.0 <= similarity2 <= 1.0
            
            # Test with empty sequences
            empty = np.array([]).reshape(0, 33, 2)
            similarity3 = verifier.compute_dtw_similarity(empty, seq1)
            assert similarity3 == 0.0
            
            print("  ✓ GaitVerifier compute_dtw_similarity works correctly")
            return True
        finally:
            shutil.rmtree(temp_dir)
    
    def test_compute_joint_angle_correlation(self):
        """Test joint angle correlation computation."""
        print("Testing GaitVerifier compute_joint_angle_correlation...")
        
        temp_dir = self.setup_temp_signatures()
        try:
            verifier = GaitVerifier(temp_dir)
            signature = verifier.signatures["Aarav"]
            
            # Test with matching angles
            test_angles = {
                "left_knee": np.array([2.1, 2.1, 2.1]),
                "right_knee": np.array([2.0, 2.0, 2.0]),
                "left_elbow": np.array([2.5, 2.5, 2.5]),
                "right_elbow": np.array([2.4, 2.4, 2.4]),
                "left_hip": np.array([2.8, 2.8, 2.8]),
                "right_hip": np.array([2.7, 2.7, 2.7])
            }
            
            correlation = verifier.compute_joint_angle_correlation(test_angles, signature)
            assert 0.0 <= correlation <= 1.0
            assert correlation > 0.8  # Should be high for matching angles
            
            # Test with very different angles
            different_angles = {
                "left_knee": np.array([0.5, 0.5, 0.5]),
                "right_knee": np.array([0.5, 0.5, 0.5]),
            }
            correlation2 = verifier.compute_joint_angle_correlation(different_angles, signature)
            assert 0.0 <= correlation2 <= 1.0
            
            # Test with empty angles
            correlation3 = verifier.compute_joint_angle_correlation({}, signature)
            assert correlation3 == 0.0
            
            print("  ✓ GaitVerifier compute_joint_angle_correlation works correctly")
            return True
        finally:
            shutil.rmtree(temp_dir)
    
    def test_compute_stride_similarity(self):
        """Test stride similarity computation."""
        print("Testing GaitVerifier compute_stride_similarity...")
        
        temp_dir = self.setup_temp_signatures()
        try:
            verifier = GaitVerifier(temp_dir)
            signature = verifier.signatures["Aarav"]
            
            # Test with matching stride features
            test_strides = {
                "stride_length_mean": 0.45,
                "stride_frequency": 1.8,
                "left_right_symmetry": 0.98,
                "arm_swing_amplitude": 0.3
            }
            
            similarity = verifier.compute_stride_similarity(test_strides, signature)
            assert 0.0 <= similarity <= 1.0
            assert similarity > 0.8  # Should be high for matching features
            
            # Test with different features
            different_strides = {
                "stride_length_mean": 0.1,
                "stride_frequency": 0.5,
            }
            similarity2 = verifier.compute_stride_similarity(different_strides, signature)
            assert 0.0 <= similarity2 <= 1.0
            
            # Test with empty features
            similarity3 = verifier.compute_stride_similarity({}, signature)
            assert similarity3 == 0.0
            
            print("  ✓ GaitVerifier compute_stride_similarity works correctly")
            return True
        finally:
            shutil.rmtree(temp_dir)
    
    def test_compute_authenticity_score(self):
        """Test authenticity score computation."""
        print("Testing GaitVerifier compute_authenticity_score...")
        
        temp_dir = self.setup_temp_signatures()
        try:
            verifier = GaitVerifier(temp_dir)
            signature = verifier.signatures["Aarav"]
            
            # Use one of the sample sequences from the signature
            test_sequence = signature.sample_sequences[0]
            
            score, reasons, component_scores = verifier.compute_authenticity_score(
                test_sequence, signature
            )
            
            # Check score bounds
            assert 0.0 <= score <= 1.0
            
            # Check reasons is a list
            assert isinstance(reasons, list)
            
            # Check component scores
            assert "dtw_similarity" in component_scores
            assert "joint_angle_similarity" in component_scores
            assert "stride_similarity" in component_scores
            
            for key, value in component_scores.items():
                assert 0.0 <= value <= 1.0, f"{key} out of bounds: {value}"
            
            print("  ✓ GaitVerifier compute_authenticity_score works correctly")
            return True
        finally:
            shutil.rmtree(temp_dir)
    
    def test_verify_unknown_identity(self):
        """Test verification with unknown identity."""
        print("Testing GaitVerifier verify with unknown identity...")
        
        temp_dir = self.setup_temp_signatures()
        try:
            verifier = GaitVerifier(temp_dir)
            
            # Try to verify against unknown identity
            result = verifier.verify("dummy_video.mp4", "UnknownPerson")
            
            assert result.status == "error"
            assert result.is_authentic == False
            assert "Unknown identity" in result.error_message
            assert result.claimed_identity == "UnknownPerson"
            
            print("  ✓ GaitVerifier handles unknown identity correctly")
            return True
        finally:
            shutil.rmtree(temp_dir)
    
    def test_verify_nonexistent_video(self):
        """Test verification with nonexistent video file."""
        print("Testing GaitVerifier verify with nonexistent video...")
        
        temp_dir = self.setup_temp_signatures()
        try:
            verifier = GaitVerifier(temp_dir)
            
            # Try to verify nonexistent video
            result = verifier.verify("nonexistent_video.mp4", "Aarav")
            
            assert result.status == "error"
            assert result.is_authentic == False
            assert "not found" in result.error_message.lower()
            
            print("  ✓ GaitVerifier handles nonexistent video correctly")
            return True
        finally:
            shutil.rmtree(temp_dir)
    
    def test_verify_with_real_video(self):
        """Test verification with a real video file."""
        print("Testing GaitVerifier verify with real video...")
        
        # Check if test video exists
        test_video = "data/videos_augmented/Aarav/S/Aarav_S1_original.mp4"
        if not os.path.exists(test_video):
            print(f"  ⚠ Skipping real video test - {test_video} not found")
            return True
        
        temp_dir = self.setup_temp_signatures()
        try:
            verifier = GaitVerifier(temp_dir)
            
            # Verify the video
            result = verifier.verify(test_video, "Aarav")
            
            # Should complete successfully (even if not authentic since signature is synthetic)
            assert result.status == "success"
            assert 0.0 <= result.authenticity_score <= 1.0
            assert result.claimed_identity == "Aarav"
            assert result.frames_analyzed > 0
            assert result.threshold == 0.7
            assert isinstance(result.reasons, list)
            assert isinstance(result.component_scores, dict)
            
            print(f"  ✓ Verification completed: score={result.authenticity_score:.2f}, authentic={result.is_authentic}")
            return True
        finally:
            shutil.rmtree(temp_dir)
    
    def test_threshold_classification(self):
        """Test that threshold-based classification is consistent."""
        print("Testing GaitVerifier threshold classification...")
        
        temp_dir = self.setup_temp_signatures()
        try:
            # Test with different thresholds
            for threshold in [0.3, 0.5, 0.7, 0.9]:
                verifier = GaitVerifier(temp_dir, threshold=threshold)
                
                # Create a mock result to verify threshold logic
                result = VerificationResult.success(
                    video_path="test.mp4",
                    claimed_identity="Aarav",
                    authenticity_score=0.6,
                    threshold=threshold,
                    reasons=[],
                    component_scores={},
                    frames_analyzed=100
                )
                
                # Verify threshold classification
                if 0.6 >= threshold:
                    assert result.is_authentic == True, f"Score 0.6 should be authentic with threshold {threshold}"
                else:
                    assert result.is_authentic == False, f"Score 0.6 should not be authentic with threshold {threshold}"
            
            print("  ✓ GaitVerifier threshold classification works correctly")
            return True
        finally:
            shutil.rmtree(temp_dir)
    
    def test_reload_signatures(self):
        """Test reloading signatures."""
        print("Testing GaitVerifier reload_signatures...")
        
        temp_dir = self.setup_temp_signatures()
        try:
            verifier = GaitVerifier(temp_dir)
            assert len(verifier.signatures) == 3
            
            # Add a new signature
            builder = SignatureBuilder()
            new_sig = create_test_signature("NewPerson")
            builder.save_signature(new_sig, temp_dir)
            
            # Reload
            verifier.reload_signatures()
            assert len(verifier.signatures) == 4
            assert "NewPerson" in verifier.signatures
            
            print("  ✓ GaitVerifier reload_signatures works correctly")
            return True
        finally:
            shutil.rmtree(temp_dir)


def run_all_tests():
    """Run all checkpoint 6 validation tests."""
    print("=" * 60)
    print("CHECKPOINT 6: Verification Logic Validation Tests")
    print("=" * 60)
    print()
    
    passed = 0
    failed = 0
    
    # SignatureBuilder tests
    print("-" * 40)
    print("SignatureBuilder Tests")
    print("-" * 40)
    
    sb_tests = TestSignatureBuilder()
    for test_method in [
        sb_tests.test_initialization,
        sb_tests.test_save_and_load_signature,
        sb_tests.test_load_all_signatures,
        sb_tests.test_load_nonexistent_signature,
        sb_tests.test_load_from_empty_directory,
    ]:
        try:
            if test_method():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"  ✗ {test_method.__name__} failed with exception: {e}")
            failed += 1
        print()
    
    # GaitVerifier tests
    print("-" * 40)
    print("GaitVerifier Tests")
    print("-" * 40)
    
    gv_tests = TestGaitVerifier()
    for test_method in [
        gv_tests.test_initialization,
        gv_tests.test_get_known_identities,
        gv_tests.test_compute_dtw_similarity,
        gv_tests.test_compute_joint_angle_correlation,
        gv_tests.test_compute_stride_similarity,
        gv_tests.test_compute_authenticity_score,
        gv_tests.test_verify_unknown_identity,
        gv_tests.test_verify_nonexistent_video,
        gv_tests.test_verify_with_real_video,
        gv_tests.test_threshold_classification,
        gv_tests.test_reload_signatures,
    ]:
        try:
            if test_method():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"  ✗ {test_method.__name__} failed with exception: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
        print()
    
    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
