#!/usr/bin/env python3
"""
Structured Test Suite for Gait-Based Deepfake Detection

Based on research best practices:
- Separate test categories for different evaluation purposes
- Original videos for baseline identity accuracy
- Actual deepfakes for deepfake detection rate
- Augmented videos for robustness testing (separate metric)
- Cross-identity tests for false acceptance rate

Usage:
    python run_structured_tests.py                    # Run all tests
    python run_structured_tests.py --category identity  # Run only identity tests
    python run_structured_tests.py --category deepfake  # Run only deepfake tests
    python run_structured_tests.py --category robustness # Run robustness tests
"""
import json
import os
import sys
import argparse
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import List, Optional, Dict, Any

# Suppress TensorFlow warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from detect import GaitDetector


@dataclass
class TestCase:
    """Single test case definition"""
    video_path: str
    claimed_identity: Optional[str]
    test_name: str
    category: str  # 'identity', 'deepfake', 'robustness', 'cross_identity'
    expected_authentic: Optional[bool]
    description: str = ""


@dataclass
class TestResult:
    """Result of a single test"""
    test_name: str
    category: str
    video_path: str
    claimed_identity: Optional[str]
    expected_authentic: Optional[bool]
    actual_authentic: Optional[bool]
    predicted_identity: Optional[str]
    confidence: float
    is_correct: Optional[bool]
    status: str  # 'success', 'error'
    error_message: Optional[str] = None
    is_low_confidence: bool = False
    frames_analyzed: int = 0


# =============================================================================
# TEST DEFINITIONS - Organized by Category
# =============================================================================

# Category 1: IDENTITY ACCURACY (Original videos only)
# Purpose: Baseline accuracy for person identification
# These should ALL pass with high confidence
IDENTITY_TESTS = [
    TestCase("data/videos_augmented/Aarav/F/Aarav_F1_original.mp4", "Aarav", "identity_aarav", "identity", True, "Original Aarav video"),
    TestCase("data/videos_augmented/Ananya/F/Ananya_F1_original.mp4", "Ananya", "identity_ananya", "identity", True, "Original Ananya video"),
    TestCase("data/videos_augmented/Arhaan/F/Arhaan_F1_original.mp4", "Arhaan", "identity_arhaan", "identity", True, "Original Arhaan video"),
    TestCase("data/videos_augmented/Bharti/F/Bharti_F1_original.mp4", "Bharti", "identity_bharti", "identity", True, "Original Bharti video"),
    TestCase("data/videos_augmented/Devika/F/Devika_F1_original.mp4", "Devika", "identity_devika", "identity", True, "Original Devika video"),
    TestCase("data/videos_augmented/Prakhar/F/Prakhar_F1_original.mp4", "Prakhar", "identity_prakhar", "identity", True, "Original Prakhar video"),
    TestCase("data/videos_augmented/Prayag/F/Prayag_F1_original.mp4", "Prayag", "identity_prayag", "identity", True, "Original Prayag video"),
    TestCase("data/videos_augmented/Som/F/Som_F1_original.mp4", "Som", "identity_som", "identity", True, "Original Som video"),
    TestCase("data/videos_augmented/Teja/F/Teja_F1_original.mp4", "Teja", "identity_teja", "identity", True, "Original Teja video"),
    TestCase("data/videos_augmented/Vedant/F/Vedant_F1_original.mp4", "Vedant", "identity_vedant", "identity", True, "Original Vedant video"),
    TestCase("data/videos_augmented/Vibhav/F/Vibhav_F1_original.mp4", "Vibhav", "identity_vibhav", "identity", True, "Original Vibhav video"),
]

# Category 2: DEEPFAKE DETECTION (Actual deepfakes only)
# Purpose: Measure deepfake detection rate (True Positive Rate for fakes)
# These should be flagged as NOT authentic
# NOTE: Deepfakes MUST have a claimed identity to test properly
#       Without a claim, we can only flag low confidence (which may not trigger)
DEEPFAKE_TESTS = [
    TestCase("data/deepfake/aarav.mp4", "Aarav", "deepfake_aarav_claimed", "deepfake", False, "Deepfake claiming to be Aarav - should fail identity check"),
    # Add more deepfake videos here as they become available
]

# Category 3: ROBUSTNESS TESTING (Augmented real videos)
# Purpose: Test model robustness to transformations (separate metric)
# Note: These are REAL videos with augmentations, NOT deepfakes
# Failures here indicate robustness issues, not deepfake detection failures
ROBUSTNESS_TESTS = [
    TestCase("data/videos_augmented/Aarav/F/Aarav_F1_aug1_temporal.mp4", "Aarav", "robust_aarav_temporal", "robustness", True, "Temporal augmentation"),
    TestCase("data/videos_augmented/Aarav/F/Aarav_F1_aug2_flip.mp4", "Aarav", "robust_aarav_flip", "robustness", True, "Horizontal flip"),
    TestCase("data/videos_augmented/Aarav/F/Aarav_F1_aug3_brightness.mp4", "Aarav", "robust_aarav_brightness", "robustness", True, "Brightness change"),
    TestCase("data/videos_augmented/Aarav/F/Aarav_F1_aug4_blur.mp4", "Aarav", "robust_aarav_blur", "robustness", True, "Blur effect"),
    TestCase("data/videos_augmented/Bharti/F/Bharti_F1_aug1_temporal.mp4", "Bharti", "robust_bharti_temporal", "robustness", True, "Temporal augmentation"),
]

# Category 4: CROSS-IDENTITY (False Acceptance Rate)
# Purpose: Ensure system rejects wrong identity claims
# These should ALL fail (is_authentic = False)
CROSS_IDENTITY_TESTS = [
    TestCase("data/videos_augmented/Aarav/F/Aarav_F1_original.mp4", "Bharti", "cross_aarav_as_bharti", "cross_identity", False, "Aarav claiming to be Bharti"),
    TestCase("data/videos_augmented/Bharti/F/Bharti_F1_original.mp4", "Aarav", "cross_bharti_as_aarav", "cross_identity", False, "Bharti claiming to be Aarav"),
    TestCase("data/videos_augmented/Devika/F/Devika_F1_original.mp4", "Som", "cross_devika_as_som", "cross_identity", False, "Devika claiming to be Som"),
    TestCase("data/videos_augmented/Som/F/Som_F1_original.mp4", "Devika", "cross_som_as_devika", "cross_identity", False, "Som claiming to be Devika"),
]


def get_tests_by_category(category: Optional[str] = None) -> List[TestCase]:
    """Get test cases filtered by category"""
    all_tests = {
        'identity': IDENTITY_TESTS,
        'deepfake': DEEPFAKE_TESTS,
        'robustness': ROBUSTNESS_TESTS,
        'cross_identity': CROSS_IDENTITY_TESTS,
    }
    
    if category is None:
        # Return all tests
        tests = []
        for cat_tests in all_tests.values():
            tests.extend(cat_tests)
        return tests
    
    return all_tests.get(category, [])


def run_test(detector: GaitDetector, test: TestCase, threshold: float = 0.5) -> TestResult:
    """Run a single test case"""
    if not os.path.exists(test.video_path):
        return TestResult(
            test_name=test.test_name,
            category=test.category,
            video_path=test.video_path,
            claimed_identity=test.claimed_identity,
            expected_authentic=test.expected_authentic,
            actual_authentic=None,
            predicted_identity=None,
            confidence=0.0,
            is_correct=None,
            status="error",
            error_message="Video file not found"
        )
    
    try:
        result = detector.detect(test.video_path, threshold=threshold, claimed_identity=test.claimed_identity)
        
        if result.get("status") == "error":
            return TestResult(
                test_name=test.test_name,
                category=test.category,
                video_path=test.video_path,
                claimed_identity=test.claimed_identity,
                expected_authentic=test.expected_authentic,
                actual_authentic=None,
                predicted_identity=None,
                confidence=0.0,
                is_correct=None,
                status="error",
                error_message=result.get("message", "Unknown error")
            )
        
        actual_authentic = result.get("is_authentic", False)
        is_correct = None
        if test.expected_authentic is not None:
            is_correct = (actual_authentic == test.expected_authentic)
        
        return TestResult(
            test_name=test.test_name,
            category=test.category,
            video_path=test.video_path,
            claimed_identity=test.claimed_identity,
            expected_authentic=test.expected_authentic,
            actual_authentic=actual_authentic,
            predicted_identity=result.get("predicted_identity"),
            confidence=result.get("confidence", 0.0),
            is_correct=is_correct,
            status="success",
            is_low_confidence=result.get("is_low_confidence", False),
            frames_analyzed=result.get("frames_analyzed", 0)
        )
        
    except Exception as e:
        return TestResult(
            test_name=test.test_name,
            category=test.category,
            video_path=test.video_path,
            claimed_identity=test.claimed_identity,
            expected_authentic=test.expected_authentic,
            actual_authentic=None,
            predicted_identity=None,
            confidence=0.0,
            is_correct=None,
            status="error",
            error_message=str(e)
        )


def calculate_metrics(results: List[TestResult], category: str) -> Dict[str, Any]:
    """Calculate metrics for a category of tests"""
    category_results = [r for r in results if r.category == category]
    
    if not category_results:
        return {"total": 0, "message": "No tests in this category"}
    
    total = len(category_results)
    errors = sum(1 for r in category_results if r.status == "error")
    successful = [r for r in category_results if r.status == "success"]
    
    if not successful:
        return {
            "total": total,
            "errors": errors,
            "accuracy": 0.0,
            "message": "All tests errored"
        }
    
    correct = sum(1 for r in successful if r.is_correct)
    accuracy = correct / len(successful) if successful else 0.0
    
    # Category-specific metrics
    metrics = {
        "total": total,
        "successful": len(successful),
        "errors": errors,
        "correct": correct,
        "accuracy": accuracy,
    }
    
    if category == "identity":
        # For identity: measure True Positive Rate (correctly identified real people)
        avg_confidence = sum(r.confidence for r in successful) / len(successful)
        metrics["avg_confidence"] = avg_confidence
        metrics["description"] = "Baseline identity accuracy on original videos"
        
    elif category == "deepfake":
        # For deepfake: measure detection rate (correctly flagged as fake)
        detected = sum(1 for r in successful if not r.actual_authentic)
        metrics["detection_rate"] = detected / len(successful) if successful else 0.0
        metrics["false_negatives"] = len(successful) - detected  # Deepfakes that passed as real
        metrics["description"] = "Deepfake detection rate (should be flagged as NOT authentic)"
        
    elif category == "robustness":
        # For robustness: measure how many augmented real videos still pass
        passed = sum(1 for r in successful if r.actual_authentic)
        low_conf = sum(1 for r in successful if r.is_low_confidence)
        metrics["robustness_rate"] = passed / len(successful) if successful else 0.0
        metrics["low_confidence_count"] = low_conf
        metrics["description"] = "Robustness to augmentations (real videos should still pass)"
        
    elif category == "cross_identity":
        # For cross-identity: measure False Acceptance Rate
        rejected = sum(1 for r in successful if not r.actual_authentic)
        metrics["rejection_rate"] = rejected / len(successful) if successful else 0.0
        metrics["false_accepts"] = len(successful) - rejected  # Wrong identities that passed
        metrics["description"] = "Cross-identity rejection (wrong claims should be rejected)"
    
    return metrics


def print_results(results: List[TestResult], metrics: Dict[str, Dict]) -> None:
    """Print formatted test results"""
    print("\n" + "=" * 70)
    print("STRUCTURED TEST RESULTS")
    print("=" * 70)
    
    categories = ['identity', 'deepfake', 'robustness', 'cross_identity']
    category_names = {
        'identity': '1. IDENTITY ACCURACY (Original Videos)',
        'deepfake': '2. DEEPFAKE DETECTION (Actual Deepfakes)',
        'robustness': '3. ROBUSTNESS (Augmented Real Videos)',
        'cross_identity': '4. CROSS-IDENTITY (False Acceptance)',
    }
    
    for cat in categories:
        cat_results = [r for r in results if r.category == cat]
        if not cat_results:
            continue
            
        print(f"\n{category_names[cat]}")
        print("-" * 50)
        
        for r in cat_results:
            if r.status == "error":
                print(f"  ❌ {r.test_name}: ERROR - {r.error_message}")
            else:
                icon = "✅" if r.is_correct else "⚠️"
                auth_str = "AUTHENTIC" if r.actual_authentic else "NOT_AUTH"
                conf = r.confidence if r.confidence else 0.0
                pred = r.predicted_identity if r.predicted_identity else "Unknown"
                print(f"  {icon} {r.test_name}: {auth_str} ({conf:.1%}) → {pred}")
        
        # Category metrics
        m = metrics.get(cat, {})
        if m.get("total", 0) > 0:
            print(f"\n  📊 {m.get('description', '')}")
            print(f"     Accuracy: {m.get('accuracy', 0):.1%} ({m.get('correct', 0)}/{m.get('successful', 0)})")
            
            if cat == "identity":
                print(f"     Avg Confidence: {m.get('avg_confidence', 0):.1%}")
            elif cat == "deepfake":
                print(f"     Detection Rate: {m.get('detection_rate', 0):.1%}")
                print(f"     False Negatives: {m.get('false_negatives', 0)}")
            elif cat == "robustness":
                print(f"     Robustness Rate: {m.get('robustness_rate', 0):.1%}")
                print(f"     Low Confidence: {m.get('low_confidence_count', 0)}")
            elif cat == "cross_identity":
                print(f"     Rejection Rate: {m.get('rejection_rate', 0):.1%}")
                print(f"     False Accepts: {m.get('false_accepts', 0)}")
    
    # Overall summary
    print("\n" + "=" * 70)
    print("OVERALL SUMMARY")
    print("=" * 70)
    
    total_tests = len(results)
    total_errors = sum(1 for r in results if r.status == "error")
    total_correct = sum(1 for r in results if r.is_correct)
    total_successful = sum(1 for r in results if r.status == "success")
    
    print(f"  Total Tests: {total_tests}")
    print(f"  Successful: {total_successful}")
    print(f"  Errors: {total_errors}")
    print(f"  Correct Predictions: {total_correct}/{total_successful} ({total_correct/total_successful:.1%})" if total_successful > 0 else "  No successful tests")
    
    # Key metrics summary
    print(f"\n  KEY METRICS:")
    if 'identity' in metrics and metrics['identity'].get('successful', 0) > 0:
        print(f"    • Identity Accuracy: {metrics['identity'].get('accuracy', 0):.1%}")
    if 'deepfake' in metrics and metrics['deepfake'].get('successful', 0) > 0:
        print(f"    • Deepfake Detection: {metrics['deepfake'].get('detection_rate', 0):.1%}")
    if 'robustness' in metrics and metrics['robustness'].get('successful', 0) > 0:
        print(f"    • Robustness: {metrics['robustness'].get('robustness_rate', 0):.1%}")
    if 'cross_identity' in metrics and metrics['cross_identity'].get('successful', 0) > 0:
        print(f"    • Cross-ID Rejection: {metrics['cross_identity'].get('rejection_rate', 0):.1%}")


def main():
    parser = argparse.ArgumentParser(description="Structured Test Suite for Gait-Based Deepfake Detection")
    parser.add_argument('--category', '-c', choices=['identity', 'deepfake', 'robustness', 'cross_identity'],
                        help='Run only tests in this category')
    parser.add_argument('--threshold', '-t', type=float, default=0.5,
                        help='Detection threshold (default: 0.5)')
    parser.add_argument('--output', '-o', type=str, default='structured_test_results.json',
                        help='Output JSON file')
    args = parser.parse_args()
    
    print("=" * 70)
    print("GAIT-BASED DEEPFAKE DETECTION - STRUCTURED TEST SUITE")
    print("=" * 70)
    print("\nTest Categories:")
    print("  1. Identity Accuracy - Original videos (baseline)")
    print("  2. Deepfake Detection - Actual deepfakes")
    print("  3. Robustness - Augmented real videos (separate metric)")
    print("  4. Cross-Identity - Wrong identity claims")
    
    # Initialize detector
    print("\nInitializing detector...")
    detector = GaitDetector()
    
    # Get tests
    tests = get_tests_by_category(args.category)
    print(f"\nRunning {len(tests)} tests" + (f" (category: {args.category})" if args.category else ""))
    
    # Run tests
    results: List[TestResult] = []
    for i, test in enumerate(tests, 1):
        print(f"\n[{i}/{len(tests)}] {test.test_name} ({test.category})")
        result = run_test(detector, test, args.threshold)
        results.append(result)
    
    # Calculate metrics per category
    metrics = {}
    for cat in ['identity', 'deepfake', 'robustness', 'cross_identity']:
        metrics[cat] = calculate_metrics(results, cat)
    
    # Print results
    print_results(results, metrics)
    
    # Save to JSON
    output_data = {
        "timestamp": datetime.now().isoformat(),
        "threshold": args.threshold,
        "category_filter": args.category,
        "metrics": metrics,
        "results": [asdict(r) for r in results]
    }
    
    with open(args.output, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\n📁 Results saved to: {args.output}")


if __name__ == "__main__":
    main()
