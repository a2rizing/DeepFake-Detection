"""
Gait Verification System
Verifies if a person's gait matches their claimed identity
This is Phase 2: Building towards deepfake detection
"""

import os
import json
import numpy as np
import joblib
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

class GaitVerificationSystem:
    """
    System to verify if a person's gait matches their claimed identity
    
    Use Cases:
    1. Verify if test video matches person's known gait profile
    2. Detect anomalous gait patterns (potential deepfakes)
    3. Calculate confidence scores for authentication
    """
    
    def __init__(self, models_dir='models', data_dir='data/processed'):
        self.models_dir = models_dir
        self.data_dir = data_dir
        self.profiles = {}
        self.label_mapping = {}
        self.scaler = None
        
        # Load existing data
        self.load_gait_profiles()
    
    def load_gait_profiles(self):
        """Load pre-trained gait profiles for each person"""
        
        print("📊 Loading gait profiles...")
        
        # Load labels
        labels_path = os.path.join(self.data_dir, 'labels.json')
        if os.path.exists(labels_path):
            with open(labels_path, 'r') as f:
                self.label_mapping = json.load(f)
            
            # Reverse mapping
            self.id_to_name = {v: k for k, v in self.label_mapping.items()}
        
        # Load feature data
        X_path = os.path.join(self.data_dir, 'X.npy')
        y_path = os.path.join(self.data_dir, 'y.npy')
        
        if os.path.exists(X_path) and os.path.exists(y_path):
            X = np.load(X_path)
            y = np.load(y_path)
            
            # Build profile for each person
            for person_id in np.unique(y):
                person_name = self.id_to_name[person_id]
                person_data = X[y == person_id]
                
                # Store profile
                self.profiles[person_name] = {
                    'id': person_id,
                    'reference_gaits': person_data,
                    'mean_gait': np.mean(person_data, axis=0),
                    'std_gait': np.std(person_data, axis=0),
                    'n_samples': len(person_data)
                }
            
            print(f"✅ Loaded {len(self.profiles)} gait profiles")
            for name, profile in self.profiles.items():
                print(f"   - {name.capitalize()}: {profile['n_samples']} sample(s)")
        else:
            print("❌ No gait profiles found. Please run training first.")
    
    def verify_gait(self, test_gait, claimed_identity, method='cosine'):
        """
        Verify if test gait matches the claimed identity
        
        Args:
            test_gait: numpy array of gait features (shape: sequence_length, features)
            claimed_identity: name of person (e.g., "aditya")
            method: 'cosine', 'euclidean', or 'mahalanobis'
        
        Returns:
            dict with verification results
        """
        
        if claimed_identity not in self.profiles:
            return {
                'verified': False,
                'reason': f'Unknown identity: {claimed_identity}',
                'confidence': 0.0
            }
        
        profile = self.profiles[claimed_identity]
        
        # Flatten test gait
        test_flat = test_gait.flatten() if test_gait.ndim > 1 else test_gait
        reference_flat = profile['mean_gait'].flatten()
        
        # Calculate similarity based on method
        if method == 'cosine':
            similarity = cosine_similarity([test_flat], [reference_flat])[0][0]
            distance = 1 - similarity
            threshold = 0.3  # Lower distance = more similar
            
        elif method == 'euclidean':
            distance = euclidean_distances([test_flat], [reference_flat])[0][0]
            # Normalize by feature dimension
            distance = distance / np.sqrt(len(test_flat))
            threshold = 0.5
            
        else:  # mahalanobis or default to euclidean
            distance = euclidean_distances([test_flat], [reference_flat])[0][0]
            distance = distance / np.sqrt(len(test_flat))
            threshold = 0.5
        
        # Calculate all distances to all profiles (for ranking)
        all_distances = {}
        for name, prof in self.profiles.items():
            ref = prof['mean_gait'].flatten()
            if method == 'cosine':
                sim = cosine_similarity([test_flat], [ref])[0][0]
                all_distances[name] = 1 - sim
            else:
                dist = euclidean_distances([test_flat], [ref])[0][0]
                all_distances[name] = dist / np.sqrt(len(test_flat))
        
        # Sort by distance (closest first)
        ranked_matches = sorted(all_distances.items(), key=lambda x: x[1])
        best_match = ranked_matches[0][0]
        
        # Verification decision
        verified = distance < threshold
        confidence = max(0.0, min(1.0, 1 - (distance / threshold)))
        
        return {
            'verified': verified,
            'claimed_identity': claimed_identity,
            'best_match': best_match,
            'distance': float(distance),
            'confidence': float(confidence),
            'threshold': threshold,
            'method': method,
            'all_distances': {k: float(v) for k, v in all_distances.items()},
            'ranked_matches': [(name, float(dist)) for name, dist in ranked_matches],
            'match_status': 'AUTHENTIC' if verified and best_match == claimed_identity else 'SUSPICIOUS'
        }
    
    def detect_anomaly(self, test_gait, expected_identity=None):
        """
        Detect if gait is anomalous (potential deepfake)
        
        Args:
            test_gait: gait features to test
            expected_identity: if provided, specifically check against this person
        
        Returns:
            dict with anomaly detection results
        """
        
        test_flat = test_gait.flatten() if test_gait.ndim > 1 else test_gait
        
        if expected_identity:
            # Check against specific person
            result = self.verify_gait(test_gait, expected_identity)
            is_anomaly = not result['verified']
            
            return {
                'is_anomaly': is_anomaly,
                'expected_identity': expected_identity,
                'actual_best_match': result['best_match'],
                'confidence': result['confidence'],
                'status': 'DEEPFAKE SUSPECTED' if is_anomaly else 'AUTHENTIC',
                'details': result
            }
        else:
            # Check against all profiles
            all_results = {}
            for name in self.profiles.keys():
                all_results[name] = self.verify_gait(test_gait, name)
            
            # Find best match
            best_match = min(all_results.items(), 
                           key=lambda x: x[1]['distance'])
            
            # Check if best match is confident enough
            is_anomaly = best_match[1]['confidence'] < 0.7
            
            return {
                'is_anomaly': is_anomaly,
                'best_match': best_match[0],
                'confidence': best_match[1]['confidence'],
                'status': 'UNKNOWN PERSON' if is_anomaly else 'RECOGNIZED',
                'all_results': {k: v['distance'] for k, v in all_results.items()}
            }
    
    def visualize_verification(self, verification_result, save_path=None):
        """Create visualization of verification results"""
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Distance comparison bar chart
        distances = verification_result['all_distances']
        names = list(distances.keys())
        values = list(distances.values())
        colors = ['green' if name == verification_result['claimed_identity'] else 'red' 
                 if name == verification_result['best_match'] else 'gray' 
                 for name in names]
        
        ax1.barh(names, values, color=colors)
        ax1.axvline(x=verification_result['threshold'], color='blue', 
                   linestyle='--', label='Threshold')
        ax1.set_xlabel('Distance (lower = more similar)', fontweight='bold')
        ax1.set_title('Gait Distance from All Profiles', fontweight='bold')
        ax1.legend()
        ax1.grid(axis='x', alpha=0.3)
        
        # Verification status
        ax2.axis('off')
        
        status_text = f"""
VERIFICATION RESULTS
{'='*40}

Claimed Identity: {verification_result['claimed_identity'].upper()}
Best Match: {verification_result['best_match'].upper()}

Status: {verification_result['match_status']}
Confidence: {verification_result['confidence']:.1%}

Distance: {verification_result['distance']:.4f}
Threshold: {verification_result['threshold']:.4f}

Method: {verification_result['method'].upper()}

{'✅ VERIFIED' if verification_result['verified'] else '⚠️ SUSPICIOUS'}
        """
        
        bg_color = 'lightgreen' if verification_result['verified'] else 'lightcoral'
        ax2.text(0.5, 0.5, status_text, 
                ha='center', va='center',
                fontsize=11, family='monospace',
                bbox=dict(boxstyle='round', facecolor=bg_color, alpha=0.3))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Visualization saved: {save_path}")
        
        plt.show()
        
        return fig


def demo_gait_verification():
    """
    Demonstration of gait verification system
    Shows how to use the system for authentication
    """
    
    print("\n" + "="*80)
    print("🎯 GAIT VERIFICATION SYSTEM DEMO")
    print("="*80)
    
    # Initialize system
    verifier = GaitVerificationSystem()
    
    if not verifier.profiles:
        print("\n❌ No gait profiles loaded. Please run training first.")
        return
    
    print("\n📋 Available Profiles:")
    for name in verifier.profiles.keys():
        print(f"   - {name.capitalize()}")
    
    # Load test data
    X = np.load('data/processed/X.npy')
    y = np.load('data/processed/y.npy')
    
    print("\n" + "="*80)
    print("TEST 1: Authentic Verification")
    print("="*80)
    
    # Test 1: Verify with correct identity
    person_id = 0  # Aditya
    person_name = verifier.id_to_name[person_id]
    test_gait = X[y == person_id][0]  # Use first sample
    
    print(f"\n🎬 Testing video of: {person_name.upper()}")
    print(f"   Claimed identity: {person_name}")
    
    result1 = verifier.verify_gait(test_gait, person_name, method='cosine')
    
    print(f"\n📊 Results:")
    print(f"   Status: {result1['match_status']}")
    print(f"   Verified: {result1['verified']}")
    print(f"   Confidence: {result1['confidence']:.1%}")
    print(f"   Distance: {result1['distance']:.4f}")
    print(f"   Best Match: {result1['best_match']}")
    
    # Visualize
    verifier.visualize_verification(result1, 
                                    'results/verification_authentic.png')
    
    print("\n" + "="*80)
    print("TEST 2: Suspicious Video (Simulated Deepfake)")
    print("="*80)
    
    # Test 2: Claim wrong identity (simulate deepfake)
    actual_person = verifier.id_to_name[2]  # Krees
    claimed_person = verifier.id_to_name[0]  # Claim it's Aditya
    test_gait2 = X[y == 2][0]
    
    print(f"\n🎬 Testing video of: {actual_person.upper()}")
    print(f"   BUT claimed identity: {claimed_person} (WRONG!)")
    
    result2 = verifier.verify_gait(test_gait2, claimed_person, method='cosine')
    
    print(f"\n📊 Results:")
    print(f"   Status: {result2['match_status']}")
    print(f"   Verified: {result2['verified']}")
    print(f"   Confidence: {result2['confidence']:.1%}")
    print(f"   Distance: {result2['distance']:.4f}")
    print(f"   Best Match: {result2['best_match']} ← ACTUAL PERSON")
    
    # Visualize
    verifier.visualize_verification(result2, 
                                    'results/verification_suspicious.png')
    
    print("\n" + "="*80)
    print("TEST 3: Anomaly Detection")
    print("="*80)
    
    # Test 3: Detect anomaly
    print(f"\n🎬 Testing unknown gait against {claimed_person}'s profile")
    
    anomaly_result = verifier.detect_anomaly(test_gait2, claimed_person)
    
    print(f"\n📊 Anomaly Detection Results:")
    print(f"   Is Anomaly: {anomaly_result['is_anomaly']}")
    print(f"   Status: {anomaly_result['status']}")
    print(f"   Expected: {anomaly_result['expected_identity']}")
    print(f"   Actual Best Match: {anomaly_result['actual_best_match']}")
    print(f"   Confidence: {anomaly_result['confidence']:.1%}")
    
    print("\n" + "="*80)
    print("✅ DEMO COMPLETE")
    print("="*80)
    
    print("\n💡 Key Insights:")
    print("   • System can verify authentic gaits with high confidence")
    print("   • System detects when gait doesn't match claimed identity")
    print("   • This is the foundation for deepfake detection")
    print("   • Next step: Add face recognition for complete system")


if __name__ == "__main__":
    # Create results directory
    os.makedirs('results', exist_ok=True)
    
    # Run demo
    demo_gait_verification()
