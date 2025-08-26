#!/usr/bin/env python
"""
Simplified validation test to demonstrate framework working
"""

import numpy as np
import pickle
from pathlib import Path

def simple_dtw(trace1, trace2):
    """Simple DTW implementation"""
    n, m = len(trace1), len(trace2)
    dtw = np.full((n+1, m+1), np.inf)
    dtw[0, 0] = 0
    
    for i in range(1, n+1):
        for j in range(1, m+1):
            cost = np.linalg.norm(trace1[i-1, :2] - trace2[j-1, :2])
            dtw[i, j] = cost + min(dtw[i-1, j], dtw[i, j-1], dtw[i-1, j-1])
    
    path_length = np.sum(np.linalg.norm(np.diff(trace1[:, :2], axis=0), axis=1))
    return dtw[n, m], dtw[n, m] / (path_length + 1e-6)

def validate_simple():
    """Simple validation test"""
    print("="*60)
    print("SIMPLE VALIDATION TEST")
    print("="*60)
    
    # Load sample dataset
    dataset_path = Path("real_datasets/sample_validation_dataset.pkl")
    if dataset_path.exists():
        with open(dataset_path, 'rb') as f:
            dataset = pickle.load(f)
        print(f"✅ Loaded validation dataset with {len(dataset['samples'])} samples")
    else:
        print("❌ No validation dataset found")
        return
    
    # Test with our optimized generator
    from optimized_swype_generator import OptimizedSwypeGenerator
    
    generator = OptimizedSwypeGenerator()
    
    # Test a few words
    test_words = ['hello', 'world', 'the', 'computer', 'gesture']
    results = []
    
    print("\nValidation Results:")
    print("-"*60)
    
    for word in test_words:
        # Find real samples for this word
        real_samples = [s for s in dataset['samples'] if s['word'] == word]
        
        if not real_samples:
            continue
        
        # Generate synthetic trace
        try:
            synthetic = generator.generate_trajectory(word, style='natural')
            synthetic_trace = synthetic['trajectory']
            
            # Compare with first real sample
            real_trace = real_samples[0]['trace']
            
            # Calculate DTW distance
            dtw_dist, dtw_norm = simple_dtw(synthetic_trace, real_trace)
            
            # Calculate path lengths
            synth_length = np.sum(np.linalg.norm(np.diff(synthetic_trace[:, :2], axis=0), axis=1))
            real_length = np.sum(np.linalg.norm(np.diff(real_trace[:, :2], axis=0), axis=1))
            length_ratio = synth_length / (real_length + 1e-6)
            
            # Calculate durations
            synth_duration = synthetic_trace[-1, 2] - synthetic_trace[0, 2]
            real_duration = real_trace[-1, 2] - real_trace[0, 2]
            duration_ratio = synth_duration / (real_duration + 1e-6)
            
            results.append({
                'word': word,
                'dtw_normalized': dtw_norm,
                'length_ratio': length_ratio,
                'duration_ratio': duration_ratio,
                'valid': dtw_norm < 0.5  # Simple threshold
            })
            
            print(f"\n'{word}':")
            print(f"  DTW (normalized): {dtw_norm:.3f}")
            print(f"  Path length ratio: {length_ratio:.2f}")
            print(f"  Duration ratio: {duration_ratio:.2f}")
            print(f"  Status: {'✅ Valid' if dtw_norm < 0.5 else '⚠️  High DTW'}")
            
        except Exception as e:
            print(f"Error testing '{word}': {e}")
    
    # Summary
    if results:
        valid_count = sum(r['valid'] for r in results)
        total_count = len(results)
        percent_valid = (valid_count / total_count) * 100
        
        avg_dtw = np.mean([r['dtw_normalized'] for r in results])
        avg_length_ratio = np.mean([r['length_ratio'] for r in results])
        
        print("\n" + "="*60)
        print("VALIDATION SUMMARY")
        print("="*60)
        print(f"Words tested: {total_count}")
        print(f"Valid traces: {valid_count}/{total_count} ({percent_valid:.1f}%)")
        print(f"Average DTW (normalized): {avg_dtw:.3f}")
        print(f"Average path length ratio: {avg_length_ratio:.2f}")
        
        if percent_valid > 60:
            print("\n✅ Generator produces reasonably valid traces")
        else:
            print("\n⚠️  Generator may need tuning")
    
    print("\nNote: Using simulated validation data")
    print("Real human traces would provide better validation")

if __name__ == "__main__":
    validate_simple()