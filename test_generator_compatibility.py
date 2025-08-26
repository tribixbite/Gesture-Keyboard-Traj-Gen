#!/usr/bin/env python
"""
Test compatibility between different generators
"""

import numpy as np
from optimized_swype_generator import OptimizedSwypeGenerator
from improved_swype_generator import ImprovedSwypeGenerator

def analyze_trace_differences():
    """
    Analyze differences between generator outputs
    """
    print("="*60)
    print("GENERATOR OUTPUT ANALYSIS")
    print("="*60)
    
    word = "hello"
    
    # Generate with both
    opt_gen = OptimizedSwypeGenerator()
    imp_gen = ImprovedSwypeGenerator()
    
    opt_result = opt_gen.generate_trajectory(word)
    imp_result = imp_gen.generate_trajectory(word)
    
    opt_trace = opt_result['trajectory']
    imp_trace = imp_result['trajectory']
    
    print(f"\nAnalyzing traces for word: '{word}'")
    print("-"*40)
    
    # Basic stats
    print(f"Optimized Generator:")
    print(f"  Points: {len(opt_trace)}")
    print(f"  Shape: {opt_trace.shape}")
    print(f"  X range: [{opt_trace[:,0].min():.1f}, {opt_trace[:,0].max():.1f}]")
    print(f"  Y range: [{opt_trace[:,1].min():.1f}, {opt_trace[:,1].max():.1f}]")
    print(f"  Duration: {opt_result.get('duration', 0):.2f}s")
    
    print(f"\nImproved Generator:")
    print(f"  Points: {len(imp_trace)}")
    print(f"  Shape: {imp_trace.shape}")
    print(f"  X range: [{imp_trace[:,0].min():.1f}, {imp_trace[:,0].max():.1f}]")
    print(f"  Y range: [{imp_trace[:,1].min():.1f}, {imp_trace[:,1].max():.1f}]")
    print(f"  Duration: {imp_result.get('duration', 0):.2f}s")
    
    # Check scale differences
    opt_span = np.max([opt_trace[:,0].max() - opt_trace[:,0].min(),
                       opt_trace[:,1].max() - opt_trace[:,1].min()])
    imp_span = np.max([imp_trace[:,0].max() - imp_trace[:,0].min(),
                       imp_trace[:,1].max() - imp_trace[:,1].min()])
    
    scale_ratio = imp_span / opt_span if opt_span > 0 else 0
    print(f"\nScale ratio (Improved/Optimized): {scale_ratio:.2f}x")
    
    # Check if keyboard layouts match
    print("\nChecking keyboard alignment...")
    
    # Get key positions for 'h' and 'o' 
    import pandas as pd
    kb = pd.read_csv('holokeyboard.csv', index_col='keys')
    
    h_pos = [kb.at['h', 'x_pos'], kb.at['h', 'y_pos']]
    o_pos = [kb.at['o', 'x_pos'], kb.at['o', 'y_pos']]
    
    # Check start/end positions
    opt_start_dist = np.linalg.norm(opt_trace[0,:2] - h_pos)
    imp_start_dist = np.linalg.norm(imp_trace[0,:2] - h_pos)
    
    print(f"Distance from 'h' key:")
    print(f"  Optimized: {opt_start_dist:.1f}")
    print(f"  Improved: {imp_start_dist:.1f}")
    
    if imp_start_dist > 100:
        print("⚠️  WARNING: Improved generator may be using different keyboard layout!")
    
    return opt_trace, imp_trace


def test_with_normalization():
    """
    Test recognition with normalized traces
    """
    print("\n" + "="*60)
    print("NORMALIZED RECOGNITION TEST")
    print("="*60)
    
    def normalize_trace(trace):
        """Normalize trace to [0, 1] range"""
        t = trace[:, :2].copy()
        t_min = t.min(axis=0)
        t_max = t.max(axis=0)
        t_range = t_max - t_min
        t_range[t_range == 0] = 1
        return (t - t_min) / t_range
    
    test_words = ['hello', 'world', 'test']
    
    opt_gen = OptimizedSwypeGenerator()
    imp_gen = ImprovedSwypeGenerator()
    
    # Build normalized templates
    templates = {}
    for word in test_words:
        result = opt_gen.generate_trajectory(word)
        templates[word] = normalize_trace(result['trajectory'])
    
    print("\nTesting with normalized traces...")
    correct = 0
    total = 0
    
    for word in test_words:
        result = imp_gen.generate_trajectory(word)
        test_trace = normalize_trace(result['trajectory'])
        
        # Downsample for DTW
        if len(test_trace) > 50:
            indices = np.linspace(0, len(test_trace)-1, 50, dtype=int)
            test_trace = test_trace[indices]
        
        # Find closest template
        best_word = None
        best_dist = np.inf
        
        for template_word, template in templates.items():
            # Downsample template
            if len(template) > 50:
                indices = np.linspace(0, len(template)-1, 50, dtype=int)
                temp = template[indices]
            else:
                temp = template
            
            # Simple DTW
            n, m = len(test_trace), len(temp)
            dtw = np.full((n+1, m+1), np.inf)
            dtw[0, 0] = 0
            
            for i in range(1, n+1):
                for j in range(1, m+1):
                    cost = np.linalg.norm(test_trace[i-1] - temp[j-1])
                    dtw[i, j] = cost + min(dtw[i-1, j], dtw[i, j-1], dtw[i-1, j-1])
            
            dist = dtw[n, m]
            if dist < best_dist:
                best_dist = dist
                best_word = template_word
        
        is_correct = best_word == word
        if is_correct:
            correct += 1
        total += 1
        
        status = "✅" if is_correct else "❌"
        print(f"{status} Word: '{word}' → Decoded: '{best_word}' (dist: {best_dist:.2f})")
    
    accuracy = correct / total if total > 0 else 0
    print(f"\nNormalized Recognition Accuracy: {accuracy*100:.1f}%")
    
    return accuracy


def main():
    # Analyze differences
    opt_trace, imp_trace = analyze_trace_differences()
    
    # Test with normalization
    accuracy = test_with_normalization()
    
    if accuracy > 0.8:
        print("\n✅ With normalization, cross-generator recognition works!")
    else:
        print("\n❌ Generators have fundamental differences")


if __name__ == "__main__":
    main()