#!/usr/bin/env python
"""
Quick recognition test with optimized DTW
"""

import numpy as np
import time
from typing import Dict, List, Tuple
from optimized_swype_generator import OptimizedSwypeGenerator

def fast_dtw_distance(trace1: np.ndarray, trace2: np.ndarray, max_dist: float = np.inf) -> float:
    """
    Fast DTW with early termination
    """
    # Downsample for speed
    if len(trace1) > 50:
        indices1 = np.linspace(0, len(trace1)-1, 50, dtype=int)
        trace1 = trace1[indices1]
    if len(trace2) > 50:
        indices2 = np.linspace(0, len(trace2)-1, 50, dtype=int)
        trace2 = trace2[indices2]
    
    n, m = len(trace1), len(trace2)
    
    # Use only x,y coordinates
    t1 = trace1[:, :2] if trace1.shape[1] > 2 else trace1
    t2 = trace2[:, :2] if trace2.shape[1] > 2 else trace2
    
    # DTW with pruning
    dtw = np.full((n + 1, m + 1), np.inf)
    dtw[0, 0] = 0
    
    for i in range(1, n + 1):
        min_cost = np.inf
        for j in range(1, m + 1):
            cost = np.linalg.norm(t1[i-1] - t2[j-1])
            dtw[i, j] = cost + min(
                dtw[i-1, j],
                dtw[i, j-1], 
                dtw[i-1, j-1]
            )
            min_cost = min(min_cost, dtw[i, j])
        
        # Early termination if all costs exceed threshold
        if min_cost > max_dist:
            return np.inf
    
    return dtw[n, m]


def quick_recognition_test():
    """
    Quick test of recognition accuracy
    """
    print("="*60)
    print("QUICK RECOGNITION TEST")
    print("="*60)
    
    # Small vocabulary for quick test
    test_words = ['the', 'and', 'you', 'was', 'hello']
    
    generator = OptimizedSwypeGenerator()
    
    # Build templates
    print("\nBuilding templates...")
    templates = {}
    for word in test_words:
        result = generator.generate_trajectory(word, style='precise')
        templates[word] = result['trajectory']
    
    # Test recognition
    print("\nTesting recognition...")
    correct = 0
    total = 0
    
    for word in test_words:
        # Generate test traces with different styles
        for style in ['natural', 'fast']:
            result = generator.generate_trajectory(word, style=style)
            test_trace = result['trajectory']
            
            # Find closest template
            best_word = None
            best_dist = np.inf
            
            for template_word, template in templates.items():
                dist = fast_dtw_distance(test_trace, template)
                if dist < best_dist:
                    best_dist = dist
                    best_word = template_word
            
            # Check if correct
            is_correct = best_word == word
            if is_correct:
                correct += 1
            total += 1
            
            status = "✅" if is_correct else "❌"
            print(f"{status} Word: '{word}' ({style}) → Decoded: '{best_word}' (dist: {best_dist:.1f})")
    
    accuracy = correct / total if total > 0 else 0
    
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print(f"Recognition Accuracy: {accuracy*100:.1f}% ({correct}/{total})")
    
    if accuracy > 0.85:
        print("✅ EXCELLENT: Traces are highly recognizable!")
    elif accuracy > 0.70:
        print("✅ GOOD: Traces are recognizable")
    elif accuracy > 0.50:
        print("⚠️  FAIR: Traces need improvement")
    else:
        print("❌ POOR: Recognition needs work")
    
    return accuracy


def test_cross_generator_recognition():
    """
    Test if traces from one generator are recognizable by another
    """
    print("\n" + "="*60)
    print("CROSS-GENERATOR RECOGNITION TEST")
    print("="*60)
    
    from improved_swype_generator import ImprovedSwypeGenerator
    
    test_words = ['hello', 'world', 'test']
    
    # Generate templates with Optimized generator
    opt_gen = OptimizedSwypeGenerator()
    templates = {}
    
    print("\nBuilding templates with OptimizedSwypeGenerator...")
    for word in test_words:
        result = opt_gen.generate_trajectory(word, style='precise')
        templates[word] = result['trajectory']
    
    # Test with traces from Improved generator
    imp_gen = ImprovedSwypeGenerator()
    
    print("\nTesting recognition of ImprovedSwypeGenerator traces...")
    correct = 0
    total = 0
    
    for word in test_words:
        result = imp_gen.generate_trajectory(word)
        test_trace = result['trajectory']
        
        # Find closest template
        best_word = None
        best_dist = np.inf
        
        for template_word, template in templates.items():
            dist = fast_dtw_distance(test_trace, template, max_dist=1000)
            if dist < best_dist:
                best_dist = dist
                best_word = template_word
        
        is_correct = best_word == word
        if is_correct:
            correct += 1
        total += 1
        
        status = "✅" if is_correct else "❌"
        print(f"{status} Word: '{word}' → Decoded: '{best_word}' (dist: {best_dist:.1f})")
    
    accuracy = correct / total if total > 0 else 0
    print(f"\nCross-Generator Accuracy: {accuracy*100:.1f}%")
    
    return accuracy


def main():
    """
    Run all quick tests
    """
    # Quick recognition test
    acc1 = quick_recognition_test()
    
    # Cross-generator test
    acc2 = test_cross_generator_recognition()
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Same-Generator Recognition: {acc1*100:.1f}%")
    print(f"Cross-Generator Recognition: {acc2*100:.1f}%")
    
    overall = (acc1 + acc2) / 2
    if overall > 0.75:
        print(f"\n✅ Overall: {overall*100:.1f}% - Traces are recognizable!")
        print("Ready for production use.")
    else:
        print(f"\n⚠️  Overall: {overall*100:.1f}% - Needs improvement")
    
    return overall


if __name__ == "__main__":
    start = time.time()
    accuracy = main()
    elapsed = time.time() - start
    print(f"\nTest completed in {elapsed:.1f} seconds")