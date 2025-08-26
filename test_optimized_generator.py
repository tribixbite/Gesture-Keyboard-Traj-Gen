#!/usr/bin/env python
"""
Test optimized generator and compare with previous versions
"""

import numpy as np
import time
from optimized_swype_generator import OptimizedSwypeGenerator, StyleParameters
from improved_swype_generator import ImprovedSwypeGenerator

def test_performance():
    """Test generation speed improvements"""
    print("="*60)
    print("PERFORMANCE TEST")
    print("="*60)
    
    test_words = ['hello', 'world', 'computer', 'keyboard', 'gesture'] * 10
    
    # Test improved generator
    improved_gen = ImprovedSwypeGenerator()
    start = time.time()
    for word in test_words:
        _ = improved_gen.generate_trajectory(word, style='natural')
    improved_time = time.time() - start
    
    # Test optimized generator
    optimized_gen = OptimizedSwypeGenerator()
    start = time.time()
    for word in test_words:
        _ = optimized_gen.generate_trajectory(word, style='natural')
    optimized_time = time.time() - start
    
    print(f"\nGeneration Time (50 traces):")
    print(f"  Improved Generator: {improved_time:.2f}s ({len(test_words)/improved_time:.1f} traces/sec)")
    print(f"  Optimized Generator: {optimized_time:.2f}s ({len(test_words)/optimized_time:.1f} traces/sec)")
    print(f"  Speedup: {improved_time/optimized_time:.2f}x")
    
    return optimized_time < improved_time

def test_quality():
    """Test trajectory quality metrics"""
    print("\n" + "="*60)
    print("QUALITY TEST")
    print("="*60)
    
    optimized_gen = OptimizedSwypeGenerator()
    test_cases = [
        ('hello', 'natural'),
        ('computer', 'precise'),
        ('quick', 'fast'),
        ('gesture', 'sloppy')
    ]
    
    results = []
    for word, style in test_cases:
        trace = optimized_gen.generate_trajectory(word, style=style)
        
        print(f"\n{word} ({style}):")
        print(f"  Mean jerk: {trace['metadata']['mean_jerk']:,.0f} units/s³")
        print(f"  Max jerk: {trace['metadata']['max_jerk']:,.0f} units/s³")
        print(f"  Mean velocity: {trace['metadata']['mean_velocity']:.1f} units/s")
        
        # Check if meets target
        if trace['metadata']['mean_jerk'] < 1000000:
            print(f"  ✅ Meets <1M jerk target")
        else:
            print(f"  ⚠️  Above 1M jerk target")
        
        results.append(trace['metadata']['mean_jerk'])
    
    avg_jerk = np.mean(results)
    print(f"\nOverall average jerk: {avg_jerk:,.0f} units/s³")
    
    return avg_jerk < 1000000

def test_continuous_styles():
    """Test continuous style parameters"""
    print("\n" + "="*60)
    print("CONTINUOUS STYLE TEST")  
    print("="*60)
    
    optimized_gen = OptimizedSwypeGenerator()
    
    # Test with custom style parameters
    custom_styles = [
        StyleParameters(smoothness=0.9, speed=0.5, precision=1.0, fluidity=0.9, pressure_variation=0.1),
        StyleParameters(smoothness=0.3, speed=2.0, precision=0.3, fluidity=0.2, pressure_variation=0.8),
        StyleParameters(smoothness=0.6, speed=1.2, precision=0.7, fluidity=0.5, pressure_variation=0.4)
    ]
    
    style_names = ["Ultra-precise", "Chaotic-fast", "Balanced-custom"]
    
    for style, name in zip(custom_styles, style_names):
        trace = optimized_gen.generate_trajectory('hello', style=style)
        
        print(f"\n{name}:")
        print(f"  Smoothness: {style.smoothness:.1f}")
        print(f"  Speed: {style.speed:.1f}")
        print(f"  Precision: {style.precision:.1f}")
        print(f"  Fluidity: {style.fluidity:.1f}")
        print(f"  Pressure variation: {style.pressure_variation:.1f}")
        print(f"  → Mean jerk: {trace['metadata']['mean_jerk']:,.0f} units/s³")
        print(f"  → Duration: {trace['duration']:.2f}s")
    
    return True

def test_pressure_feature():
    """Test new pressure variation feature"""
    print("\n" + "="*60)
    print("PRESSURE VARIATION TEST")
    print("="*60)
    
    optimized_gen = OptimizedSwypeGenerator()
    
    # Generate with pressure
    trace = optimized_gen.generate_trajectory('pressure', style='natural')
    
    if 'pressure' in trace and len(trace['pressure']) > 0:
        pressure = trace['pressure']
        print(f"\nPressure profile for 'pressure':")
        print(f"  Min pressure: {np.min(pressure):.2f}")
        print(f"  Max pressure: {np.max(pressure):.2f}")
        print(f"  Mean pressure: {np.mean(pressure):.2f}")
        print(f"  Std pressure: {np.std(pressure):.2f}")
        
        # Check pressure variation at start/end
        print(f"  Start pressure: {pressure[0]:.2f}")
        print(f"  End pressure: {pressure[-1]:.2f}")
        
        if pressure[0] < 0.7 and pressure[-1] < 0.7:
            print(f"  ✅ Realistic pressure fade at boundaries")
        
        return True
    else:
        print("  ❌ No pressure data generated")
        return False

def test_curvature_velocity():
    """Test curvature-dependent velocity modulation"""
    print("\n" + "="*60)
    print("CURVATURE-VELOCITY TEST")
    print("="*60)
    
    optimized_gen = OptimizedSwypeGenerator()
    
    # Word with corners vs straight
    corner_word = 'zigzag'  # Many direction changes
    straight_word = 'ill'    # Mostly vertical
    
    corner_trace = optimized_gen.generate_trajectory(corner_word, style='natural')
    straight_trace = optimized_gen.generate_trajectory(straight_word, style='natural')
    
    print(f"\nVelocity comparison:")
    print(f"  '{corner_word}' (many corners):")
    print(f"    Mean velocity: {corner_trace['metadata']['mean_velocity']:.1f} units/s")
    print(f"  '{straight_word}' (mostly straight):")
    print(f"    Mean velocity: {straight_trace['metadata']['mean_velocity']:.1f} units/s")
    
    # Expect lower velocity for corner-heavy trajectory
    if corner_trace['metadata']['mean_velocity'] < straight_trace['metadata']['mean_velocity']:
        print(f"  ✅ Corner trajectory has lower average velocity (realistic)")
        return True
    else:
        print(f"  ⚠️  Unexpected velocity relationship")
        return False

def main():
    """Run all tests"""
    print("="*60)
    print("OPTIMIZED GENERATOR TEST SUITE")
    print("="*60)
    
    tests = [
        ("Performance", test_performance),
        ("Quality", test_quality),
        ("Continuous Styles", test_continuous_styles),
        ("Pressure Feature", test_pressure_feature),
        ("Curvature Velocity", test_curvature_velocity)
    ]
    
    results = []
    for name, test_func in tests:
        try:
            passed = test_func()
            results.append((name, passed))
        except Exception as e:
            print(f"\n❌ {name} test failed with error: {e}")
            results.append((name, False))
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    for name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"  {name}: {status}")
    
    all_passed = all(p for _, p in results)
    if all_passed:
        print("\n🎉 All tests passed! Optimized generator is ready.")
    else:
        print("\n⚠️  Some tests failed. Review issues above.")
    
    return all_passed

if __name__ == "__main__":
    main()