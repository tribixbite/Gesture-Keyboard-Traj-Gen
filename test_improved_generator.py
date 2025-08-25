#!/usr/bin/env python
"""
Test the improved generator to verify jerk reduction
"""

import numpy as np
from improved_swype_generator import ImprovedSwypeGenerator
from enhanced_swype_generator import EnhancedSwypeGenerator

def test_generators():
    """Compare original and improved generators"""
    
    # Test words
    test_words = ['hello', 'world', 'computer', 'the', 'quick']
    styles = ['precise', 'natural', 'fast', 'sloppy']
    
    print("="*60)
    print("GENERATOR COMPARISON TEST")
    print("="*60)
    
    # Initialize generators
    print("\nInitializing generators...")
    enhanced_gen = EnhancedSwypeGenerator()
    improved_gen = ImprovedSwypeGenerator()
    
    # Results storage
    enhanced_results = []
    improved_results = []
    
    print("\nTesting trajectory generation...")
    print("-"*60)
    
    for word in test_words:
        for style in styles:
            # Test enhanced generator
            enhanced_gen.velocity_profile = 'bell'  # Use best profile from before
            enhanced_trace = enhanced_gen.generate_trajectory(
                word=word,
                user_speed=1.0,
                precision=0.8
            )
            
            # Calculate enhanced jerk
            if 'jerk' in enhanced_trace:
                enhanced_jerk = np.mean(np.abs(enhanced_trace['jerk']))
                enhanced_max_jerk = np.max(np.abs(enhanced_trace['jerk']))
            else:
                enhanced_jerk = 0
                enhanced_max_jerk = 0
            
            # Test improved generator
            improved_trace = improved_gen.generate_trajectory(
                word=word,
                user_speed=1.0,
                precision=0.8,
                style=style
            )
            
            # Get improved jerk from metadata
            improved_jerk = improved_trace['metadata']['mean_jerk']
            improved_max_jerk = improved_trace['metadata']['max_jerk']
            
            # Store results
            enhanced_results.append({
                'word': word,
                'style': style,
                'mean_jerk': enhanced_jerk,
                'max_jerk': enhanced_max_jerk
            })
            
            improved_results.append({
                'word': word,
                'style': style,
                'mean_jerk': improved_jerk,
                'max_jerk': improved_max_jerk
            })
            
            # Print comparison
            print(f"\nWord: '{word}' | Style: {style}")
            print(f"  Enhanced - Mean Jerk: {enhanced_jerk:,.0f} | Max: {enhanced_max_jerk:,.0f}")
            print(f"  Improved - Mean Jerk: {improved_jerk:,.0f} | Max: {improved_max_jerk:,.0f}")
            
            if enhanced_jerk > 0:
                reduction = (1 - improved_jerk / enhanced_jerk) * 100
                print(f"  Reduction: {reduction:.1f}%")
                
                if improved_jerk < 1000000:
                    print(f"  ✅ Achieved target (<1M units/s³)")
                else:
                    print(f"  ⚠️  Above target (>1M units/s³)")
    
    # Calculate overall statistics
    print("\n" + "="*60)
    print("OVERALL STATISTICS")
    print("="*60)
    
    # Enhanced stats
    enhanced_mean = np.mean([r['mean_jerk'] for r in enhanced_results if r['mean_jerk'] > 0])
    enhanced_max = np.max([r['max_jerk'] for r in enhanced_results if r['max_jerk'] > 0])
    
    # Improved stats
    improved_mean = np.mean([r['mean_jerk'] for r in improved_results])
    improved_max = np.max([r['max_jerk'] for r in improved_results])
    
    print(f"\nEnhanced Generator:")
    print(f"  Average Mean Jerk: {enhanced_mean:,.0f} units/s³")
    print(f"  Maximum Jerk: {enhanced_max:,.0f} units/s³")
    
    print(f"\nImproved Generator:")
    print(f"  Average Mean Jerk: {improved_mean:,.0f} units/s³")
    print(f"  Maximum Jerk: {improved_max:,.0f} units/s³")
    
    if enhanced_mean > 0:
        overall_reduction = (1 - improved_mean / enhanced_mean) * 100
        print(f"\nOverall Jerk Reduction: {overall_reduction:.1f}%")
    
    # Style-specific analysis
    print("\n" + "="*60)
    print("STYLE-SPECIFIC PERFORMANCE")
    print("="*60)
    
    for style in styles:
        style_results = [r for r in improved_results if r['style'] == style]
        style_mean = np.mean([r['mean_jerk'] for r in style_results])
        style_max = np.max([r['max_jerk'] for r in style_results])
        
        print(f"\n{style.capitalize()} Style:")
        print(f"  Average Mean Jerk: {style_mean:,.0f} units/s³")
        print(f"  Maximum Jerk: {style_max:,.0f} units/s³")
        
        if style_mean < 1000000:
            print(f"  ✅ Meets target for human-like motion")
        else:
            print(f"  ⚠️  Needs further optimization")
    
    # Check trajectory properties
    print("\n" + "="*60)
    print("TRAJECTORY PROPERTIES CHECK")
    print("="*60)
    
    # Test one example in detail
    test_trace = improved_gen.generate_trajectory(
        word='hello',
        style='natural',
        user_speed=1.0,
        precision=0.8
    )
    
    print(f"\nExample: 'hello' with natural style")
    print(f"  Duration: {test_trace['duration']:.2f} seconds")
    print(f"  Sampling rate: {test_trace['sampling_rate']} Hz")
    print(f"  Number of points: {len(test_trace['points'])}")
    print(f"  Key positions: {len(test_trace['key_positions'])}")
    
    # Velocity check
    velocity = test_trace['velocity']
    vel_magnitude = np.sqrt(velocity[:, 0]**2 + velocity[:, 1]**2)
    print(f"  Mean velocity: {np.mean(vel_magnitude):.1f} units/s")
    print(f"  Max velocity: {np.max(vel_magnitude):.1f} units/s")
    
    # Acceleration check
    acceleration = test_trace['acceleration']
    acc_magnitude = np.sqrt(acceleration[:, 0]**2 + acceleration[:, 1]**2)
    print(f"  Mean acceleration: {np.mean(acc_magnitude):.1f} units/s²")
    print(f"  Max acceleration: {np.max(acc_magnitude):.1f} units/s²")
    
    # Path length
    trajectory = test_trace['trajectory']
    path_length = 0
    for i in range(1, len(trajectory)):
        dx = trajectory[i, 0] - trajectory[i-1, 0]
        dy = trajectory[i, 1] - trajectory[i-1, 1]
        path_length += np.sqrt(dx**2 + dy**2)
    
    print(f"  Total path length: {path_length:.1f} units")
    print(f"  Path per character: {path_length/len('hello'):.1f} units/char")
    
    print("\n" + "="*60)
    print("TEST COMPLETE")
    print("="*60)
    
    # Summary
    if improved_mean < 1000000:
        print("\n🎉 SUCCESS: Improved generator achieves human-like jerk values!")
        print(f"   Target: <1,000,000 units/s³")
        print(f"   Achieved: {improved_mean:,.0f} units/s³")
    else:
        print("\n⚠️  PARTIAL SUCCESS: Significant improvement but needs more tuning")
        print(f"   Target: <1,000,000 units/s³")
        print(f"   Current: {improved_mean:,.0f} units/s³")
    
    return improved_results

if __name__ == "__main__":
    results = test_generators()