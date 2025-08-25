#!/usr/bin/env python
# coding=utf-8
"""
Compare different swype trajectory generation approaches
"""

import numpy as np
import pandas as pd
import json
from enhanced_swype_generator import EnhancedSwypeGenerator

def compare_generation_methods():
    """Compare different trajectory generation methods and parameters"""
    
    # Create enhanced generator
    generator = EnhancedSwypeGenerator()
    
    # Test configurations
    test_configs = [
        {
            'name': 'precise_slow',
            'word': 'hello',
            'user_speed': 0.7,
            'precision': 0.95,
            'description': 'Precise and slow typing'
        },
        {
            'name': 'fast_sloppy',
            'word': 'hello',
            'user_speed': 1.5,
            'precision': 0.5,
            'description': 'Fast but sloppy typing'
        },
        {
            'name': 'normal',
            'word': 'hello',
            'user_speed': 1.0,
            'precision': 0.8,
            'description': 'Normal typing'
        }
    ]
    
    # Different velocity profiles
    velocity_profiles = ['bell', 'uniform', 'realistic']
    
    results = {}
    
    print("Trajectory Generation Comparison")
    print("=" * 70)
    
    for config in test_configs:
        print(f"\nConfiguration: {config['name']}")
        print(f"Description: {config['description']}")
        print("-" * 40)
        
        config_results = {}
        
        for profile in velocity_profiles:
            # Set velocity profile
            generator.velocity_profile = profile
            
            # Generate trajectory
            traj_data = generator.generate_trajectory(
                word=config['word'],
                user_speed=config['user_speed'],
                precision=config['precision']
            )
            
            # Calculate metrics
            trajectory = traj_data['trajectory']
            velocity = traj_data['velocity']
            acceleration = traj_data['acceleration']
            jerk = traj_data['jerk']
            
            # Path length
            path_length = np.sum(np.sqrt(
                np.diff(trajectory[:, 0])**2 + 
                np.diff(trajectory[:, 1])**2
            ))
            
            # Velocity statistics
            vel_mag = np.sqrt(velocity[:, 0]**2 + velocity[:, 1]**2)
            avg_velocity = np.mean(vel_mag)
            max_velocity = np.max(vel_mag)
            
            # Jerk statistics (smoothness indicator)
            jerk_mag = np.sqrt(jerk[:, 0]**2 + jerk[:, 1]**2)
            avg_jerk = np.mean(jerk_mag)
            max_jerk = np.max(jerk_mag)
            
            # Store results
            profile_results = {
                'path_length': path_length,
                'avg_velocity': avg_velocity,
                'max_velocity': max_velocity,
                'avg_jerk': avg_jerk,
                'max_jerk': max_jerk,
                'duration': traj_data['duration'],
                'n_points': len(trajectory)
            }
            
            config_results[profile] = profile_results
            
            print(f"\n  Velocity Profile: {profile}")
            print(f"    Path length: {path_length:.2f} units")
            print(f"    Duration: {traj_data['duration']:.3f} seconds")
            print(f"    Avg velocity: {avg_velocity:.2f} units/s")
            print(f"    Max velocity: {max_velocity:.2f} units/s")
            print(f"    Avg jerk: {avg_jerk:.0f} units/s³ (lower is smoother)")
            
        results[config['name']] = config_results
    
    # Compare smoothness across configurations
    print("\n" + "=" * 70)
    print("Smoothness Comparison (Average Jerk - Lower is Better)")
    print("-" * 70)
    
    for config_name, config_results in results.items():
        print(f"\n{config_name}:")
        for profile, metrics in config_results.items():
            print(f"  {profile:12s}: {metrics['avg_jerk']:10.0f} units/s³")
    
    # Save comparison results
    with open('trajectory_comparison.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("\nResults saved to trajectory_comparison.json")
    
    return results

def test_word_variations():
    """Test trajectory generation for different words"""
    
    generator = EnhancedSwypeGenerator()
    generator.velocity_profile = 'realistic'
    
    test_words = ['hello', 'world', 'quick', 'brown', 'jumps', 'that', 'test']
    
    print("\n" + "=" * 70)
    print("Word Variation Test")
    print("=" * 70)
    
    word_stats = []
    
    for word in test_words:
        traj_data = generator.generate_trajectory(
            word=word,
            user_speed=1.0,
            precision=0.8
        )
        
        trajectory = traj_data['trajectory']
        
        # Calculate path complexity
        path_length = np.sum(np.sqrt(
            np.diff(trajectory[:, 0])**2 + 
            np.diff(trajectory[:, 1])**2
        ))
        
        # Calculate curvature (as a measure of complexity)
        dx = np.diff(trajectory[:, 0])
        dy = np.diff(trajectory[:, 1])
        ddx = np.diff(dx)
        ddy = np.diff(dy)
        
        # Approximate curvature
        curvature = np.abs(ddx * dy[:-1] - ddy * dx[:-1]) / (dx[:-1]**2 + dy[:-1]**2)**1.5
        avg_curvature = np.mean(curvature[np.isfinite(curvature)])
        
        word_stat = {
            'word': word,
            'length': len(word),
            'path_length': path_length,
            'path_per_char': path_length / len(word),
            'duration': traj_data['duration'],
            'avg_curvature': avg_curvature
        }
        
        word_stats.append(word_stat)
        
        print(f"\nWord: '{word}'")
        print(f"  Characters: {len(word)}")
        print(f"  Path length: {path_length:.2f} units")
        print(f"  Path/char: {path_length/len(word):.2f} units")
        print(f"  Duration: {traj_data['duration']:.3f} seconds")
        print(f"  Complexity (curvature): {avg_curvature:.4f}")
        
        # Save individual trajectory
        json_data = generator.export_trajectory(traj_data, format='json')
        with open(f'word_traj_{word}.json', 'w') as f:
            f.write(json_data)
    
    # Save word statistics
    df = pd.DataFrame(word_stats)
    df.to_csv('word_statistics.csv', index=False)
    print("\nWord statistics saved to word_statistics.csv")
    
    return word_stats

if __name__ == '__main__':
    # Run comparison tests
    comparison_results = compare_generation_methods()
    
    # Test word variations
    word_results = test_word_variations()
    
    print("\n" + "=" * 70)
    print("Testing Complete!")
    print("Generated files:")
    print("  - trajectory_comparison.json: Method comparison results")
    print("  - word_statistics.csv: Statistics for different words")
    print("  - word_traj_*.json: Individual word trajectories")