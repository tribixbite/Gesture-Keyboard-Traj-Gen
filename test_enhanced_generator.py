#!/usr/bin/env python
# coding=utf-8
"""
Test the enhanced swype generator without visualization
"""

import numpy as np
import pandas as pd
import json
from enhanced_swype_generator import EnhancedSwypeGenerator

def test_generator():
    """Test the enhanced generator functionality"""
    
    # Create generator
    generator = EnhancedSwypeGenerator()
    
    # Test words
    test_words = ['hello', 'that', 'world', 'test']
    
    for word in test_words:
        print(f"\n{'='*50}")
        print(f"Generating trajectory for '{word}'")
        print('='*50)
        
        # Generate trajectory
        trajectory_data = generator.generate_trajectory(
            word=word,
            user_speed=1.0,
            precision=0.8
        )
        
        # Print statistics
        trajectory = trajectory_data['trajectory']
        velocity = trajectory_data['velocity']
        acceleration = trajectory_data['acceleration']
        jerk = trajectory_data['jerk']
        
        print(f"Word: {trajectory_data['word']}")
        print(f"Duration: {trajectory_data['duration']:.2f} seconds")
        print(f"Number of points: {len(trajectory)}")
        print(f"Sampling rate: {trajectory_data['sampling_rate']} Hz")
        
        # Calculate path length
        path_length = np.sum(np.sqrt(np.diff(trajectory[:, 0])**2 + 
                                     np.diff(trajectory[:, 1])**2))
        print(f"Path length: {path_length:.2f} units")
        
        # Velocity statistics
        velocity_mag = np.sqrt(velocity[:, 0]**2 + velocity[:, 1]**2)
        print(f"Average velocity: {np.mean(velocity_mag):.2f} units/s")
        print(f"Max velocity: {np.max(velocity_mag):.2f} units/s")
        
        # Acceleration statistics
        accel_mag = np.sqrt(acceleration[:, 0]**2 + acceleration[:, 1]**2)
        print(f"Average acceleration: {np.mean(accel_mag):.2f} units/s²")
        print(f"Max acceleration: {np.max(accel_mag):.2f} units/s²")
        
        # Jerk statistics
        jerk_mag = np.sqrt(jerk[:, 0]**2 + jerk[:, 1]**2)
        print(f"Average jerk: {np.mean(jerk_mag):.2f} units/s³")
        print(f"Max jerk: {np.max(jerk_mag):.2f} units/s³")
        
        # Save to JSON
        json_data = generator.export_trajectory(trajectory_data, format='json')
        filename = f'trajectory_{word}.json'
        with open(filename, 'w') as f:
            f.write(json_data)
        print(f"\nTrajectory saved to {filename}")

if __name__ == '__main__':
    test_generator()