#!/usr/bin/env python
# coding=utf-8
"""
Hybrid Swype Trajectory Generator
Combines Enhanced Generator with Jerk-Minimization optimization
"""

import numpy as np
import pandas as pd
import sys
import os
sys.path.append('./Jerk-Minimization')
from traj_gen import optim_trajectory as opt_t
from enhanced_swype_generator import EnhancedSwypeGenerator
from typing import Dict, List, Optional
import json

class HybridSwypeGenerator:
    """
    Hybrid generator combining multiple trajectory generation techniques
    """
    
    def __init__(self, keyboard_layout_path: str = 'holokeyboard.csv'):
        """
        Initialize the hybrid generator
        
        Args:
            keyboard_layout_path: Path to CSV file containing keyboard layout
        """
        self.keyboard_df = pd.read_csv(keyboard_layout_path, index_col='keys')
        self.enhanced_gen = EnhancedSwypeGenerator(keyboard_layout_path)
        
    def generate_optimized_trajectory(self,
                                     word: str,
                                     optimization_weights: np.ndarray = None,
                                     duration: float = None,
                                     pnt_density: int = 10,
                                     use_enhanced_init: bool = True) -> Dict:
        """
        Generate trajectory using jerk-minimization with optional enhanced initialization
        
        Args:
            word: Word to generate trajectory for
            optimization_weights: Weights for derivatives [position, vel, accel, jerk]
            duration: Total duration in seconds
            pnt_density: Points per second for optimization
            use_enhanced_init: Use enhanced generator for initial guess
            
        Returns:
            Dictionary containing trajectory data
        """
        word = word.lower()
        
        # Default weights if not provided (minimize jerk)
        if optimization_weights is None:
            optimization_weights = np.array([0, 0, 0, 1])
            
        # Auto-calculate duration if not provided
        if duration is None:
            duration = len(word) * 0.15 * 1.5  # Slightly longer for optimization
            
        # Get key positions for the word
        key_positions = self._get_key_positions(word)
        n_keys = len(key_positions)
        
        # Setup optimization problem
        dim = 2
        knots = np.array([0.0, duration])
        opt_traj = opt_t.OptimTrajGen(knots, dim, pnt_density)
        
        # Create time points for keys
        ts = np.linspace(0, duration, n_keys)
        
        # Add position constraints at key times
        for i, (t, pos) in enumerate(zip(ts, key_positions)):
            pin_ = {'t': t, 'd': 0, 'X': pos}
            opt_traj.addPin(pin_)
        
        # Add velocity constraints at start and end
        v_init = np.array([0, 0])
        v_final = np.array([0, 0])
        
        if use_enhanced_init:
            # Use enhanced generator to get better initial velocities
            enhanced_traj = self.enhanced_gen.generate_trajectory(
                word=word,
                duration=duration,
                user_speed=1.0,
                precision=0.9
            )
            velocities = enhanced_traj['velocity']
            v_init = velocities[0]
            v_final = velocities[-1]
        
        # Add velocity pins
        pin_v_init = {'t': ts[0], 'd': 1, 'X': v_init}
        pin_v_final = {'t': ts[-1], 'd': 1, 'X': v_final}
        opt_traj.addPin(pin_v_init)
        opt_traj.addPin(pin_v_final)
        
        # Add acceleration constraint at end
        a_final = np.array([0, 0])
        pin_a = {'t': ts[-1], 'd': 2, 'X': a_final}
        opt_traj.addPin(pin_a)
        
        # Solve optimization
        opt_traj.setDerivativeObj(optimization_weights)
        opt_traj.solve()
        
        # Extract trajectory if solved
        if opt_traj.isSolved:
            # Sample the optimized trajectory
            n_samples = int(duration * 100)  # 100 Hz sampling
            t_sample = np.linspace(0, duration, n_samples)
            
            # Evaluate positions
            positions = opt_traj.eval(t_sample, 0).T
            
            # Evaluate velocities
            velocities = opt_traj.eval(t_sample, 1).T
            
            # Evaluate accelerations
            accelerations = opt_traj.eval(t_sample, 2).T
            
            # Evaluate jerks
            jerks = opt_traj.eval(t_sample[:-1], 3).T
            # Pad jerk to match other arrays
            jerks = np.vstack([jerks, jerks[-1]])
            
            # Create trajectory array
            trajectory = np.column_stack([positions, t_sample])
            
            return {
                'word': word,
                'trajectory': trajectory,
                'velocity': velocities,
                'acceleration': accelerations,
                'jerk': jerks,
                'duration': duration,
                'sampling_rate': 100,
                'key_positions': key_positions,
                'optimization_weights': optimization_weights,
                'method': 'jerk_minimization'
            }
        else:
            print(f"Optimization failed for '{word}'. Using enhanced generator fallback.")
            return self.enhanced_gen.generate_trajectory(word=word, duration=duration)
    
    def generate_hybrid_trajectory(self,
                                 word: str,
                                 blend_factor: float = 0.5,
                                 duration: float = None,
                                 user_speed: float = 1.0,
                                 precision: float = 0.8) -> Dict:
        """
        Generate trajectory by blending enhanced and optimized approaches
        
        Args:
            word: Word to generate trajectory for
            blend_factor: Weight for blending (0=enhanced only, 1=optimized only)
            duration: Total duration in seconds
            user_speed: Speed multiplier
            precision: User precision level
            
        Returns:
            Dictionary containing trajectory data
        """
        word = word.lower()
        
        # Auto-calculate duration if not provided
        if duration is None:
            base_time = len(word) * 0.15
            duration = base_time * (1.2 - 0.2 * user_speed)
        
        # Generate enhanced trajectory
        enhanced_traj = self.enhanced_gen.generate_trajectory(
            word=word,
            duration=duration,
            user_speed=user_speed,
            precision=precision
        )
        
        # Generate optimized trajectory
        opt_weights = np.array([0, 0, 0.5, 0.5])  # Balance acceleration and jerk
        optimized_traj = self.generate_optimized_trajectory(
            word=word,
            optimization_weights=opt_weights,
            duration=duration,
            use_enhanced_init=True
        )
        
        # Blend trajectories
        if optimized_traj['method'] == 'jerk_minimization':
            # Successful optimization - blend results
            blended_trajectory = (1 - blend_factor) * enhanced_traj['trajectory'] + \
                               blend_factor * optimized_traj['trajectory']
            
            blended_velocity = (1 - blend_factor) * enhanced_traj['velocity'] + \
                             blend_factor * optimized_traj['velocity']
            
            blended_acceleration = (1 - blend_factor) * enhanced_traj['acceleration'] + \
                                 blend_factor * optimized_traj['acceleration']
            
            blended_jerk = (1 - blend_factor) * enhanced_traj['jerk'] + \
                         blend_factor * optimized_traj['jerk']
            
            return {
                'word': word,
                'trajectory': blended_trajectory,
                'velocity': blended_velocity,
                'acceleration': blended_acceleration,
                'jerk': blended_jerk,
                'duration': duration,
                'sampling_rate': 100,
                'key_positions': enhanced_traj['key_positions'],
                'blend_factor': blend_factor,
                'method': 'hybrid'
            }
        else:
            # Optimization failed - return enhanced only
            enhanced_traj['method'] = 'enhanced_fallback'
            return enhanced_traj
    
    def generate_multi_style_trajectories(self,
                                        word: str,
                                        styles: List[str] = None) -> Dict[str, Dict]:
        """
        Generate trajectories in multiple styles
        
        Args:
            word: Word to generate trajectories for
            styles: List of styles to generate
            
        Returns:
            Dictionary mapping style names to trajectory data
        """
        if styles is None:
            styles = ['precise', 'fast', 'smooth', 'natural', 'sloppy']
        
        results = {}
        
        for style in styles:
            if style == 'precise':
                # High precision, slow speed, heavily optimized
                results[style] = self.generate_hybrid_trajectory(
                    word=word,
                    blend_factor=0.8,
                    user_speed=0.7,
                    precision=0.95
                )
            elif style == 'fast':
                # Fast speed, lower precision
                results[style] = self.generate_hybrid_trajectory(
                    word=word,
                    blend_factor=0.3,
                    user_speed=1.5,
                    precision=0.6
                )
            elif style == 'smooth':
                # Smooth, optimized for minimum jerk
                opt_weights = np.array([0, 0, 0, 1])
                results[style] = self.generate_optimized_trajectory(
                    word=word,
                    optimization_weights=opt_weights,
                    pnt_density=15
                )
            elif style == 'natural':
                # Natural human-like motion
                results[style] = self.generate_hybrid_trajectory(
                    word=word,
                    blend_factor=0.5,
                    user_speed=1.0,
                    precision=0.8
                )
            elif style == 'sloppy':
                # Sloppy, fast with low precision
                traj = self.enhanced_gen.generate_trajectory(
                    word=word,
                    user_speed=1.3,
                    precision=0.4
                )
                traj['method'] = 'sloppy'
                results[style] = traj
        
        return results
    
    def _get_key_positions(self, word: str) -> np.ndarray:
        """Get (x, y) positions for each character in the word"""
        positions = []
        for char in word:
            if char in self.keyboard_df.index:
                pos = self.keyboard_df.loc[char][['x_pos', 'y_pos']].values
                positions.append(pos)
            elif char == ' ' and 'space' in self.keyboard_df.index:
                pos = self.keyboard_df.loc['space'][['x_pos', 'y_pos']].values
                positions.append(pos)
        return np.array(positions)
    
    def export_batch(self, trajectories: Dict[str, Dict], output_dir: str = './'):
        """
        Export multiple trajectories to files
        
        Args:
            trajectories: Dictionary mapping names to trajectory data
            output_dir: Directory to save files
        """
        os.makedirs(output_dir, exist_ok=True)
        
        for name, traj_data in trajectories.items():
            # Export as JSON
            json_data = self.enhanced_gen.export_trajectory(traj_data, format='json')
            json_path = os.path.join(output_dir, f'trajectory_{name}.json')
            with open(json_path, 'w') as f:
                f.write(json_data)
            
            # Export as CSV
            csv_data = self.enhanced_gen.export_trajectory(traj_data, format='csv')
            csv_path = os.path.join(output_dir, f'trajectory_{name}.csv')
            with open(csv_path, 'w') as f:
                f.write(csv_data)
            
            print(f"Exported {name}: {json_path}, {csv_path}")


# Example usage
if __name__ == '__main__':
    # Create hybrid generator
    generator = HybridSwypeGenerator()
    
    # Test word
    test_word = 'hello'
    
    print(f"\nGenerating trajectories for '{test_word}'")
    print("="*60)
    
    # Generate multiple styles
    multi_style = generator.generate_multi_style_trajectories(test_word)
    
    # Print statistics for each style
    for style, traj_data in multi_style.items():
        print(f"\nStyle: {style}")
        print(f"  Method: {traj_data.get('method', 'unknown')}")
        print(f"  Duration: {traj_data['duration']:.2f}s")
        print(f"  Points: {len(traj_data['trajectory'])}")
        
        # Calculate path length
        traj = traj_data['trajectory']
        path_length = np.sum(np.sqrt(np.diff(traj[:, 0])**2 + np.diff(traj[:, 1])**2))
        print(f"  Path length: {path_length:.2f}")
        
        # Calculate smoothness (average jerk magnitude)
        jerk = traj_data['jerk']
        jerk_mag = np.sqrt(jerk[:, 0]**2 + jerk[:, 1]**2)
        print(f"  Avg jerk: {np.mean(jerk_mag):.2f}")
    
    # Export all styles
    output_dir = './hybrid_trajectories'
    generator.export_batch(multi_style, output_dir)
    print(f"\nTrajectories exported to {output_dir}")