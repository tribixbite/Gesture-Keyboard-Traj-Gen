#!/usr/bin/env python
"""
Generator calibrated to match real swipe trace characteristics
Based on analysis of real user data
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

@dataclass
class RealTraceStats:
    """Statistics from real trace analysis"""
    avg_points_per_trace: float = 85.8
    avg_duration_seconds: float = 0.8  # Median is more realistic than mean
    avg_path_length: float = 699.7
    avg_velocity: float = 456.6  # pixels/sec
    points_per_char: float = 20.0  # Approximate
    duration_per_char: float = 0.2  # seconds

class RealCalibratedGenerator:
    """
    Swipe trajectory generator calibrated to real user data
    """
    
    def __init__(self, keyboard_layout_path: str = 'holokeyboard.csv'):
        """Initialize with keyboard layout"""
        self.keyboard_df = pd.read_csv(keyboard_layout_path, index_col='keys')
        self.real_stats = RealTraceStats()
        
        # Calibrated parameters from real data
        self.sampling_rate = 60  # Hz (match real data density)
        self.base_velocity = 450  # pixels/sec (from real data)
        self.velocity_variation = 0.3  # 30% variation
        
        # Screen scaling (real data is from mobile screens ~360x760)
        # Our keyboard uses larger coordinates
        self.coordinate_scale = 0.3  # Scale down to match mobile screens
        
    def generate_trajectory(self, word: str, style: str = 'natural') -> Dict:
        """
        Generate trajectory calibrated to real data
        
        Args:
            word: Word to generate trajectory for
            style: Style of trajectory ('natural', 'fast', 'slow', 'precise')
            
        Returns:
            Dictionary with trajectory and metadata
        """
        
        # Get key positions
        key_positions = self._get_key_positions(word)
        
        if len(key_positions) < 2:
            raise ValueError(f"Word '{word}' too short or contains invalid characters")
        
        # Calculate realistic duration based on word length
        base_duration = len(word) * self.real_stats.duration_per_char
        
        # Adjust for style
        style_multipliers = {
            'fast': 0.7,
            'natural': 1.0,
            'slow': 1.3,
            'precise': 1.2
        }
        duration = base_duration * style_multipliers.get(style, 1.0)
        
        # Calculate number of points (match real data density)
        num_points = int(duration * self.sampling_rate)
        num_points = max(num_points, len(word) * 5)  # At least 5 points per char
        num_points = min(num_points, len(word) * 30)  # At most 30 points per char
        
        # Generate base trajectory
        trajectory = self._generate_base_path(key_positions, num_points)
        
        # Add realistic noise
        trajectory = self._add_realistic_noise(trajectory, style)
        
        # Smooth to match real data characteristics
        trajectory = self._smooth_trajectory(trajectory)
        
        # Add timestamps
        timestamps = np.linspace(0, duration * 1000, len(trajectory))
        trajectory = np.column_stack([trajectory, timestamps])
        
        # Calculate metrics
        metrics = self._calculate_metrics(trajectory)
        
        return {
            'word': word,
            'trajectory': trajectory,
            'duration': duration,
            'num_points': len(trajectory),
            'style': style,
            'metrics': metrics
        }
    
    def _get_key_positions(self, word: str) -> np.ndarray:
        """Get keyboard positions for each character"""
        positions = []
        
        for char in word.lower():
            if char in self.keyboard_df.index:
                x = self.keyboard_df.at[char, 'x_pos']
                y = self.keyboard_df.at[char, 'y_pos']
                positions.append([x * self.coordinate_scale, y * self.coordinate_scale])
            elif char == ' ':
                # Space bar position (approximate)
                positions.append([0, -250 * self.coordinate_scale])
        
        return np.array(positions)
    
    def _generate_base_path(self, key_positions: np.ndarray, num_points: int) -> np.ndarray:
        """Generate base path through key positions"""
        
        # Use fewer control points for more direct paths (like real users)
        # Real users don't perfectly hit every key center
        
        # Calculate cumulative distances
        distances = [0]
        for i in range(1, len(key_positions)):
            dist = np.linalg.norm(key_positions[i] - key_positions[i-1])
            distances.append(distances[-1] + dist)
        
        total_distance = distances[-1]
        
        # Generate points along path
        t = np.linspace(0, total_distance, num_points)
        trajectory = np.zeros((num_points, 2))
        
        for i in range(num_points):
            target_dist = t[i]
            
            # Find segment
            for j in range(len(distances) - 1):
                if distances[j] <= target_dist <= distances[j + 1]:
                    # Interpolate within segment
                    segment_t = (target_dist - distances[j]) / (distances[j + 1] - distances[j] + 1e-6)
                    
                    # Add slight curve (not perfectly straight)
                    curve_factor = 0.1 * np.sin(segment_t * np.pi)
                    
                    # Linear interpolation with slight curve
                    trajectory[i] = (
                        key_positions[j] * (1 - segment_t) +
                        key_positions[j + 1] * segment_t
                    )
                    
                    # Add perpendicular offset for curve
                    if j < len(key_positions) - 1:
                        direction = key_positions[j + 1] - key_positions[j]
                        perpendicular = np.array([-direction[1], direction[0]])
                        perpendicular = perpendicular / (np.linalg.norm(perpendicular) + 1e-6)
                        trajectory[i] += perpendicular * curve_factor * 20 * self.coordinate_scale
                    
                    break
        
        return trajectory
    
    def _add_realistic_noise(self, trajectory: np.ndarray, style: str) -> np.ndarray:
        """Add noise calibrated to real user behavior"""
        
        noise_levels = {
            'precise': 5.0 * self.coordinate_scale,
            'natural': 15.0 * self.coordinate_scale,
            'fast': 25.0 * self.coordinate_scale,
            'slow': 10.0 * self.coordinate_scale
        }
        
        noise_level = noise_levels.get(style, 15.0)
        
        # Add correlated noise (fingers don't jump randomly)
        noise = np.random.randn(len(trajectory), 2) * noise_level
        
        # Smooth noise for correlation
        for i in range(1, len(noise)):
            noise[i] = 0.7 * noise[i-1] + 0.3 * noise[i]
        
        # Reduce noise near key positions (users are more accurate there)
        for i in range(len(trajectory)):
            progress = i / len(trajectory)
            # Less noise at key positions (assumed at regular intervals)
            key_proximity = abs(np.sin(progress * len(trajectory) * np.pi / 10))
            noise[i] *= (0.3 + 0.7 * key_proximity)
        
        return trajectory + noise
    
    def _smooth_trajectory(self, trajectory: np.ndarray) -> np.ndarray:
        """Smooth trajectory to match real data characteristics"""
        
        # Simple moving average (real traces are somewhat smooth but not perfect)
        window_size = 3
        smoothed = trajectory.copy()
        
        for i in range(window_size, len(trajectory) - window_size):
            smoothed[i] = np.mean(trajectory[i-window_size:i+window_size+1], axis=0)
        
        return smoothed
    
    def _calculate_metrics(self, trajectory: np.ndarray) -> Dict:
        """Calculate trajectory metrics"""
        
        # Path length
        path_length = 0
        for i in range(1, len(trajectory)):
            path_length += np.linalg.norm(trajectory[i, :2] - trajectory[i-1, :2])
        
        # Velocities
        velocities = []
        for i in range(1, len(trajectory)):
            dist = np.linalg.norm(trajectory[i, :2] - trajectory[i-1, :2])
            dt = (trajectory[i, 2] - trajectory[i-1, 2]) / 1000.0
            if dt > 0:
                velocities.append(dist / dt)
        
        return {
            'path_length': path_length,
            'mean_velocity': np.mean(velocities) if velocities else 0,
            'max_velocity': np.max(velocities) if velocities else 0,
            'points': len(trajectory)
        }


def test_calibration():
    """Test calibrated generator against real data statistics"""
    print("="*60)
    print("TESTING CALIBRATED GENERATOR")
    print("="*60)
    
    generator = RealCalibratedGenerator()
    
    test_words = ['hello', 'world', 'the', 'quick', 'brown']
    
    results = []
    for word in test_words:
        result = generator.generate_trajectory(word, style='natural')
        
        print(f"\n'{word}':")
        print(f"  Points: {result['num_points']} (target: ~{len(word) * 20})")
        print(f"  Duration: {result['duration']:.2f}s (target: ~{len(word) * 0.2:.2f}s)")
        print(f"  Path length: {result['metrics']['path_length']:.1f}")
        print(f"  Mean velocity: {result['metrics']['mean_velocity']:.1f} px/s")
        
        results.append(result)
    
    # Compare with real statistics
    avg_points = np.mean([r['num_points'] for r in results])
    avg_duration = np.mean([r['duration'] for r in results])
    avg_velocity = np.mean([r['metrics']['mean_velocity'] for r in results])
    
    print("\n" + "-"*40)
    print("COMPARISON WITH REAL DATA")
    print("-"*40)
    print(f"Average points: {avg_points:.1f} (real: 85.8)")
    print(f"Average duration: {avg_duration:.2f}s (real: 0.8s)")
    print(f"Average velocity: {avg_velocity:.1f} px/s (real: 456.6 px/s)")
    
    # Check if within acceptable range
    points_ratio = avg_points / 85.8
    duration_ratio = avg_duration / 0.8
    velocity_ratio = avg_velocity / 456.6
    
    print("\nRatios (synthetic/real):")
    print(f"  Points: {points_ratio:.2f}x")
    print(f"  Duration: {duration_ratio:.2f}x")
    print(f"  Velocity: {velocity_ratio:.2f}x")
    
    if 0.5 < points_ratio < 2.0 and 0.5 < duration_ratio < 2.0:
        print("\n✅ Generator is well-calibrated to real data!")
    else:
        print("\n⚠️  Generator needs further calibration")


if __name__ == "__main__":
    test_calibration()