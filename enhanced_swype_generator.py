#!/usr/bin/env python
# coding=utf-8
"""
Enhanced Swype Trajectory Generator
Combines and improves upon existing models for realistic swype trace generation
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Optional
import json

# Make matplotlib optional
try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

class EnhancedSwypeGenerator:
    """
    Enhanced generator for synthetic swype trajectories with improved realism
    """
    
    def __init__(self, keyboard_layout_path: str = 'holokeyboard.csv'):
        """
        Initialize the generator with keyboard layout
        
        Args:
            keyboard_layout_path: Path to CSV file containing keyboard layout
        """
        self.keyboard_df = pd.read_csv(keyboard_layout_path, index_col='keys')
        self.velocity_profile = 'bell'  # Options: 'bell', 'uniform', 'realistic'
        self.corner_smoothing = True
        self.add_noise = True
        self.noise_level = 0.02  # Relative to keyboard size
        
    def generate_trajectory(self, 
                           word: str, 
                           sampling_rate: int = 100,
                           duration: float = None,
                           user_speed: float = 1.0,
                           precision: float = 0.8) -> Dict:
        """
        Generate a complete swype trajectory for a word
        
        Args:
            word: The word to generate trajectory for
            sampling_rate: Number of points per second
            duration: Total duration in seconds (auto-calculated if None)
            user_speed: Speed multiplier (1.0 = normal, >1 = faster)
            precision: User precision level (0-1, affects noise and smoothing)
            
        Returns:
            Dictionary containing trajectory data
        """
        word = word.lower()
        
        # Get key positions
        key_positions = self._get_key_positions(word)
        
        # Auto-calculate duration based on word length if not provided
        if duration is None:
            duration = self._estimate_duration(word, user_speed)
            
        # Generate base trajectory
        trajectory = self._generate_base_trajectory(key_positions, sampling_rate, duration)
        
        # Apply velocity profile
        trajectory = self._apply_velocity_profile(trajectory, self.velocity_profile)
        
        # Apply corner smoothing
        if self.corner_smoothing:
            trajectory = self._smooth_corners(trajectory, precision)
            
        # Add realistic noise
        if self.add_noise:
            trajectory = self._add_realistic_noise(trajectory, precision)
            
        # Calculate additional features
        velocity = self._calculate_velocity(trajectory)
        acceleration = self._calculate_acceleration(velocity)
        jerk = self._calculate_jerk(acceleration)
        
        return {
            'word': word,
            'trajectory': trajectory,
            'velocity': velocity,
            'acceleration': acceleration,
            'jerk': jerk,
            'duration': duration,
            'sampling_rate': sampling_rate,
            'key_positions': key_positions
        }
    
    def _get_key_positions(self, word: str) -> np.ndarray:
        """Get (x, y) positions for each character in the word"""
        positions = []
        for char in word:
            if char in self.keyboard_df.index:
                pos = self.keyboard_df.loc[char][['x_pos', 'y_pos']].values
                positions.append(pos)
            else:
                # Handle space or unknown characters
                if char == ' ':
                    # Use space bar position if available
                    if 'space' in self.keyboard_df.index:
                        pos = self.keyboard_df.loc['space'][['x_pos', 'y_pos']].values
                        positions.append(pos)
        return np.array(positions)
    
    def _estimate_duration(self, word: str, user_speed: float) -> float:
        """Estimate duration based on word length and user speed"""
        base_time_per_char = 0.15  # seconds per character
        word_length_factor = 1.0 + 0.05 * max(0, len(word) - 4)  # Longer words take proportionally more time
        return (len(word) * base_time_per_char * word_length_factor) / user_speed
    
    def _generate_base_trajectory(self, key_positions: np.ndarray, 
                                 sampling_rate: int, duration: float) -> np.ndarray:
        """Generate base trajectory using polynomial interpolation"""
        n_keys = len(key_positions)
        n_samples = int(sampling_rate * duration)
        
        # Create time points for keys (not uniformly distributed)
        key_times = self._distribute_key_times(n_keys, duration)
        
        # Interpolation time points
        t_interp = np.linspace(0, duration, n_samples)
        
        # Use polynomial interpolation for smooth trajectory
        x_interp = np.interp(t_interp, key_times, key_positions[:, 0])
        y_interp = np.interp(t_interp, key_times, key_positions[:, 1])
        
        # Apply smoothing to make it more like a spline
        x_smooth = self._smooth_array(x_interp, window_size=5)
        y_smooth = self._smooth_array(y_interp, window_size=5)
        
        trajectory = np.column_stack([x_smooth, y_smooth, t_interp])
        
        return trajectory
    
    def _smooth_array(self, arr: np.ndarray, window_size: int = 5) -> np.ndarray:
        """Apply moving average smoothing to an array"""
        if window_size < 1:
            return arr
        if window_size % 2 == 0:
            window_size += 1  # Make sure window size is odd
        
        # Simple moving average
        kernel = np.ones(window_size) / window_size
        # Pad the array to handle edges
        padded = np.pad(arr, (window_size//2, window_size//2), mode='edge')
        smoothed = np.convolve(padded, kernel, mode='valid')
        return smoothed
    
    def _distribute_key_times(self, n_keys: int, duration: float) -> np.ndarray:
        """Distribute key times with realistic timing patterns"""
        if n_keys == 1:
            return np.array([duration / 2])
        
        # Start and end slightly inside the total duration
        start_offset = 0.05 * duration
        end_offset = 0.95 * duration
        
        # Create non-uniform distribution (slower at start/end)
        base_times = np.linspace(0, 1, n_keys)
        
        # Apply sigmoid-like transformation for more realistic timing
        adjusted_times = 1 / (1 + np.exp(-10 * (base_times - 0.5)))
        adjusted_times = (adjusted_times - adjusted_times.min()) / (adjusted_times.max() - adjusted_times.min())
        
        key_times = start_offset + adjusted_times * (end_offset - start_offset)
        
        return key_times
    
    def _apply_velocity_profile(self, trajectory: np.ndarray, profile: str) -> np.ndarray:
        """Apply velocity profile to trajectory"""
        if profile == 'uniform':
            return trajectory
        elif profile == 'bell':
            # Bell-shaped velocity profile (slow-fast-slow)
            n_points = len(trajectory)
            t = np.linspace(0, 1, n_points)
            
            # Bell curve multiplier
            velocity_mult = np.exp(-((t - 0.5) ** 2) / (2 * 0.15 ** 2))
            velocity_mult = 0.3 + 0.7 * velocity_mult  # Scale between 0.3 and 1.0
            
            # Apply to trajectory positions (not time)
            center = trajectory[n_points // 2, :2]
            for i in range(n_points):
                if i != n_points // 2:
                    direction = trajectory[i, :2] - center
                    trajectory[i, :2] = center + direction * velocity_mult[i]
                    
        elif profile == 'realistic':
            # More complex realistic profile with acceleration/deceleration
            n_points = len(trajectory)
            t = np.linspace(0, 1, n_points)
            
            # Combination of sigmoid functions for acceleration and deceleration
            accel_phase = 1 / (1 + np.exp(-15 * (t - 0.15)))
            decel_phase = 1 - 1 / (1 + np.exp(-15 * (t - 0.85)))
            velocity_mult = accel_phase * decel_phase
            velocity_mult = 0.2 + 0.8 * (velocity_mult / velocity_mult.max())
            
            # Apply temporal stretching
            new_times = np.cumsum(1 / velocity_mult)
            new_times = new_times / new_times[-1] * trajectory[-1, 2]
            trajectory[:, 2] = new_times
            
        return trajectory
    
    def _smooth_corners(self, trajectory: np.ndarray, precision: float) -> np.ndarray:
        """Apply corner smoothing for more natural movements"""
        window_size = int(5 * (1.1 - precision))  # Less smoothing for higher precision
        if window_size < 3:
            window_size = 3
        
        trajectory[:, 0] = self._smooth_array(trajectory[:, 0], window_size)
        trajectory[:, 1] = self._smooth_array(trajectory[:, 1], window_size)
        
        return trajectory
    
    def _add_realistic_noise(self, trajectory: np.ndarray, precision: float) -> np.ndarray:
        """Add realistic noise to simulate human imprecision"""
        n_points = len(trajectory)
        
        # Calculate keyboard dimensions for scaling
        kbd_width = self.keyboard_df['x_pos'].max() - self.keyboard_df['x_pos'].min()
        kbd_height = self.keyboard_df['y_pos'].max() - self.keyboard_df['y_pos'].min()
        
        # Noise decreases with precision
        noise_scale = self.noise_level * (1.0 - precision)
        
        # Generate correlated noise (human tremor is not white noise)
        noise_x = np.random.randn(n_points) * kbd_width * noise_scale
        noise_y = np.random.randn(n_points) * kbd_height * noise_scale
        
        # Smooth the noise to make it more realistic
        noise_x = self._smooth_array(noise_x, window_size=3)
        noise_y = self._smooth_array(noise_y, window_size=3)
        
        # Reduce noise at key positions (higher accuracy at targets)
        key_indices = self._find_key_indices(trajectory)
        for idx in key_indices:
            window = 5
            start = max(0, idx - window)
            end = min(n_points, idx + window + 1)
            
            # Taper noise near keys
            for i in range(start, end):
                dist_to_key = abs(i - idx) / window
                noise_x[i] *= dist_to_key
                noise_y[i] *= dist_to_key
        
        trajectory[:, 0] += noise_x
        trajectory[:, 1] += noise_y
        
        return trajectory
    
    def _find_key_indices(self, trajectory: np.ndarray) -> List[int]:
        """Find indices in trajectory closest to key positions"""
        # This is a simplified version - could be improved with actual key detection
        n_points = len(trajectory)
        n_segments = len(self.keyboard_df.loc[self.keyboard_df.index[0]])  # Approximate
        
        indices = []
        segment_size = n_points // max(n_segments, 3)
        
        for i in range(0, n_points, segment_size):
            indices.append(i)
            
        return indices
    
    def _calculate_velocity(self, trajectory: np.ndarray) -> np.ndarray:
        """Calculate velocity from trajectory"""
        dt = np.diff(trajectory[:, 2])
        dx = np.diff(trajectory[:, 0])
        dy = np.diff(trajectory[:, 1])
        
        vx = dx / dt
        vy = dy / dt
        
        # Pad to match trajectory length
        vx = np.concatenate([[vx[0]], vx])
        vy = np.concatenate([[vy[0]], vy])
        
        return np.column_stack([vx, vy])
    
    def _calculate_acceleration(self, velocity: np.ndarray) -> np.ndarray:
        """Calculate acceleration from velocity"""
        dt = 1.0 / 100  # Assuming standard sampling rate
        
        ax = np.gradient(velocity[:, 0], dt)
        ay = np.gradient(velocity[:, 1], dt)
        
        return np.column_stack([ax, ay])
    
    def _calculate_jerk(self, acceleration: np.ndarray) -> np.ndarray:
        """Calculate jerk from acceleration"""
        dt = 1.0 / 100  # Assuming standard sampling rate
        
        jx = np.gradient(acceleration[:, 0], dt)
        jy = np.gradient(acceleration[:, 1], dt)
        
        return np.column_stack([jx, jy])
    
    def visualize_trajectory(self, trajectory_data: Dict, save_path: Optional[str] = None):
        """Visualize the generated trajectory"""
        if not HAS_MATPLOTLIB:
            print("Warning: matplotlib not available. Cannot visualize trajectory.")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        trajectory = trajectory_data['trajectory']
        velocity = trajectory_data['velocity']
        acceleration = trajectory_data['acceleration']
        
        # Plot trajectory
        ax = axes[0, 0]
        ax.plot(trajectory[:, 0], trajectory[:, 1], 'b-', alpha=0.6, linewidth=2)
        ax.scatter(trajectory[0, 0], trajectory[0, 1], c='green', s=100, marker='o', label='Start')
        ax.scatter(trajectory[-1, 0], trajectory[-1, 1], c='red', s=100, marker='s', label='End')
        
        # Mark key positions
        for pos in trajectory_data['key_positions']:
            ax.scatter(pos[0], pos[1], c='black', s=50, marker='x', alpha=0.5)
        
        ax.set_title(f"Swype Trajectory: '{trajectory_data['word']}'")
        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.invert_yaxis()  # Invert Y axis for screen coordinates
        
        # Plot velocity magnitude
        ax = axes[0, 1]
        velocity_mag = np.sqrt(velocity[:, 0]**2 + velocity[:, 1]**2)
        time = trajectory[:, 2]
        ax.plot(time, velocity_mag)
        ax.set_title('Velocity Profile')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Velocity')
        ax.grid(True, alpha=0.3)
        
        # Plot acceleration magnitude
        ax = axes[1, 0]
        accel_mag = np.sqrt(acceleration[:, 0]**2 + acceleration[:, 1]**2)
        ax.plot(time, accel_mag)
        ax.set_title('Acceleration Profile')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Acceleration')
        ax.grid(True, alpha=0.3)
        
        # Plot jerk magnitude
        ax = axes[1, 1]
        jerk = trajectory_data['jerk']
        jerk_mag = np.sqrt(jerk[:, 0]**2 + jerk[:, 1]**2)
        ax.plot(time, jerk_mag)
        ax.set_title('Jerk Profile')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Jerk')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        plt.show()
        
    def export_trajectory(self, trajectory_data: Dict, format: str = 'json') -> str:
        """Export trajectory data in various formats"""
        if format == 'json':
            # Convert numpy arrays to lists for JSON serialization
            export_data = {
                'word': trajectory_data['word'],
                'duration': trajectory_data['duration'],
                'sampling_rate': trajectory_data['sampling_rate'],
                'trajectory': trajectory_data['trajectory'].tolist(),
                'velocity': trajectory_data['velocity'].tolist(),
                'acceleration': trajectory_data['acceleration'].tolist(),
                'jerk': trajectory_data['jerk'].tolist(),
                'key_positions': trajectory_data['key_positions'].tolist()
            }
            return json.dumps(export_data, indent=2)
        
        elif format == 'csv':
            # Create DataFrame for CSV export
            df = pd.DataFrame({
                'time': trajectory_data['trajectory'][:, 2],
                'x': trajectory_data['trajectory'][:, 0],
                'y': trajectory_data['trajectory'][:, 1],
                'vx': trajectory_data['velocity'][:, 0],
                'vy': trajectory_data['velocity'][:, 1],
                'ax': trajectory_data['acceleration'][:, 0],
                'ay': trajectory_data['acceleration'][:, 1],
                'jx': trajectory_data['jerk'][:, 0],
                'jy': trajectory_data['jerk'][:, 1]
            })
            return df.to_csv(index=False)
        
        else:
            raise ValueError(f"Unsupported format: {format}")


# Example usage
if __name__ == '__main__':
    # Create generator
    generator = EnhancedSwypeGenerator()
    
    # Generate trajectory for a word
    trajectory_data = generator.generate_trajectory(
        word='hello',
        user_speed=1.0,
        precision=0.8
    )
    
    # Visualize the trajectory (if matplotlib is available)
    if HAS_MATPLOTLIB:
        generator.visualize_trajectory(trajectory_data, save_path='swype_trajectory_hello.png')
    else:
        print("Skipping visualization (matplotlib not available)")
    
    # Export to JSON
    json_data = generator.export_trajectory(trajectory_data, format='json')
    with open('trajectory_hello.json', 'w') as f:
        f.write(json_data)
    
    print(f"Generated trajectory for '{trajectory_data['word']}'")
    print(f"Duration: {trajectory_data['duration']:.2f} seconds")
    print(f"Number of points: {len(trajectory_data['trajectory'])}")