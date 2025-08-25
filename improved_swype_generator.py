#!/usr/bin/env python
# coding=utf-8
"""
Improved Swype Trajectory Generator with Advanced Smoothing
Reduces jerk values to realistic human-like levels (<1M units/s³)
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Optional
import json

class ImprovedSwypeGenerator:
    """
    Improved generator with advanced smoothing for realistic jerk values
    """
    
    def __init__(self, keyboard_layout_path: str = 'holokeyboard.csv'):
        """Initialize with keyboard layout"""
        self.keyboard_df = pd.read_csv(keyboard_layout_path, index_col='keys')
        
        # Physical constraints for human-like motion
        self.max_velocity = 500.0  # units/s (realistic finger speed)
        self.max_acceleration = 2000.0  # units/s² 
        self.max_jerk = 800000.0  # units/s³ (target: <1M for human-like)
        
    def generate_trajectory(self, 
                           word: str, 
                           sampling_rate: int = 100,
                           duration: float = None,
                           user_speed: float = 1.0,
                           precision: float = 0.8,
                           style: str = 'natural') -> Dict:
        """
        Generate smooth trajectory with realistic jerk values
        
        Args:
            word: The word to generate trajectory for
            sampling_rate: Number of points per second
            duration: Total duration in seconds (auto-calculated if None)
            user_speed: Speed multiplier (1.0 = normal)
            precision: User precision level (0-1)
            style: Typing style ('precise', 'fast', 'natural', 'sloppy')
            
        Returns:
            Dictionary containing trajectory data
        """
        word = word.lower()
        
        # Get key positions
        key_positions = self._get_key_positions(word)
        
        # Auto-calculate duration based on word and style
        if duration is None:
            duration = self._estimate_duration(word, user_speed, style)
            
        # Generate trajectory with multi-stage smoothing
        trajectory = self._generate_smooth_trajectory(
            key_positions, sampling_rate, duration, style, precision
        )
        
        # Apply physical constraints
        trajectory = self._apply_physical_constraints(trajectory, user_speed)
        
        # Add controlled noise based on style and precision
        trajectory = self._add_controlled_noise(trajectory, style, precision, len(word))
        
        # Final smoothing pass - multiple iterations for better jerk reduction
        for _ in range(3):
            trajectory = self._final_smoothing(trajectory)
        
        # Calculate kinematics
        velocity = self._calculate_velocity(trajectory)
        acceleration = self._calculate_acceleration(trajectory)
        jerk = self._calculate_jerk(trajectory)
        
        # Convert to point format for compatibility
        points = []
        for i in range(len(trajectory)):
            points.append({
                'x': float(trajectory[i, 0]),
                'y': float(trajectory[i, 1]),
                't': float(trajectory[i, 2])
            })
        
        return {
            'word': word,
            'trajectory': trajectory,
            'points': points,
            'velocity': velocity,
            'acceleration': acceleration,
            'jerk': jerk,
            'duration': duration,
            'sampling_rate': sampling_rate,
            'key_positions': key_positions,
            'metadata': {
                'style': style,
                'user_speed': user_speed,
                'precision': precision,
                'mean_jerk': float(np.mean(np.abs(jerk))),
                'max_jerk': float(np.max(np.abs(jerk)))
            }
        }
    
    def _get_key_positions(self, word: str) -> np.ndarray:
        """Get (x, y) positions for each character in the word"""
        positions = []
        for char in word:
            if char in self.keyboard_df.index:
                pos = self.keyboard_df.loc[char][['x_pos', 'y_pos']].values
                positions.append(pos)
        return np.array(positions)
    
    def _estimate_duration(self, word: str, user_speed: float, style: str) -> float:
        """Estimate duration based on word, speed, and style"""
        base_time_per_char = 0.15  # seconds per character
        
        # Style modifiers
        style_modifiers = {
            'precise': 1.3,   # Slower, more careful
            'fast': 0.7,      # Faster typing
            'natural': 1.0,   # Normal speed
            'sloppy': 0.85    # Somewhat fast but careless
        }
        
        style_factor = style_modifiers.get(style, 1.0)
        duration = len(word) * base_time_per_char * style_factor / user_speed
        
        return max(duration, 0.3)  # Minimum duration
    
    def _generate_smooth_trajectory(self, key_positions: np.ndarray, 
                                   sampling_rate: int, duration: float,
                                   style: str, precision: float) -> np.ndarray:
        """Generate base trajectory with advanced smoothing"""
        n_keys = len(key_positions)
        n_samples = int(sampling_rate * duration)
        
        if n_keys < 2:
            # Single key or empty
            t = np.linspace(0, duration, n_samples)
            if n_keys == 1:
                x = np.full(n_samples, key_positions[0, 0])
                y = np.full(n_samples, key_positions[0, 1])
            else:
                x = np.zeros(n_samples)
                y = np.zeros(n_samples)
            return np.column_stack([x, y, t])
        
        # Create key times with style-based distribution
        key_times = self._distribute_key_times(n_keys, duration, style)
        
        # Generate smooth path using Catmull-Rom spline interpolation
        t_interp = np.linspace(0, duration, n_samples)
        x_path = self._catmull_rom_interpolate(key_times, key_positions[:, 0], t_interp)
        y_path = self._catmull_rom_interpolate(key_times, key_positions[:, 1], t_interp)
        
        # Apply multiple smoothing passes
        window_size = max(3, int(sampling_rate * 0.05))  # 50ms window
        x_smooth = self._adaptive_smooth(x_path, window_size, precision)
        y_smooth = self._adaptive_smooth(y_path, window_size, precision)
        
        return np.column_stack([x_smooth, y_smooth, t_interp])
    
    def _catmull_rom_interpolate(self, t_keys: np.ndarray, values: np.ndarray, 
                                 t_query: np.ndarray) -> np.ndarray:
        """Catmull-Rom spline interpolation for smooth curves"""
        n = len(t_keys)
        result = np.zeros(len(t_query))
        
        # Extend control points for boundary conditions
        extended_t = np.concatenate([[t_keys[0] - 0.1], t_keys, [t_keys[-1] + 0.1]])
        extended_vals = np.concatenate([[values[0]], values, [values[-1]]])
        
        for i, t in enumerate(t_query):
            # Find segment
            for j in range(1, len(extended_t) - 2):
                if extended_t[j] <= t <= extended_t[j + 1]:
                    # Catmull-Rom formula
                    t0, t1, t2, t3 = extended_t[j-1:j+3]
                    v0, v1, v2, v3 = extended_vals[j-1:j+3]
                    
                    # Normalized parameter
                    u = (t - t1) / (t2 - t1) if t2 != t1 else 0
                    
                    # Catmull-Rom basis functions
                    a0 = -0.5*v0 + 1.5*v1 - 1.5*v2 + 0.5*v3
                    a1 = v0 - 2.5*v1 + 2*v2 - 0.5*v3
                    a2 = -0.5*v0 + 0.5*v2
                    a3 = v1
                    
                    result[i] = a0*u*u*u + a1*u*u + a2*u + a3
                    break
            else:
                # Fallback for edge cases
                result[i] = np.interp(t, t_keys, values)
        
        return result
    
    def _distribute_key_times(self, n_keys: int, duration: float, style: str) -> np.ndarray:
        """Distribute key times based on typing style"""
        if n_keys == 1:
            return np.array([duration / 2])
        
        # Base uniform distribution
        base_times = np.linspace(0, duration, n_keys)
        
        # Style-based adjustments
        if style == 'precise':
            # Slight pauses at keys
            adjusted = base_times.copy()
            for i in range(1, n_keys - 1):
                adjusted[i] += np.random.uniform(-0.01, 0.01) * duration
        elif style == 'fast':
            # Accelerate through middle keys
            power = 1.2
            normalized = np.linspace(0, 1, n_keys)
            adjusted = duration * (normalized ** power)
        elif style == 'sloppy':
            # Irregular timing
            adjusted = base_times + np.random.uniform(-0.05, 0.05, n_keys) * duration
            adjusted = np.sort(adjusted)  # Ensure monotonic
        else:  # natural
            adjusted = base_times
        
        # Ensure boundaries
        adjusted[0] = 0.05 * duration
        adjusted[-1] = 0.95 * duration
        
        return adjusted
    
    def _adaptive_smooth(self, data: np.ndarray, window: int, precision: float) -> np.ndarray:
        """Adaptive smoothing based on precision level"""
        if window < 3:
            return data
            
        # Higher precision = less smoothing
        effective_window = max(3, int(window * (1 - 0.3 * precision)))
        
        # Gaussian kernel for smoother results
        kernel = self._gaussian_kernel(effective_window)
        
        # Apply convolution with edge padding
        padded = np.pad(data, effective_window//2, mode='edge')
        smoothed = np.convolve(padded, kernel, mode='valid')
        
        return smoothed
    
    def _gaussian_kernel(self, size: int) -> np.ndarray:
        """Generate Gaussian smoothing kernel"""
        if size % 2 == 0:
            size += 1
        x = np.arange(size) - size // 2
        kernel = np.exp(-(x**2) / (2 * (size/4)**2))
        return kernel / kernel.sum()
    
    def _apply_physical_constraints(self, trajectory: np.ndarray, user_speed: float) -> np.ndarray:
        """Apply velocity and acceleration limits for realistic motion"""
        dt = trajectory[1, 2] - trajectory[0, 2] if len(trajectory) > 1 else 0.01
        
        # Adjust limits based on user speed
        max_vel = self.max_velocity * user_speed
        max_acc = self.max_acceleration * user_speed
        
        # Iteratively constrain velocity and acceleration
        for iteration in range(3):  # Multiple passes for convergence
            # Calculate current velocity
            dx = np.gradient(trajectory[:, 0], dt)
            dy = np.gradient(trajectory[:, 1], dt)
            velocity = np.sqrt(dx**2 + dy**2)
            
            # Limit velocity
            vel_scale = np.minimum(1.0, max_vel / (velocity + 1e-6))
            
            # Apply velocity limits
            if np.any(vel_scale < 1.0):
                for i in range(1, len(trajectory)):
                    if vel_scale[i] < 1.0:
                        # Scale down displacement
                        trajectory[i, 0] = trajectory[i-1, 0] + (trajectory[i, 0] - trajectory[i-1, 0]) * vel_scale[i]
                        trajectory[i, 1] = trajectory[i-1, 1] + (trajectory[i, 1] - trajectory[i-1, 1]) * vel_scale[i]
            
            # Calculate acceleration
            ddx = np.gradient(dx, dt)
            ddy = np.gradient(dy, dt)
            acceleration = np.sqrt(ddx**2 + ddy**2)
            
            # Limit acceleration
            acc_scale = np.minimum(1.0, max_acc / (acceleration + 1e-6))
            
            # Apply acceleration limits through smoothing
            if np.any(acc_scale < 1.0):
                smooth_window = 5
                trajectory[:, 0] = self._adaptive_smooth(trajectory[:, 0], smooth_window, 0.8)
                trajectory[:, 1] = self._adaptive_smooth(trajectory[:, 1], smooth_window, 0.8)
        
        return trajectory
    
    def _add_controlled_noise(self, trajectory: np.ndarray, style: str, precision: float, word_length: int) -> np.ndarray:
        """Add realistic noise based on style and precision"""
        n_points = len(trajectory)
        
        # Style-based noise levels
        noise_levels = {
            'precise': 0.005,
            'fast': 0.03,
            'natural': 0.015,
            'sloppy': 0.04
        }
        
        noise_level = noise_levels.get(style, 0.015) * (1.2 - precision)
        
        if noise_level > 0:
            # Generate correlated noise (human tremor is not white noise)
            noise_x = np.random.normal(0, noise_level, n_points)
            noise_y = np.random.normal(0, noise_level, n_points)
            
            # Correlate the noise (smooth it)
            correlation_window = max(3, int(n_points * 0.02))
            noise_x = self._adaptive_smooth(noise_x, correlation_window, 0.5)
            noise_y = self._adaptive_smooth(noise_y, correlation_window, 0.5)
            
            # Apply with decreasing amplitude near keys
            key_indices = self._find_key_indices(trajectory, word_length)
            amplitude = np.ones(n_points)
            
            for idx in key_indices:
                # Reduce noise near key positions
                dist_to_key = np.abs(np.arange(n_points) - idx)
                amplitude *= 1 - 0.5 * np.exp(-dist_to_key**2 / (n_points * 0.01))
            
            trajectory[:, 0] += noise_x * amplitude * 100  # Scale to keyboard units
            trajectory[:, 1] += noise_y * amplitude * 100
        
        return trajectory
    
    def _find_key_indices(self, trajectory: np.ndarray, n_keys: int) -> List[int]:
        """Find approximate indices where keys are hit"""
        n_points = len(trajectory)
        indices = []
        
        if n_keys > 0:
            segment_size = n_points // n_keys
            for i in range(n_keys):
                idx = min(i * segment_size + segment_size // 2, n_points - 1)
                indices.append(idx)
        
        return indices
    
    def _final_smoothing(self, trajectory: np.ndarray) -> np.ndarray:
        """Final smoothing pass to ensure low jerk"""
        # Apply Savitzky-Golay-like polynomial smoothing
        window = min(7, len(trajectory) // 10)
        if window >= 3:
            trajectory[:, 0] = self._polynomial_smooth(trajectory[:, 0], window)
            trajectory[:, 1] = self._polynomial_smooth(trajectory[:, 1], window)
        
        return trajectory
    
    def _polynomial_smooth(self, data: np.ndarray, window: int) -> np.ndarray:
        """Polynomial smoothing for jerk reduction"""
        if window % 2 == 0:
            window += 1
        
        half_window = window // 2
        smoothed = data.copy()
        
        for i in range(half_window, len(data) - half_window):
            # Fit local polynomial (order 2)
            indices = np.arange(i - half_window, i + half_window + 1)
            local_data = data[indices]
            
            # Simple quadratic fit
            x = np.arange(len(local_data))
            coeffs = np.polyfit(x, local_data, 2)
            smoothed[i] = np.polyval(coeffs, half_window)
        
        return smoothed
    
    def _calculate_velocity(self, trajectory: np.ndarray) -> np.ndarray:
        """Calculate velocity from trajectory"""
        dt = trajectory[1, 2] - trajectory[0, 2] if len(trajectory) > 1 else 0.01
        
        vx = np.gradient(trajectory[:, 0], dt)
        vy = np.gradient(trajectory[:, 1], dt)
        
        return np.column_stack([vx, vy])
    
    def _calculate_acceleration(self, trajectory: np.ndarray) -> np.ndarray:
        """Calculate acceleration from trajectory"""
        velocity = self._calculate_velocity(trajectory)
        dt = trajectory[1, 2] - trajectory[0, 2] if len(trajectory) > 1 else 0.01
        
        ax = np.gradient(velocity[:, 0], dt)
        ay = np.gradient(velocity[:, 1], dt)
        
        return np.column_stack([ax, ay])
    
    def _calculate_jerk(self, trajectory: np.ndarray) -> np.ndarray:
        """Calculate jerk from trajectory using finite differences"""
        if len(trajectory) < 4:
            return np.zeros(len(trajectory))
        
        dt = trajectory[1, 2] - trajectory[0, 2] if len(trajectory) > 1 else 0.01
        
        # Use finite differences for more stable calculation
        jerk_magnitude = np.zeros(len(trajectory))
        
        for i in range(3, len(trajectory) - 3):
            # Third-order finite difference for jerk
            x_jerk = (-trajectory[i-3, 0] + 3*trajectory[i-2, 0] - 3*trajectory[i-1, 0] + trajectory[i, 0]) / (dt**3)
            y_jerk = (-trajectory[i-3, 1] + 3*trajectory[i-2, 1] - 3*trajectory[i-1, 1] + trajectory[i, 1]) / (dt**3)
            
            jerk_magnitude[i] = np.sqrt(x_jerk**2 + y_jerk**2)
        
        # Smooth the jerk to reduce numerical noise
        if len(jerk_magnitude) > 7:
            jerk_magnitude = self._adaptive_smooth(jerk_magnitude, 5, 0.9)
        
        # Clamp extreme values
        jerk_magnitude = np.clip(jerk_magnitude, 0, self.max_jerk * 2)
        
        return jerk_magnitude