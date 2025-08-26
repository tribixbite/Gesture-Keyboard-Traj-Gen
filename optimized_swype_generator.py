#!/usr/bin/env python
# coding=utf-8
"""
Optimized Swype Trajectory Generator
Addresses Gemini critique with improved algorithms and performance
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Optional, Union
import json
import warnings
from dataclasses import dataclass
from enum import Enum

# Try to import scipy for better interpolation
try:
    from scipy import signal
    from scipy.interpolate import CubicSpline
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    warnings.warn("scipy not available - using fallback methods")

@dataclass
class StyleParameters:
    """Continuous style parameters instead of discrete styles"""
    smoothness: float = 0.7      # 0-1: Jerk penalty weight
    speed: float = 1.0           # 0.5-2.0: Velocity multiplier  
    precision: float = 0.8       # 0-1: Inverse of noise amplitude
    fluidity: float = 0.6        # 0-1: Corner rounding amount
    pressure_variation: float = 0.3  # 0-1: Pen pressure variation

class OptimizedSwypeGenerator:
    """
    Optimized generator addressing performance and realism issues
    """
    
    def __init__(self, keyboard_layout_path: str = 'holokeyboard.csv'):
        """Initialize with keyboard layout"""
        self.keyboard_df = pd.read_csv(keyboard_layout_path, index_col='keys')
        
        # Physical constraints with adaptive limits
        self.base_max_velocity = 500.0
        self.base_max_acceleration = 2000.0
        self.target_jerk = 800000.0
        
        # Convergence parameters
        self.convergence_tolerance = 0.01
        self.max_iterations = 10
        
        # Savitzky-Golay parameters
        self.savgol_window = 7
        self.savgol_order = 3
    
    def generate_trajectory(self,
                           word: str,
                           sampling_rate: int = 100,
                           style: Union[str, StyleParameters] = 'natural',
                           adaptive_sampling: bool = True) -> Dict:
        """
        Generate optimized trajectory with improved algorithms
        
        Args:
            word: Word to generate trajectory for
            sampling_rate: Base sampling rate (adaptive if enabled)
            style: Style name or StyleParameters object
            adaptive_sampling: Use curvature-based adaptive sampling
            
        Returns:
            Complete trajectory data with kinematics
        """
        word = word.lower()
        
        # Convert style to parameters
        if isinstance(style, str):
            style_params = self._get_style_params(style)
        else:
            style_params = style
            
        # Get key positions
        key_positions = self._get_key_positions(word)
        if len(key_positions) == 0:
            return self._empty_trajectory()
        
        # Calculate duration with improved estimation
        duration = self._estimate_duration_advanced(word, key_positions, style_params)
        
        # Generate trajectory with proper interpolation
        if HAS_SCIPY:
            trajectory = self._generate_trajectory_scipy(
                key_positions, duration, sampling_rate, style_params, adaptive_sampling
            )
        else:
            trajectory = self._generate_trajectory_fallback(
                key_positions, duration, sampling_rate, style_params
            )
        
        # Apply curvature-dependent velocity modulation
        trajectory = self._apply_curvature_velocity(trajectory, style_params)
        
        # Apply adaptive physical constraints
        trajectory = self._apply_adaptive_constraints(trajectory, style_params)
        
        # Add realistic noise with pressure
        trajectory, pressure = self._add_noise_with_pressure(trajectory, style_params)
        
        # Calculate derivatives with Savitzky-Golay
        velocity, acceleration, jerk = self._calculate_derivatives_savgol(trajectory)
        
        # Package results
        return self._package_results(
            word, trajectory, velocity, acceleration, jerk, 
            pressure, duration, sampling_rate, style_params
        )
    
    def _get_style_params(self, style_name: str) -> StyleParameters:
        """Convert style name to continuous parameters"""
        presets = {
            'precise': StyleParameters(0.9, 0.7, 0.95, 0.8, 0.2),
            'natural': StyleParameters(0.7, 1.0, 0.8, 0.6, 0.3),
            'fast': StyleParameters(0.4, 1.5, 0.6, 0.4, 0.4),
            'sloppy': StyleParameters(0.3, 1.2, 0.5, 0.3, 0.5)
        }
        return presets.get(style_name, presets['natural'])
    
    def _get_key_positions(self, word: str) -> np.ndarray:
        """Get keyboard positions for word characters"""
        positions = []
        for char in word:
            if char in self.keyboard_df.index:
                pos = self.keyboard_df.loc[char][['x_pos', 'y_pos']].values
                positions.append(pos)
        return np.array(positions) if positions else np.array([])
    
    def _estimate_duration_advanced(self, word: str, key_positions: np.ndarray, 
                                   style: StyleParameters) -> float:
        """Advanced duration estimation based on path complexity"""
        if len(key_positions) < 2:
            return 0.15 * len(word) / style.speed
        
        # Calculate total path length
        path_length = 0
        for i in range(1, len(key_positions)):
            dist = np.linalg.norm(key_positions[i] - key_positions[i-1])
            path_length += dist
        
        # Base duration on path length and style
        base_duration = (path_length / 300.0) / style.speed  # 300 units/sec base speed
        
        # Add time for corners (direction changes)
        corner_penalty = 0.05 * (len(key_positions) - 1) * (2 - style.fluidity)
        
        return max(base_duration + corner_penalty, 0.2)
    
    def _generate_trajectory_scipy(self, key_positions: np.ndarray, duration: float,
                                   sampling_rate: int, style: StyleParameters,
                                   adaptive: bool) -> np.ndarray:
        """Generate trajectory using scipy CubicSpline with adaptive sampling"""
        n_keys = len(key_positions)
        
        # Create time points for keys with curvature-aware distribution
        key_times = self._distribute_key_times_adaptive(key_positions, duration, style)
        
        # Create cubic spline with natural boundary conditions
        cs_x = CubicSpline(key_times, key_positions[:, 0], bc_type='natural')
        cs_y = CubicSpline(key_times, key_positions[:, 1], bc_type='natural')
        
        if adaptive:
            # Adaptive sampling based on curvature
            t_samples = self._adaptive_time_samples(cs_x, cs_y, duration, sampling_rate)
        else:
            t_samples = np.linspace(0, duration, int(duration * sampling_rate))
        
        # Evaluate spline at sample points
        x_traj = cs_x(t_samples)
        y_traj = cs_y(t_samples)
        
        return np.column_stack([x_traj, y_traj, t_samples])
    
    def _generate_trajectory_fallback(self, key_positions: np.ndarray, duration: float,
                                     sampling_rate: int, style: StyleParameters) -> np.ndarray:
        """Fallback trajectory generation without scipy"""
        # Similar to improved generator but with optimizations
        n_samples = int(duration * sampling_rate)
        t_samples = np.linspace(0, duration, n_samples)
        
        key_times = self._distribute_key_times_adaptive(key_positions, duration, style)
        
        # Use numpy interp with smoothing
        x_traj = np.interp(t_samples, key_times, key_positions[:, 0])
        y_traj = np.interp(t_samples, key_times, key_positions[:, 1])
        
        # Apply smoothing
        window = min(7, n_samples // 10)
        if window >= 3:
            x_traj = self._smooth_trajectory(x_traj, window, style.smoothness)
            y_traj = self._smooth_trajectory(y_traj, window, style.smoothness)
        
        return np.column_stack([x_traj, y_traj, t_samples])
    
    def _distribute_key_times_adaptive(self, key_positions: np.ndarray, 
                                      duration: float, style: StyleParameters) -> np.ndarray:
        """Distribute key times based on path segments and style"""
        n_keys = len(key_positions)
        if n_keys == 1:
            return np.array([duration / 2])
        
        # Calculate segment lengths
        segments = []
        total_length = 0
        for i in range(1, n_keys):
            length = np.linalg.norm(key_positions[i] - key_positions[i-1])
            segments.append(length)
            total_length += length
        
        # Distribute time proportionally to segment lengths
        times = [0.05 * duration]  # Start slightly after beginning
        cumulative_time = times[0]
        
        for i, seg_length in enumerate(segments):
            # Time for this segment
            seg_time = (seg_length / total_length) * duration * 0.9
            
            # Adjust for corners (slow down at direction changes)
            if i > 0:
                v1 = key_positions[i] - key_positions[i-1]
                v2 = key_positions[i+1] - key_positions[i]
                angle = self._calculate_angle(v1, v2)
                corner_factor = 1 + (1 - style.fluidity) * (angle / np.pi)
                seg_time *= corner_factor
            
            cumulative_time += seg_time
            times.append(cumulative_time)
        
        # Normalize to fit duration
        times = np.array(times)
        times = times / times[-1] * (duration * 0.95)
        
        return times
    
    def _calculate_angle(self, v1: np.ndarray, v2: np.ndarray) -> float:
        """Calculate angle between two vectors"""
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
        return np.arccos(np.clip(cos_angle, -1, 1))
    
    def _adaptive_time_samples(self, cs_x, cs_y, duration: float, 
                              base_rate: int) -> np.ndarray:
        """Generate adaptive time samples based on curvature"""
        # Start with uniform samples
        t_uniform = np.linspace(0, duration, int(duration * base_rate / 2))
        
        # Calculate curvature at uniform points
        dx = cs_x(t_uniform, 1)  # First derivative
        dy = cs_y(t_uniform, 1)
        ddx = cs_x(t_uniform, 2)  # Second derivative
        ddy = cs_y(t_uniform, 2)
        
        # Curvature formula: |x'y'' - y'x''| / (x'^2 + y'^2)^(3/2)
        numerator = np.abs(dx * ddy - dy * ddx)
        denominator = np.power(dx**2 + dy**2 + 1e-8, 1.5)
        curvature = numerator / denominator
        
        # Add more samples where curvature is high
        additional_samples = []
        for i in range(len(t_uniform) - 1):
            if curvature[i] > np.mean(curvature):
                # Add extra samples in high curvature regions
                n_extra = int(curvature[i] / np.mean(curvature))
                extra_t = np.linspace(t_uniform[i], t_uniform[i+1], n_extra + 2)[1:-1]
                additional_samples.extend(extra_t)
        
        # Combine and sort all samples
        all_samples = np.sort(np.concatenate([t_uniform, additional_samples]))
        
        return all_samples
    
    def _apply_curvature_velocity(self, trajectory: np.ndarray, 
                                 style: StyleParameters) -> np.ndarray:
        """Apply realistic velocity modulation based on curvature"""
        if len(trajectory) < 3:
            return trajectory
        
        # Calculate local curvature
        curvature = self._calculate_curvature(trajectory)
        
        # Create velocity profile based on curvature
        # Slow down in curves, speed up in straight sections
        velocity_factor = 1.0 - style.fluidity * curvature / (np.max(curvature) + 1e-8)
        velocity_factor = np.clip(velocity_factor, 0.3, 1.0)
        
        # Apply velocity modulation by adjusting time stamps
        dt_original = np.diff(trajectory[:, 2])
        dt_adjusted = dt_original / velocity_factor[:-1]
        
        # Reconstruct time stamps
        t_new = np.zeros(len(trajectory))
        t_new[1:] = np.cumsum(dt_adjusted)
        
        # Normalize to maintain total duration
        t_new = t_new * (trajectory[-1, 2] / t_new[-1])
        trajectory[:, 2] = t_new
        
        return trajectory
    
    def _calculate_curvature(self, trajectory: np.ndarray) -> np.ndarray:
        """Calculate curvature at each point"""
        if len(trajectory) < 3:
            return np.zeros(len(trajectory))
        
        # Use finite differences for derivatives
        dx = np.gradient(trajectory[:, 0])
        dy = np.gradient(trajectory[:, 1])
        ddx = np.gradient(dx)
        ddy = np.gradient(dy)
        
        # Curvature formula
        numerator = np.abs(dx * ddy - dy * ddx)
        denominator = np.power(dx**2 + dy**2 + 1e-8, 1.5)
        curvature = numerator / denominator
        
        # Smooth curvature to reduce noise
        if HAS_SCIPY and len(curvature) > 5:
            curvature = signal.savgol_filter(curvature, 5, 2)
        
        return curvature
    
    def _apply_adaptive_constraints(self, trajectory: np.ndarray, 
                                   style: StyleParameters) -> np.ndarray:
        """Apply physical constraints with adaptive convergence"""
        max_vel = self.base_max_velocity * style.speed
        max_acc = self.base_max_acceleration * style.speed
        
        converged = False
        iterations = 0
        
        while not converged and iterations < self.max_iterations:
            old_trajectory = trajectory.copy()
            
            # Apply velocity constraints
            trajectory = self._constrain_velocity(trajectory, max_vel)
            
            # Apply acceleration constraints
            trajectory = self._constrain_acceleration(trajectory, max_acc)
            
            # Check convergence
            max_change = np.max(np.abs(trajectory[:, :2] - old_trajectory[:, :2]))
            converged = max_change < self.convergence_tolerance
            iterations += 1
        
        return trajectory
    
    def _constrain_velocity(self, trajectory: np.ndarray, max_vel: float) -> np.ndarray:
        """Constrain velocity using proper time interpolation"""
        if len(trajectory) < 2:
            return trajectory
        
        dt = np.diff(trajectory[:, 2])
        dt = np.where(dt > 0, dt, 1e-6)  # Avoid division by zero
        
        dx = np.diff(trajectory[:, 0])
        dy = np.diff(trajectory[:, 1])
        
        velocity = np.sqrt((dx/dt)**2 + (dy/dt)**2)
        
        # Find segments that exceed max velocity
        excess_mask = velocity > max_vel
        if np.any(excess_mask):
            scale_factor = max_vel / (velocity + 1e-8)
            scale_factor = np.minimum(scale_factor, 1.0)
            
            # Apply scaling to positions
            for i in np.where(excess_mask)[0]:
                trajectory[i+1, 0] = trajectory[i, 0] + dx[i] * scale_factor[i]
                trajectory[i+1, 1] = trajectory[i, 1] + dy[i] * scale_factor[i]
        
        return trajectory
    
    def _constrain_acceleration(self, trajectory: np.ndarray, max_acc: float) -> np.ndarray:
        """Constrain acceleration through smoothing"""
        if len(trajectory) < 3:
            return trajectory
        
        # Calculate acceleration
        velocity = self._calculate_velocity_robust(trajectory)
        dt = trajectory[1, 2] - trajectory[0, 2] if len(trajectory) > 1 else 0.01
        
        acc_x = np.gradient(velocity[:, 0], dt)
        acc_y = np.gradient(velocity[:, 1], dt)
        acc_magnitude = np.sqrt(acc_x**2 + acc_y**2)
        
        # Smooth regions with high acceleration
        if np.any(acc_magnitude > max_acc):
            # Apply local smoothing where needed
            window = min(5, len(trajectory) // 10)
            if window >= 3:
                trajectory[:, 0] = self._smooth_trajectory(trajectory[:, 0], window, 0.5)
                trajectory[:, 1] = self._smooth_trajectory(trajectory[:, 1], window, 0.5)
        
        return trajectory
    
    def _add_noise_with_pressure(self, trajectory: np.ndarray, 
                                style: StyleParameters) -> Tuple[np.ndarray, np.ndarray]:
        """Add noise and pressure variation"""
        n_points = len(trajectory)
        
        # Generate correlated noise
        noise_amplitude = (1 - style.precision) * 0.05
        if noise_amplitude > 0:
            noise_x = self._generate_correlated_noise(n_points, noise_amplitude)
            noise_y = self._generate_correlated_noise(n_points, noise_amplitude)
            
            # Reduce noise near key positions (simplified)
            envelope = self._create_noise_envelope(n_points, len(trajectory) // 20)
            
            trajectory[:, 0] += noise_x * envelope * 100
            trajectory[:, 1] += noise_y * envelope * 100
        
        # Generate pressure profile
        pressure = self._generate_pressure_profile(n_points, style)
        
        return trajectory, pressure
    
    def _generate_correlated_noise(self, n_points: int, amplitude: float) -> np.ndarray:
        """Generate correlated noise (not white noise)"""
        # Generate white noise
        white_noise = np.random.normal(0, amplitude, n_points)
        
        # Correlate it with moving average
        window = max(3, n_points // 50)
        kernel = np.ones(window) / window
        
        # Pad for convolution
        padded = np.pad(white_noise, window//2, mode='edge')
        correlated = np.convolve(padded, kernel, mode='valid')[:n_points]
        
        return correlated
    
    def _create_noise_envelope(self, n_points: int, n_keys: int) -> np.ndarray:
        """Create envelope that reduces noise near keys"""
        envelope = np.ones(n_points)
        
        # Simple approximation: reduce noise at regular intervals
        if n_keys > 0:
            key_interval = n_points // max(n_keys, 1)
            for i in range(0, n_points, key_interval):
                # Create Gaussian dip around key position
                sigma = key_interval / 6
                for j in range(max(0, i - key_interval//2), min(n_points, i + key_interval//2)):
                    dist = abs(j - i)
                    envelope[j] *= 0.5 + 0.5 * np.exp(-(dist**2) / (2 * sigma**2))
        
        return envelope
    
    def _generate_pressure_profile(self, n_points: int, style: StyleParameters) -> np.ndarray:
        """Generate realistic pen pressure variation"""
        if style.pressure_variation == 0:
            return np.ones(n_points)
        
        # Base pressure with slow variation
        base_freq = 2 * np.pi / n_points
        slow_variation = 0.8 + 0.2 * np.sin(base_freq * np.arange(n_points))
        
        # Add faster variations
        fast_variation = 1 + style.pressure_variation * 0.1 * np.sin(10 * base_freq * np.arange(n_points))
        
        # Combine and clip
        pressure = slow_variation * fast_variation
        pressure = np.clip(pressure, 0.5, 1.0)
        
        # Lower pressure at beginning and end
        pressure[:5] *= np.linspace(0.5, 1, 5)
        pressure[-5:] *= np.linspace(1, 0.5, 5)
        
        return pressure
    
    def _calculate_velocity_robust(self, trajectory: np.ndarray) -> np.ndarray:
        """Calculate velocity with proper handling of time"""
        if len(trajectory) < 2:
            return np.zeros((len(trajectory), 2))
        
        # Use centered differences where possible
        dt = trajectory[1, 2] - trajectory[0, 2] if len(trajectory) > 1 else 0.01
        
        # Avoid division by zero with proper epsilon
        dt = max(dt, 1e-6)
        
        vx = np.gradient(trajectory[:, 0], trajectory[:, 2])
        vy = np.gradient(trajectory[:, 1], trajectory[:, 2])
        
        return np.column_stack([vx, vy])
    
    def _calculate_derivatives_savgol(self, trajectory: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Calculate derivatives using Savitzky-Golay filter"""
        if not HAS_SCIPY or len(trajectory) < self.savgol_window:
            # Fallback to simple gradients
            velocity = self._calculate_velocity_robust(trajectory)
            dt = trajectory[1, 2] - trajectory[0, 2] if len(trajectory) > 1 else 0.01
            
            acc_x = np.gradient(velocity[:, 0], dt)
            acc_y = np.gradient(velocity[:, 1], dt)
            acceleration = np.column_stack([acc_x, acc_y])
            
            jerk_x = np.gradient(acc_x, dt)
            jerk_y = np.gradient(acc_y, dt)
            jerk = np.sqrt(jerk_x**2 + jerk_y**2)
            
            return velocity, acceleration, jerk
        
        # Use Savitzky-Golay for smooth derivatives
        dt = np.mean(np.diff(trajectory[:, 2]))
        
        # First derivative (velocity)
        vx = signal.savgol_filter(trajectory[:, 0], self.savgol_window, self.savgol_order, deriv=1, delta=dt)
        vy = signal.savgol_filter(trajectory[:, 1], self.savgol_window, self.savgol_order, deriv=1, delta=dt)
        velocity = np.column_stack([vx, vy])
        
        # Second derivative (acceleration)
        ax = signal.savgol_filter(trajectory[:, 0], self.savgol_window, self.savgol_order, deriv=2, delta=dt)
        ay = signal.savgol_filter(trajectory[:, 1], self.savgol_window, self.savgol_order, deriv=2, delta=dt)
        acceleration = np.column_stack([ax, ay])
        
        # Third derivative (jerk)
        if self.savgol_order >= 3:
            jx = signal.savgol_filter(trajectory[:, 0], self.savgol_window, self.savgol_order, deriv=3, delta=dt)
            jy = signal.savgol_filter(trajectory[:, 1], self.savgol_window, self.savgol_order, deriv=3, delta=dt)
        else:
            # Use gradient of acceleration
            jx = np.gradient(ax, dt)
            jy = np.gradient(ay, dt)
        
        jerk = np.sqrt(jx**2 + jy**2)
        
        return velocity, acceleration, jerk
    
    def _smooth_trajectory(self, data: np.ndarray, window: int, strength: float) -> np.ndarray:
        """Smooth trajectory with adjustable strength"""
        if window < 3:
            return data
        
        # Gaussian kernel with adjustable strength
        x = np.arange(window) - window // 2
        kernel = np.exp(-(x**2) / (2 * (window/4)**2))
        kernel = kernel / kernel.sum()
        
        # Blend between original and smoothed
        smoothed = np.convolve(np.pad(data, window//2, mode='edge'), kernel, mode='valid')
        
        return strength * smoothed + (1 - strength) * data
    
    def _empty_trajectory(self) -> Dict:
        """Return empty trajectory for invalid input"""
        return {
            'word': '',
            'trajectory': np.array([]),
            'velocity': np.array([]),
            'acceleration': np.array([]),
            'jerk': np.array([]),
            'pressure': np.array([]),
            'duration': 0,
            'sampling_rate': 0,
            'metadata': {}
        }
    
    def _package_results(self, word: str, trajectory: np.ndarray,
                        velocity: np.ndarray, acceleration: np.ndarray,
                        jerk: np.ndarray, pressure: np.ndarray,
                        duration: float, sampling_rate: int,
                        style: StyleParameters) -> Dict:
        """Package all results into output dictionary"""
        return {
            'word': word,
            'trajectory': trajectory,
            'velocity': velocity,
            'acceleration': acceleration,
            'jerk': jerk,
            'pressure': pressure,
            'duration': duration,
            'sampling_rate': sampling_rate,
            'metadata': {
                'mean_jerk': float(np.mean(jerk)),
                'max_jerk': float(np.max(jerk)),
                'mean_velocity': float(np.mean(np.linalg.norm(velocity, axis=1))),
                'max_velocity': float(np.max(np.linalg.norm(velocity, axis=1))),
                'style_params': {
                    'smoothness': style.smoothness,
                    'speed': style.speed,
                    'precision': style.precision,
                    'fluidity': style.fluidity,
                    'pressure_variation': style.pressure_variation
                }
            }
        }