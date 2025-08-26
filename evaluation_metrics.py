#!/usr/bin/env python
"""
Comprehensive evaluation metrics for trajectory generation models
Includes DTW, Fréchet distance, smoothness metrics, and visual comparison
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
# import matplotlib.pyplot as plt  # Optional for plotting
from pathlib import Path
try:
    from scipy.spatial.distance import directed_hausdorff
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
import json
from datetime import datetime

try:
    from dtaidistance import dtw
    HAS_DTW = True
except ImportError:
    HAS_DTW = False
    print("Warning: dtaidistance not installed, using simple DTW implementation")

class TrajectoryMetrics:
    """Calculate various metrics for trajectory quality evaluation"""
    
    def __init__(self):
        self.metrics_history = []
    
    def calculate_smoothness(self, trajectory: np.ndarray) -> Dict[str, float]:
        """
        Calculate smoothness metrics: velocity, acceleration, jerk
        
        Args:
            trajectory: Shape (n_points, 2) or (n_points, 3) with x, y, [pen]
            
        Returns:
            Dictionary with smoothness metrics
        """
        if len(trajectory) < 3:
            return {'velocity': 0, 'acceleration': 0, 'jerk': 0}
        
        # Extract x, y coordinates
        positions = trajectory[:, :2]
        
        # Calculate derivatives
        dt = 1.0 / 120  # Assume 120Hz sampling
        
        # Velocity (first derivative)
        velocity = np.diff(positions, axis=0) / dt
        velocity_magnitude = np.linalg.norm(velocity, axis=1)
        
        # Acceleration (second derivative)
        acceleration = np.diff(velocity, axis=0) / dt
        acceleration_magnitude = np.linalg.norm(acceleration, axis=1)
        
        # Jerk (third derivative)
        jerk = np.diff(acceleration, axis=0) / dt
        jerk_magnitude = np.linalg.norm(jerk, axis=1)
        
        return {
            'velocity_mean': float(np.mean(velocity_magnitude)) if len(velocity_magnitude) > 0 else 0,
            'velocity_max': float(np.max(velocity_magnitude)) if len(velocity_magnitude) > 0 else 0,
            'velocity_std': float(np.std(velocity_magnitude)) if len(velocity_magnitude) > 0 else 0,
            'acceleration_mean': float(np.mean(acceleration_magnitude)) if len(acceleration_magnitude) > 0 else 0,
            'acceleration_max': float(np.max(acceleration_magnitude)) if len(acceleration_magnitude) > 0 else 0,
            'acceleration_std': float(np.std(acceleration_magnitude)) if len(acceleration_magnitude) > 0 else 0,
            'jerk_mean': float(np.mean(jerk_magnitude)) if len(jerk_magnitude) > 0 else 0,
            'jerk_max': float(np.max(jerk_magnitude)) if len(jerk_magnitude) > 0 else 0,
            'jerk_std': float(np.std(jerk_magnitude)) if len(jerk_magnitude) > 0 else 0,
            'smoothness_score': float(1.0 / (1.0 + (np.mean(jerk_magnitude) if len(jerk_magnitude) > 0 else 0) / 1000))
        }
    
    def calculate_dtw_distance(self, traj1: np.ndarray, traj2: np.ndarray) -> float:
        """
        Calculate Dynamic Time Warping distance between two trajectories
        
        Args:
            traj1, traj2: Trajectories with shape (n_points, 2)
            
        Returns:
            DTW distance (normalized by trajectory length)
        """
        if HAS_DTW:
            # Use fast DTW implementation
            distance = dtw.distance(traj1[:, 0], traj2[:, 0])
            distance += dtw.distance(traj1[:, 1], traj2[:, 1])
        else:
            # Simple DTW implementation
            distance = self._simple_dtw(traj1, traj2)
        
        # Normalize by trajectory length
        return distance / max(len(traj1), len(traj2))
    
    def _simple_dtw(self, traj1: np.ndarray, traj2: np.ndarray) -> float:
        """Simple DTW implementation for when dtaidistance is not available"""
        n, m = len(traj1), len(traj2)
        dtw_matrix = np.full((n + 1, m + 1), np.inf)
        dtw_matrix[0, 0] = 0
        
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                cost = np.linalg.norm(traj1[i-1] - traj2[j-1])
                dtw_matrix[i, j] = cost + min(
                    dtw_matrix[i-1, j],    # insertion
                    dtw_matrix[i, j-1],    # deletion
                    dtw_matrix[i-1, j-1]   # match
                )
        
        return dtw_matrix[n, m]
    
    def calculate_frechet_distance(self, traj1: np.ndarray, traj2: np.ndarray) -> float:
        """
        Calculate discrete Fréchet distance between trajectories
        
        Args:
            traj1, traj2: Trajectories with shape (n_points, 2)
            
        Returns:
            Fréchet distance
        """
        def euclidean(p1, p2):
            return np.linalg.norm(p1 - p2)
        
        n, m = len(traj1), len(traj2)
        ca = np.full((n, m), -1.0)
        
        def compute(i, j):
            if ca[i, j] != -1:
                return ca[i, j]
            
            if i == 0 and j == 0:
                ca[i, j] = euclidean(traj1[0], traj2[0])
            elif i == 0:
                ca[i, j] = max(compute(0, j-1), euclidean(traj1[0], traj2[j]))
            elif j == 0:
                ca[i, j] = max(compute(i-1, 0), euclidean(traj1[i], traj2[0]))
            else:
                ca[i, j] = max(
                    min(compute(i-1, j), compute(i, j-1), compute(i-1, j-1)),
                    euclidean(traj1[i], traj2[j])
                )
            
            return ca[i, j]
        
        return compute(n-1, m-1)
    
    def calculate_hausdorff_distance(self, traj1: np.ndarray, traj2: np.ndarray) -> Dict[str, float]:
        """
        Calculate Hausdorff distance (both directions)
        
        Args:
            traj1, traj2: Trajectories with shape (n_points, 2)
            
        Returns:
            Dictionary with forward, backward, and symmetric Hausdorff distances
        """
        if HAS_SCIPY:
            forward = directed_hausdorff(traj1[:, :2], traj2[:, :2])[0]
            backward = directed_hausdorff(traj2[:, :2], traj1[:, :2])[0]
        else:
            # Simple implementation without scipy
            def simple_hausdorff(A, B):
                max_dist = 0
                for a in A:
                    min_dist = float('inf')
                    for b in B:
                        dist = np.linalg.norm(a - b)
                        min_dist = min(min_dist, dist)
                    max_dist = max(max_dist, min_dist)
                return max_dist
            
            forward = simple_hausdorff(traj1[:, :2], traj2[:, :2])
            backward = simple_hausdorff(traj2[:, :2], traj1[:, :2])
        
        return {
            'hausdorff_forward': float(forward),
            'hausdorff_backward': float(backward),
            'hausdorff_symmetric': float(max(forward, backward))
        }
    
    def calculate_trajectory_length(self, trajectory: np.ndarray) -> float:
        """Calculate total length of trajectory"""
        if len(trajectory) < 2:
            return 0.0
        
        positions = trajectory[:, :2]
        segments = np.diff(positions, axis=0)
        lengths = np.linalg.norm(segments, axis=1)
        return float(np.sum(lengths))
    
    def calculate_curvature(self, trajectory: np.ndarray) -> Dict[str, float]:
        """Calculate curvature statistics"""
        if len(trajectory) < 3:
            return {'curvature_mean': 0, 'curvature_max': 0, 'curvature_total': 0}
        
        positions = trajectory[:, :2]
        
        # Calculate curvature using three consecutive points
        curvatures = []
        for i in range(1, len(positions) - 1):
            p1, p2, p3 = positions[i-1], positions[i], positions[i+1]
            
            # Menger curvature formula
            area = 0.5 * np.abs((p2[0] - p1[0]) * (p3[1] - p1[1]) - 
                               (p3[0] - p1[0]) * (p2[1] - p1[1]))
            a = np.linalg.norm(p2 - p1)
            b = np.linalg.norm(p3 - p2)
            c = np.linalg.norm(p3 - p1)
            
            if a * b * c > 0:
                curvature = 4 * area / (a * b * c)
            else:
                curvature = 0
            
            curvatures.append(curvature)
        
        curvatures = np.array(curvatures)
        
        return {
            'curvature_mean': float(np.mean(curvatures)),
            'curvature_max': float(np.max(curvatures)),
            'curvature_std': float(np.std(curvatures)),
            'curvature_total': float(np.sum(curvatures))
        }
    
    def evaluate_trajectory(self, 
                          generated: np.ndarray, 
                          reference: Optional[np.ndarray] = None,
                          word: Optional[str] = None) -> Dict:
        """
        Comprehensive evaluation of a generated trajectory
        
        Args:
            generated: Generated trajectory
            reference: Optional reference trajectory for comparison
            word: Optional word associated with trajectory
            
        Returns:
            Dictionary with all metrics
        """
        metrics = {
            'timestamp': datetime.now().isoformat(),
            'word': word,
            'trajectory_points': len(generated),
            'trajectory_length': self.calculate_trajectory_length(generated)
        }
        
        # Smoothness metrics
        metrics.update(self.calculate_smoothness(generated))
        
        # Curvature metrics
        metrics.update(self.calculate_curvature(generated))
        
        # If reference trajectory provided, calculate similarity metrics
        if reference is not None:
            metrics['reference_points'] = len(reference)
            metrics['reference_length'] = self.calculate_trajectory_length(reference)
            
            # Ensure same shape for comparison
            gen_2d = generated[:, :2]
            ref_2d = reference[:, :2]
            
            # DTW distance
            metrics['dtw_distance'] = self.calculate_dtw_distance(gen_2d, ref_2d)
            
            # Fréchet distance
            metrics['frechet_distance'] = self.calculate_frechet_distance(gen_2d, ref_2d)
            
            # Hausdorff distance
            metrics.update(self.calculate_hausdorff_distance(gen_2d, ref_2d))
            
            # Length difference ratio
            metrics['length_ratio'] = metrics['trajectory_length'] / (metrics['reference_length'] + 1e-6)
        
        # Store in history
        self.metrics_history.append(metrics)
        
        return metrics
    
    def evaluate_batch(self, 
                      generated_batch: List[np.ndarray],
                      reference_batch: Optional[List[np.ndarray]] = None,
                      words: Optional[List[str]] = None) -> Dict:
        """
        Evaluate a batch of trajectories
        
        Returns aggregate statistics
        """
        all_metrics = []
        
        for i, gen in enumerate(generated_batch):
            ref = reference_batch[i] if reference_batch else None
            word = words[i] if words else None
            metrics = self.evaluate_trajectory(gen, ref, word)
            all_metrics.append(metrics)
        
        # Aggregate metrics
        aggregate = {}
        
        # Get all numeric keys
        numeric_keys = [k for k in all_metrics[0].keys() 
                       if isinstance(all_metrics[0][k], (int, float))]
        
        for key in numeric_keys:
            values = [m[key] for m in all_metrics if key in m]
            if values:
                aggregate[f'{key}_mean'] = np.mean(values)
                aggregate[f'{key}_std'] = np.std(values)
                aggregate[f'{key}_min'] = np.min(values)
                aggregate[f'{key}_max'] = np.max(values)
        
        aggregate['num_trajectories'] = len(generated_batch)
        aggregate['timestamp'] = datetime.now().isoformat()
        
        return aggregate
    
    def save_metrics(self, filepath: str):
        """Save metrics history to JSON file"""
        with open(filepath, 'w') as f:
            json.dump(self.metrics_history, f, indent=2)
        print(f"✅ Saved {len(self.metrics_history)} metrics to {filepath}")
    
    def plot_comparison(self, 
                       generated: np.ndarray,
                       reference: Optional[np.ndarray] = None,
                       title: str = "Trajectory Comparison",
                       save_path: Optional[str] = None):
        """
        Plot trajectory comparison (requires matplotlib)
        
        Args:
            generated: Generated trajectory
            reference: Optional reference trajectory
            title: Plot title
            save_path: Optional path to save figure
        """
        try:
            import matplotlib.pyplot as plt
            
            fig, axes = plt.subplots(1, 2 if reference is not None else 1, 
                                    figsize=(12 if reference is not None else 6, 5))
            
            if reference is not None:
                # Plot reference
                ax = axes[0]
                ax.plot(reference[:, 0], reference[:, 1], 'b-', linewidth=2, alpha=0.7, label='Reference')
                ax.scatter(reference[0, 0], reference[0, 1], color='green', s=100, marker='o', label='Start')
                ax.scatter(reference[-1, 0], reference[-1, 1], color='red', s=100, marker='s', label='End')
                ax.set_title('Reference Trajectory')
                ax.set_xlabel('X')
                ax.set_ylabel('Y')
                ax.legend()
                ax.grid(True, alpha=0.3)
                ax.set_aspect('equal')
                
                # Plot generated
                ax = axes[1]
            else:
                ax = axes
            
            ax.plot(generated[:, 0], generated[:, 1], 'r-', linewidth=2, alpha=0.7, label='Generated')
            ax.scatter(generated[0, 0], generated[0, 1], color='green', s=100, marker='o', label='Start')
            ax.scatter(generated[-1, 0], generated[-1, 1], color='red', s=100, marker='s', label='End')
            ax.set_title('Generated Trajectory')
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_aspect('equal')
            
            plt.suptitle(title, fontsize=14, fontweight='bold')
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=100, bbox_inches='tight')
                print(f"✅ Saved plot to {save_path}")
            
            plt.show()
            plt.close()
        except ImportError:
            print("⚠️ Matplotlib not available, skipping plot")
    
    def generate_report(self, output_path: str = "evaluation_report.md"):
        """Generate markdown evaluation report"""
        if not self.metrics_history:
            print("No metrics to report")
            return
        
        report = []
        report.append("# Trajectory Generation Evaluation Report")
        report.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"\nTotal trajectories evaluated: {len(self.metrics_history)}")
        
        # Aggregate statistics
        report.append("\n## Aggregate Statistics\n")
        
        # Get numeric metrics
        numeric_metrics = {}
        for metrics in self.metrics_history:
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    if key not in numeric_metrics:
                        numeric_metrics[key] = []
                    numeric_metrics[key].append(value)
        
        report.append("| Metric | Mean | Std | Min | Max |")
        report.append("|--------|------|-----|-----|-----|")
        
        for key, values in numeric_metrics.items():
            if values:
                mean = np.mean(values)
                std = np.std(values)
                min_val = np.min(values)
                max_val = np.max(values)
                report.append(f"| {key} | {mean:.4f} | {std:.4f} | {min_val:.4f} | {max_val:.4f} |")
        
        # Quality scores
        report.append("\n## Quality Summary\n")
        
        if 'smoothness_score' in numeric_metrics:
            avg_smoothness = np.mean(numeric_metrics['smoothness_score'])
            report.append(f"- **Average Smoothness Score**: {avg_smoothness:.3f}")
        
        if 'dtw_distance' in numeric_metrics:
            avg_dtw = np.mean(numeric_metrics['dtw_distance'])
            report.append(f"- **Average DTW Distance**: {avg_dtw:.3f}")
        
        if 'frechet_distance' in numeric_metrics:
            avg_frechet = np.mean(numeric_metrics['frechet_distance'])
            report.append(f"- **Average Fréchet Distance**: {avg_frechet:.3f}")
        
        # Save report
        with open(output_path, 'w') as f:
            f.write('\n'.join(report))
        
        print(f"✅ Generated report: {output_path}")
        return '\n'.join(report)


def test_metrics():
    """Test the metrics system"""
    print("Testing Trajectory Metrics System")
    print("="*50)
    
    # Create test trajectories
    t = np.linspace(0, 2*np.pi, 100)
    
    # Smooth circular trajectory
    circle = np.column_stack([
        np.cos(t),
        np.sin(t),
        np.zeros_like(t)
    ])
    
    # Noisy trajectory
    noisy = circle.copy()
    noisy[:, :2] += np.random.normal(0, 0.05, (100, 2))
    
    # Create metrics evaluator
    evaluator = TrajectoryMetrics()
    
    # Evaluate single trajectory
    print("\n1. Evaluating smooth circle:")
    smooth_metrics = evaluator.evaluate_trajectory(circle, word="circle")
    print(f"   Smoothness: {smooth_metrics['smoothness_score']:.3f}")
    print(f"   Length: {smooth_metrics['trajectory_length']:.3f}")
    print(f"   Jerk mean: {smooth_metrics['jerk_mean']:.1f}")
    
    print("\n2. Evaluating noisy trajectory:")
    noisy_metrics = evaluator.evaluate_trajectory(noisy, reference=circle, word="noisy")
    print(f"   Smoothness: {noisy_metrics['smoothness_score']:.3f}")
    print(f"   DTW distance: {noisy_metrics['dtw_distance']:.3f}")
    print(f"   Fréchet distance: {noisy_metrics['frechet_distance']:.3f}")
    
    # Generate report
    print("\n3. Generating evaluation report...")
    report = evaluator.generate_report("test_metrics_report.md")
    
    # Save metrics
    evaluator.save_metrics("test_metrics.json")
    
    print("\n✅ Metrics system test complete!")
    
    return evaluator


if __name__ == "__main__":
    evaluator = test_metrics()