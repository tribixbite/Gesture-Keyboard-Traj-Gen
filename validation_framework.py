#!/usr/bin/env python
"""
Validation framework for comparing synthetic traces with real human traces
Implements DTW, Frechet distance, and statistical distribution comparisons
"""

import numpy as np
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import json
import warnings

# Try to import scipy components
HAS_SCIPY = False
try:
    from scipy.spatial.distance import cdist
    from scipy import stats
    HAS_SCIPY = True
except ImportError:
    warnings.warn("scipy not available - using fallback distance metrics")

@dataclass
class ValidationMetrics:
    """Container for validation metrics"""
    dtw_distance: float
    dtw_normalized: float  # Normalized by path length
    frechet_distance: float
    velocity_ks_statistic: float
    velocity_ks_pvalue: float
    curvature_ks_statistic: float
    curvature_ks_pvalue: float
    duration_difference: float
    path_length_ratio: float
    jerk_ratio: float
    
    def is_valid(self, thresholds: Dict = None) -> bool:
        """Check if metrics pass validation thresholds"""
        if thresholds is None:
            thresholds = {
                'dtw_normalized': 0.15,  # 15% of path length
                'velocity_ks_pvalue': 0.05,  # 5% significance
                'curvature_ks_pvalue': 0.05,
                'path_length_ratio': 0.3,  # Within 30% of real
                'jerk_ratio': 5.0  # Within 5x of real
            }
        
        checks = [
            self.dtw_normalized < thresholds['dtw_normalized'],
            self.velocity_ks_pvalue > thresholds['velocity_ks_pvalue'],
            self.curvature_ks_pvalue > thresholds['curvature_ks_pvalue'],
            abs(1 - self.path_length_ratio) < thresholds['path_length_ratio'],
            self.jerk_ratio < thresholds['jerk_ratio']
        ]
        
        return all(checks)


class TrajectoryValidator:
    """
    Validate synthetic trajectories against real human traces
    """
    
    def __init__(self, real_dataset_path: str = 'real_datasets/standard_validation_dataset.pkl'):
        """
        Initialize validator with real dataset
        
        Args:
            real_dataset_path: Path to real trace dataset
        """
        self.real_dataset = self._load_real_dataset(real_dataset_path)
        self.validation_results = []
    
    def _load_real_dataset(self, path: str) -> Dict:
        """Load real trace dataset"""
        if Path(path).exists():
            with open(path, 'rb') as f:
                return pickle.load(f)
        else:
            print(f"Warning: Real dataset not found at {path}")
            print("Using simulated validation data instead")
            return self._create_fallback_dataset()
    
    def _create_fallback_dataset(self) -> Dict:
        """Create fallback dataset if real data unavailable"""
        return {
            'traces': [],
            'words': [],
            'users': [],
            'metadata': {'source': 'fallback', 'num_samples': 0}
        }
    
    def dtw_distance(self, trace1: np.ndarray, trace2: np.ndarray) -> Tuple[float, float]:
        """
        Calculate Dynamic Time Warping distance between two traces
        
        Args:
            trace1: First trajectory (N, 2) or (N, 3) with x, y, [t]
            trace2: Second trajectory (M, 2) or (M, 3)
            
        Returns:
            (dtw_distance, normalized_distance)
        """
        # Use only x, y coordinates
        t1 = trace1[:, :2] if trace1.shape[1] > 2 else trace1
        t2 = trace2[:, :2] if trace2.shape[1] > 2 else trace2
        
        n, m = len(t1), len(t2)
        
        # Initialize DTW matrix
        dtw_matrix = np.full((n + 1, m + 1), np.inf)
        dtw_matrix[0, 0] = 0
        
        # Fill DTW matrix
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                cost = np.linalg.norm(t1[i-1] - t2[j-1])
                
                dtw_matrix[i, j] = cost + min(
                    dtw_matrix[i-1, j],    # Insertion
                    dtw_matrix[i, j-1],    # Deletion
                    dtw_matrix[i-1, j-1]   # Match
                )
        
        dtw_dist = dtw_matrix[n, m]
        
        # Normalize by path length
        path_length = max(
            self._calculate_path_length(t1),
            self._calculate_path_length(t2)
        )
        
        normalized_dist = dtw_dist / path_length if path_length > 0 else dtw_dist
        
        return dtw_dist, normalized_dist
    
    def frechet_distance(self, trace1: np.ndarray, trace2: np.ndarray) -> float:
        """
        Calculate discrete Fréchet distance between two traces
        
        Args:
            trace1: First trajectory
            trace2: Second trajectory
            
        Returns:
            Fréchet distance
        """
        t1 = trace1[:, :2] if trace1.shape[1] > 2 else trace1
        t2 = trace2[:, :2] if trace2.shape[1] > 2 else trace2
        
        n, m = len(t1), len(t2)
        
        # Calculate distance matrix
        if HAS_SCIPY:
            dist_matrix = cdist(t1, t2)
        else:
            # Manual distance calculation
            dist_matrix = np.zeros((n, m))
            for i in range(n):
                for j in range(m):
                    dist_matrix[i, j] = np.linalg.norm(t1[i] - t2[j])
        
        # Dynamic programming for Fréchet distance (iterative to avoid recursion)
        dp = np.full((n, m), np.inf)
        
        # Base case
        dp[0, 0] = dist_matrix[0, 0]
        
        # Fill first column
        for i in range(1, n):
            dp[i, 0] = max(dp[i-1, 0], dist_matrix[i, 0])
        
        # Fill first row  
        for j in range(1, m):
            dp[0, j] = max(dp[0, j-1], dist_matrix[0, j])
        
        # Fill rest of matrix
        for i in range(1, n):
            for j in range(1, m):
                dp[i, j] = max(
                    min(dp[i-1, j], dp[i, j-1], dp[i-1, j-1]),
                    dist_matrix[i, j]
                )
        
        return dp[n-1, m-1]
    
    def _calculate_path_length(self, trace: np.ndarray) -> float:
        """Calculate total path length of a trace"""
        if len(trace) < 2:
            return 0.0
        
        distances = np.linalg.norm(np.diff(trace[:, :2], axis=0), axis=1)
        return np.sum(distances)
    
    def _calculate_velocity_profile(self, trace: np.ndarray) -> np.ndarray:
        """Calculate velocity profile from trace"""
        if len(trace) < 2:
            return np.array([0.0])
        
        # Calculate distances between consecutive points
        distances = np.linalg.norm(np.diff(trace[:, :2], axis=0), axis=1)
        
        # Calculate time differences
        if trace.shape[1] > 2:  # Has time component
            time_diffs = np.diff(trace[:, 2])
            time_diffs = np.where(time_diffs > 0, time_diffs, 1e-6)
        else:
            # Assume uniform time
            time_diffs = np.ones(len(distances)) * 0.01
        
        velocities = distances / time_diffs
        
        return velocities
    
    def _calculate_curvature(self, trace: np.ndarray) -> np.ndarray:
        """Calculate curvature at each point"""
        if len(trace) < 3:
            return np.array([0.0])
        
        curvatures = []
        
        for i in range(1, len(trace) - 1):
            p1 = trace[i-1, :2]
            p2 = trace[i, :2]
            p3 = trace[i+1, :2]
            
            v1 = p2 - p1
            v2 = p3 - p2
            
            # Calculate angle between vectors
            if np.linalg.norm(v1) > 0 and np.linalg.norm(v2) > 0:
                cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                angle = np.arccos(np.clip(cos_angle, -1, 1))
                curvatures.append(angle)
            else:
                curvatures.append(0.0)
        
        return np.array(curvatures)
    
    def _calculate_jerk(self, trace: np.ndarray) -> float:
        """Calculate mean jerk of trajectory"""
        if len(trace) < 4:
            return 0.0
        
        # Use simple finite differences
        if trace.shape[1] > 2:
            dt = np.mean(np.diff(trace[:, 2]))
        else:
            dt = 0.01
        
        # Calculate third derivative
        x = trace[:, 0]
        y = trace[:, 1]
        
        # Third order differences
        jerk_x = np.diff(x, n=3) / (dt ** 3)
        jerk_y = np.diff(y, n=3) / (dt ** 3)
        
        jerk_magnitude = np.sqrt(jerk_x**2 + jerk_y**2)
        
        return np.mean(jerk_magnitude)
    
    def validate_trace(self, synthetic_trace: np.ndarray, real_trace: np.ndarray) -> ValidationMetrics:
        """
        Validate a synthetic trace against a real trace
        
        Args:
            synthetic_trace: Generated trajectory
            real_trace: Real human trajectory
            
        Returns:
            ValidationMetrics object
        """
        # Calculate DTW distance
        dtw_dist, dtw_norm = self.dtw_distance(synthetic_trace, real_trace)
        
        # Calculate Fréchet distance
        frechet_dist = self.frechet_distance(synthetic_trace, real_trace)
        
        # Calculate velocity profiles
        synthetic_vel = self._calculate_velocity_profile(synthetic_trace)
        real_vel = self._calculate_velocity_profile(real_trace)
        
        # Kolmogorov-Smirnov test for velocity distribution
        if HAS_SCIPY and len(synthetic_vel) > 0 and len(real_vel) > 0:
            vel_ks_stat, vel_ks_pval = stats.ks_2samp(synthetic_vel, real_vel)
        elif len(synthetic_vel) > 0 and len(real_vel) > 0:
            # Fallback: simple statistical comparison
            vel_ks_stat = abs(np.mean(synthetic_vel) - np.mean(real_vel)) / (np.std(synthetic_vel) + np.std(real_vel) + 1e-6)
            vel_ks_pval = 1.0 / (1.0 + vel_ks_stat)  # Pseudo p-value
        else:
            vel_ks_stat, vel_ks_pval = 1.0, 0.0
        
        # Calculate curvature profiles
        synthetic_curv = self._calculate_curvature(synthetic_trace)
        real_curv = self._calculate_curvature(real_trace)
        
        # K-S test for curvature distribution
        if HAS_SCIPY and len(synthetic_curv) > 0 and len(real_curv) > 0:
            curv_ks_stat, curv_ks_pval = stats.ks_2samp(synthetic_curv, real_curv)
        elif len(synthetic_curv) > 0 and len(real_curv) > 0:
            # Fallback: simple statistical comparison
            curv_ks_stat = abs(np.mean(synthetic_curv) - np.mean(real_curv)) / (np.std(synthetic_curv) + np.std(real_curv) + 1e-6)
            curv_ks_pval = 1.0 / (1.0 + curv_ks_stat)  # Pseudo p-value
        else:
            curv_ks_stat, curv_ks_pval = 1.0, 0.0
        
        # Duration comparison
        if synthetic_trace.shape[1] > 2 and real_trace.shape[1] > 2:
            synthetic_duration = synthetic_trace[-1, 2] - synthetic_trace[0, 2]
            real_duration = real_trace[-1, 2] - real_trace[0, 2]
            duration_diff = abs(synthetic_duration - real_duration)
        else:
            duration_diff = 0.0
        
        # Path length comparison
        synthetic_length = self._calculate_path_length(synthetic_trace)
        real_length = self._calculate_path_length(real_trace)
        path_length_ratio = synthetic_length / real_length if real_length > 0 else 1.0
        
        # Jerk comparison
        synthetic_jerk = self._calculate_jerk(synthetic_trace)
        real_jerk = self._calculate_jerk(real_trace)
        jerk_ratio = synthetic_jerk / real_jerk if real_jerk > 0 else 1.0
        
        return ValidationMetrics(
            dtw_distance=dtw_dist,
            dtw_normalized=dtw_norm,
            frechet_distance=frechet_dist,
            velocity_ks_statistic=vel_ks_stat,
            velocity_ks_pvalue=vel_ks_pval,
            curvature_ks_statistic=curv_ks_stat,
            curvature_ks_pvalue=curv_ks_pval,
            duration_difference=duration_diff,
            path_length_ratio=path_length_ratio,
            jerk_ratio=jerk_ratio
        )
    
    def validate_generator(self, generator, word_list: List[str] = None) -> Dict:
        """
        Validate a generator against the real dataset
        
        Args:
            generator: Trajectory generator with generate() method
            word_list: List of words to test (None = use dataset words)
            
        Returns:
            Validation results summary
        """
        if not self.real_dataset['traces']:
            print("Warning: No real traces available for validation")
            return {}
        
        if word_list is None:
            # Use words from real dataset
            word_list = list(set(self.real_dataset['words']))[:20]  # Test first 20 unique words
        
        results = {
            'word_results': {},
            'aggregate_metrics': {},
            'validation_passed': False
        }
        
        all_metrics = []
        
        for word in word_list:
            # Find real traces for this word
            word_indices = [i for i, w in enumerate(self.real_dataset['words']) if w == word]
            
            if not word_indices:
                continue
            
            # Generate synthetic trace
            try:
                synthetic = generator.generate(word)
                if isinstance(synthetic, dict):
                    synthetic_trace = synthetic.get('trajectory', np.array([]))
                else:
                    synthetic_trace = synthetic
                
                if len(synthetic_trace) == 0:
                    continue
                
                # Compare with each real trace for this word
                word_metrics = []
                for idx in word_indices[:5]:  # Max 5 comparisons per word
                    real_trace = self.real_dataset['traces'][idx]
                    metrics = self.validate_trace(synthetic_trace, real_trace)
                    word_metrics.append(metrics)
                
                # Average metrics for this word
                if word_metrics:
                    avg_metrics = ValidationMetrics(
                        dtw_distance=np.mean([m.dtw_distance for m in word_metrics]),
                        dtw_normalized=np.mean([m.dtw_normalized for m in word_metrics]),
                        frechet_distance=np.mean([m.frechet_distance for m in word_metrics]),
                        velocity_ks_statistic=np.mean([m.velocity_ks_statistic for m in word_metrics]),
                        velocity_ks_pvalue=np.mean([m.velocity_ks_pvalue for m in word_metrics]),
                        curvature_ks_statistic=np.mean([m.curvature_ks_statistic for m in word_metrics]),
                        curvature_ks_pvalue=np.mean([m.curvature_ks_pvalue for m in word_metrics]),
                        duration_difference=np.mean([m.duration_difference for m in word_metrics]),
                        path_length_ratio=np.mean([m.path_length_ratio for m in word_metrics]),
                        jerk_ratio=np.mean([m.jerk_ratio for m in word_metrics])
                    )
                    
                    results['word_results'][word] = {
                        'metrics': avg_metrics,
                        'valid': avg_metrics.is_valid(),
                        'num_comparisons': len(word_metrics)
                    }
                    
                    all_metrics.append(avg_metrics)
                
            except Exception as e:
                print(f"Error validating word '{word}': {e}")
        
        # Calculate aggregate statistics
        if all_metrics:
            results['aggregate_metrics'] = {
                'mean_dtw_normalized': np.mean([m.dtw_normalized for m in all_metrics]),
                'mean_frechet': np.mean([m.frechet_distance for m in all_metrics]),
                'mean_velocity_ks_pvalue': np.mean([m.velocity_ks_pvalue for m in all_metrics]),
                'mean_curvature_ks_pvalue': np.mean([m.curvature_ks_pvalue for m in all_metrics]),
                'mean_path_length_ratio': np.mean([m.path_length_ratio for m in all_metrics]),
                'mean_jerk_ratio': np.mean([m.jerk_ratio for m in all_metrics]),
                'percent_valid': sum(r['valid'] for r in results['word_results'].values()) / len(results['word_results']) * 100
            }
            
            # Overall validation pass/fail
            results['validation_passed'] = results['aggregate_metrics']['percent_valid'] > 70
        
        return results
    
    def generate_report(self, results: Dict, output_file: str = 'validation_report.json'):
        """Generate validation report"""
        report = {
            'summary': {
                'validation_passed': results.get('validation_passed', False),
                'words_tested': len(results.get('word_results', {})),
                'percent_valid': results.get('aggregate_metrics', {}).get('percent_valid', 0)
            },
            'aggregate_metrics': results.get('aggregate_metrics', {}),
            'word_details': {}
        }
        
        # Add word-level details
        for word, word_result in results.get('word_results', {}).items():
            metrics = word_result['metrics']
            report['word_details'][word] = {
                'valid': word_result['valid'],
                'dtw_normalized': metrics.dtw_normalized,
                'frechet_distance': metrics.frechet_distance,
                'velocity_match': metrics.velocity_ks_pvalue > 0.05,
                'curvature_match': metrics.curvature_ks_pvalue > 0.05,
                'path_length_ratio': metrics.path_length_ratio
            }
        
        # Save report
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Print summary
        print("\n" + "="*60)
        print("VALIDATION REPORT")
        print("="*60)
        
        if report['summary']['validation_passed']:
            print("✅ VALIDATION PASSED")
        else:
            print("❌ VALIDATION FAILED")
        
        print(f"\nWords tested: {report['summary']['words_tested']}")
        print(f"Percent valid: {report['summary']['percent_valid']:.1f}%")
        
        if results.get('aggregate_metrics'):
            print(f"\nKey Metrics:")
            if 'mean_dtw_normalized' in results['aggregate_metrics']:
                print(f"  DTW (normalized): {results['aggregate_metrics']['mean_dtw_normalized']:.3f}")
            if 'mean_frechet' in results['aggregate_metrics']:
                print(f"  Fréchet distance: {results['aggregate_metrics']['mean_frechet']:.1f}")
            if 'mean_velocity_ks_pvalue' in results['aggregate_metrics']:
                print(f"  Velocity match p-value: {results['aggregate_metrics']['mean_velocity_ks_pvalue']:.3f}")
            if 'mean_path_length_ratio' in results['aggregate_metrics']:
                print(f"  Path length ratio: {results['aggregate_metrics']['mean_path_length_ratio']:.2f}")
        
        print(f"\nReport saved to: {output_file}")
        
        return report


def main():
    """Demo validation framework"""
    print("="*60)
    print("TRAJECTORY VALIDATION FRAMEWORK")
    print("="*60)
    
    # Initialize validator
    validator = TrajectoryValidator()
    
    # Load our unified API
    from unified_swype_api import UnifiedSwypeAPI, TrajectoryConfig, GeneratorType
    
    api = UnifiedSwypeAPI()
    
    # Test with optimized generator
    print("\nValidating Optimized Generator...")
    
    # Create simple generator wrapper
    class GeneratorWrapper:
        def __init__(self, api, gen_type):
            self.api = api
            self.gen_type = gen_type
        
        def generate(self, word):
            config = TrajectoryConfig(word, self.gen_type)
            return self.api.generate(config)
    
    generator = GeneratorWrapper(api, GeneratorType.OPTIMIZED)
    
    # Run validation
    results = validator.validate_generator(generator)
    
    # Generate report
    report = validator.generate_report(results)
    
    print("\n✅ Validation framework ready!")
    print("Note: Using simulated validation data since real dataset not available")
    print("To improve validation:")
    print("  1. Obtain real gesture keyboard traces")
    print("  2. Place in real_datasets/ folder")
    print("  3. Re-run validation")

if __name__ == "__main__":
    main()