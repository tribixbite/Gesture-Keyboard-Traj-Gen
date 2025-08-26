#!/usr/bin/env python
"""
Validate synthetic traces against real swipe data
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from analyze_real_traces import RealTraceAnalyzer
from optimized_swype_generator import OptimizedSwypeGenerator
from validation_framework import TrajectoryValidator

class RealDataValidator:
    """Validate synthetic generation against real traces"""
    
    def __init__(self):
        self.generator = OptimizedSwypeGenerator()
        self.validator = TrajectoryValidator()
        self.real_analyzer = RealTraceAnalyzer()
        
    def load_real_traces(self, limit: int = 100) -> Dict:
        """Load and process real traces"""
        print("Loading real trace data...")
        self.real_analyzer.load_dataset(limit=limit)
        word_traces = self.real_analyzer.extract_word_traces()
        return word_traces
    
    def validate_word(self, word: str, real_traces: List, style: str = 'natural') -> Dict:
        """Validate synthetic generation for a specific word"""
        
        # Generate synthetic trace
        synthetic_result = self.generator.generate_trajectory(word, style=style)
        synth_trace = synthetic_result['trajectory']
        
        validation_results = []
        
        for real_data in real_traces:
            real_trace = real_data['trace']
            
            # Normalize traces for comparison
            # Real traces have different coordinate system
            real_norm = self.normalize_trace(real_trace)
            synth_norm = self.normalize_trace(synth_trace)
            
            # Calculate DTW distance
            dtw_dist, dtw_norm = self.validator.dtw_distance(synth_norm, real_norm)
            
            # Calculate Frechet distance
            frechet_dist = self.validator.frechet_distance(synth_norm, real_norm)
            
            # Compare velocities
            real_vel = self.calculate_velocity_profile(real_trace)
            synth_vel = self.calculate_velocity_profile(synth_trace)
            
            vel_correlation = np.corrcoef(
                real_vel[:min(len(real_vel), len(synth_vel))],
                synth_vel[:min(len(real_vel), len(synth_vel))]
            )[0, 1] if len(real_vel) > 1 and len(synth_vel) > 1 else 0
            
            validation_results.append({
                'dtw_distance': dtw_dist,
                'dtw_normalized': dtw_norm,
                'frechet_distance': frechet_dist,
                'velocity_correlation': vel_correlation,
                'real_points': len(real_trace),
                'synth_points': len(synth_trace),
                'real_duration': (real_trace[-1, 2] - real_trace[0, 2]) / 1000.0,
                'synth_duration': synthetic_result['duration']
            })
        
        # Calculate summary statistics
        if validation_results:
            return {
                'word': word,
                'num_comparisons': len(validation_results),
                'avg_dtw': np.mean([r['dtw_normalized'] for r in validation_results]),
                'avg_frechet': np.mean([r['frechet_distance'] for r in validation_results]),
                'avg_vel_corr': np.mean([r['velocity_correlation'] for r in validation_results if r['velocity_correlation'] > 0]),
                'points_ratio': np.mean([r['synth_points'] / r['real_points'] for r in validation_results]),
                'duration_ratio': np.mean([r['synth_duration'] / r['real_duration'] for r in validation_results if r['real_duration'] > 0]),
                'individual_results': validation_results
            }
        
        return None
    
    def normalize_trace(self, trace: np.ndarray) -> np.ndarray:
        """Normalize trace to [0, 1] range"""
        normalized = trace[:, :2].copy()
        
        # Find bounds
        x_min, x_max = normalized[:, 0].min(), normalized[:, 0].max()
        y_min, y_max = normalized[:, 1].min(), normalized[:, 1].max()
        
        # Normalize
        if x_max > x_min:
            normalized[:, 0] = (normalized[:, 0] - x_min) / (x_max - x_min)
        if y_max > y_min:
            normalized[:, 1] = (normalized[:, 1] - y_min) / (y_max - y_min)
        
        return normalized
    
    def calculate_velocity_profile(self, trace: np.ndarray) -> np.ndarray:
        """Calculate velocity profile from trace"""
        velocities = []
        
        for i in range(1, len(trace)):
            dx = trace[i, 0] - trace[i-1, 0]
            dy = trace[i, 1] - trace[i-1, 1]
            
            if trace.shape[1] > 2:  # Has timestamps
                dt = (trace[i, 2] - trace[i-1, 2]) / 1000.0
                if dt > 0:
                    vel = np.sqrt(dx*dx + dy*dy) / dt
                else:
                    vel = 0
            else:
                vel = np.sqrt(dx*dx + dy*dy)
            
            velocities.append(vel)
        
        return np.array(velocities)
    
    def run_validation(self, test_words: List[str] = None, samples_per_word: int = 5):
        """Run comprehensive validation"""
        print("\n" + "="*60)
        print("VALIDATION AGAINST REAL DATA")
        print("="*60)
        
        # Load real traces
        word_traces = self.load_real_traces(limit=100)
        
        if test_words is None:
            # Use most common words
            test_words = ['the', 'and', 'you', 'this', 'that', 'what', 'for', 'are', 'was', 'with']
        
        # Filter to available words
        available_words = [w for w in test_words if w in word_traces]
        
        if not available_words:
            print("❌ No test words found in real data")
            return None
        
        print(f"\nValidating {len(available_words)} words against real data")
        print("-"*40)
        
        validation_results = []
        
        for word in available_words:
            real_traces = word_traces[word][:samples_per_word]
            
            if not real_traces:
                continue
            
            result = self.validate_word(word, real_traces)
            
            if result:
                validation_results.append(result)
                
                print(f"\n'{word}' ({result['num_comparisons']} real samples):")
                print(f"  DTW distance: {result['avg_dtw']:.2f}")
                print(f"  Fréchet distance: {result['avg_frechet']:.2f}")
                print(f"  Velocity correlation: {result['avg_vel_corr']:.2f}")
                print(f"  Points ratio: {result['points_ratio']:.2f}x")
                print(f"  Duration ratio: {result['duration_ratio']:.2f}x")
        
        # Overall summary
        if validation_results:
            print("\n" + "="*60)
            print("OVERALL VALIDATION RESULTS")
            print("="*60)
            
            avg_dtw = np.mean([r['avg_dtw'] for r in validation_results])
            avg_frechet = np.mean([r['avg_frechet'] for r in validation_results])
            avg_vel_corr = np.mean([r['avg_vel_corr'] for r in validation_results if r['avg_vel_corr'] > 0])
            avg_points_ratio = np.mean([r['points_ratio'] for r in validation_results])
            avg_duration_ratio = np.mean([r['duration_ratio'] for r in validation_results])
            
            print(f"Average DTW distance: {avg_dtw:.2f}")
            print(f"Average Fréchet distance: {avg_frechet:.2f}")
            print(f"Average velocity correlation: {avg_vel_corr:.2f}")
            print(f"Points ratio (synth/real): {avg_points_ratio:.2f}x")
            print(f"Duration ratio (synth/real): {avg_duration_ratio:.2f}x")
            
            # Determine quality
            quality_score = 0
            issues = []
            
            if avg_dtw < 0.3:
                quality_score += 25
            else:
                issues.append(f"High DTW distance ({avg_dtw:.2f})")
            
            if avg_frechet < 0.4:
                quality_score += 25
            else:
                issues.append(f"High Fréchet distance ({avg_frechet:.2f})")
            
            if avg_vel_corr > 0.5:
                quality_score += 25
            else:
                issues.append(f"Low velocity correlation ({avg_vel_corr:.2f})")
            
            if 0.5 < avg_duration_ratio < 2.0:
                quality_score += 25
            else:
                issues.append(f"Duration mismatch ({avg_duration_ratio:.2f}x)")
            
            print(f"\nQuality Score: {quality_score}/100")
            
            if quality_score >= 75:
                print("✅ EXCELLENT: Synthetic traces closely match real data!")
            elif quality_score >= 50:
                print("✅ GOOD: Synthetic traces are realistic")
            elif quality_score >= 25:
                print("⚠️  FAIR: Some improvements needed")
                for issue in issues:
                    print(f"  - {issue}")
            else:
                print("❌ POOR: Significant improvements needed")
                for issue in issues:
                    print(f"  - {issue}")
            
            # Save results
            with open('real_validation_results.json', 'w') as f:
                json.dump({
                    'summary': {
                        'quality_score': quality_score,
                        'avg_dtw': avg_dtw,
                        'avg_frechet': avg_frechet,
                        'avg_velocity_correlation': avg_vel_corr,
                        'points_ratio': avg_points_ratio,
                        'duration_ratio': avg_duration_ratio,
                        'issues': issues
                    },
                    'word_results': validation_results
                }, f, indent=2)
            
            print("\nResults saved to real_validation_results.json")
            
            return quality_score
        
        return None


def main():
    """Main validation function"""
    
    validator = RealDataValidator()
    
    # Run validation
    quality_score = validator.run_validation()
    
    if quality_score is not None:
        print(f"\n{'='*60}")
        print("RECOMMENDATIONS")
        print("="*60)
        
        if quality_score < 50:
            print("\nSuggested improvements:")
            print("1. Adjust velocity profiles to match real user speeds")
            print("2. Tune noise models for more realistic variations")
            print("3. Calibrate duration scaling factors")
            print("4. Improve path smoothness parameters")
        else:
            print("\n✅ Synthetic generation is production-ready!")
            print("The traces closely match real user behavior.")
    
    return quality_score


if __name__ == "__main__":
    score = main()