#!/usr/bin/env python
"""
Comprehensive benchmarking script for all trajectory generation models
Compares Attention RNN, WGAN-GP, and Transformer models
"""

import torch
import numpy as np
from pathlib import Path
import json
import time
from datetime import datetime
import pandas as pd
from typing import Dict, List, Tuple, Optional

# Import models
from pytorch_attention_rnn import AttentionRNN
from pytorch_wgan_gp import Generator, Discriminator
from pytorch_transformer import TransformerTrajectoryModel

# Import evaluation metrics
from evaluation_metrics import TrajectoryMetrics

# Import dataset
from train_all_models import UnifiedDataset


class ModelBenchmark:
    """Benchmark all trained models"""
    
    def __init__(self, device: str = 'cpu'):
        self.device = torch.device(device)
        self.metrics_evaluator = TrajectoryMetrics()
        self.results = {}
        
    def load_models(self) -> Dict:
        """Load all trained models from checkpoints"""
        models = {}
        checkpoint_dir = Path("checkpoints")
        
        print("Loading trained models...")
        print("="*50)
        
        # Load Attention RNN
        rnn_path = checkpoint_dir / "attention_rnn_best.pt"
        if rnn_path.exists():
            try:
                # Create dummy model to get vocab size
                dummy_model = AttentionRNN(vocab_size=50)
                checkpoint = torch.load(rnn_path, map_location=self.device, weights_only=False)
                dummy_model.load_state_dict(checkpoint['model_state'])
                dummy_model.eval()
                models['attention_rnn'] = dummy_model
                print(f"✅ Loaded Attention RNN (epoch {checkpoint.get('epoch', 'unknown')})")
            except Exception as e:
                print(f"❌ Failed to load Attention RNN: {e}")
        
        # Load WGAN-GP Generator
        wgan_files = list(checkpoint_dir.glob("wgan_gp_epoch_*.pt"))
        if wgan_files:
            try:
                latest = max(wgan_files, key=lambda x: int(x.stem.split('_')[-1]))
                checkpoint = torch.load(latest, map_location=self.device, weights_only=False)
                generator = Generator(vocab_size=50)
                generator.load_state_dict(checkpoint['generator_state'])
                generator.eval()
                models['wgan_gp'] = generator
                print(f"✅ Loaded WGAN-GP Generator (epoch {checkpoint['epoch']})")
            except Exception as e:
                print(f"❌ Failed to load WGAN-GP: {e}")
        
        # Load Transformer
        transformer_path = checkpoint_dir / "transformer_best.pt"
        if transformer_path.exists():
            try:
                transformer = TransformerTrajectoryModel(vocab_size=50)
                checkpoint = torch.load(transformer_path, map_location=self.device, weights_only=False)
                transformer.load_state_dict(checkpoint['model_state'])
                transformer.eval()
                models['transformer'] = transformer
                print(f"✅ Loaded Transformer (epoch {checkpoint.get('epoch', 'unknown')})")
            except Exception as e:
                print(f"❌ Failed to load Transformer: {e}")
        
        if not models:
            # Create untrained models for testing
            print("\n⚠️ No trained models found, creating untrained models for testing...")
            models['attention_rnn'] = AttentionRNN(vocab_size=50).to(self.device)
            models['wgan_gp'] = Generator(vocab_size=50).to(self.device)
            models['transformer'] = TransformerTrajectoryModel(vocab_size=50).to(self.device)
        
        print(f"\nLoaded {len(models)} models: {list(models.keys())}")
        return models
    
    def generate_trajectories(self, 
                            models: Dict,
                            words: torch.Tensor,
                            num_samples: int = 1) -> Dict[str, List[np.ndarray]]:
        """Generate trajectories from all models"""
        
        generated = {}
        
        print("\nGenerating trajectories...")
        print("="*50)
        
        for model_name, model in models.items():
            print(f"\nGenerating with {model_name}...")
            trajectories = []
            
            with torch.no_grad():
                for i in range(min(num_samples, words.size(0))):
                    word = words[i:i+1].to(self.device)
                    
                    try:
                        if model_name == 'attention_rnn':
                            traj = model.generate(word, max_length=150, temperature=0.8)
                        elif model_name == 'wgan_gp':
                            model.eval()  # Set to eval mode to avoid batch norm issues
                            noise = torch.randn(1, model.latent_dim, device=self.device)
                            traj = model(noise, word)
                        elif model_name == 'transformer':
                            traj = model.generate(word, max_length=150, temperature=0.8)
                        
                        # Convert to numpy
                        traj_np = traj.cpu().numpy()[0]
                        
                        # Find end of stroke
                        if traj_np.shape[1] > 2:
                            end_idx = np.where(traj_np[:, 2] > 0.5)[0]
                            if len(end_idx) > 0:
                                traj_np = traj_np[:end_idx[0] + 1]
                        
                        trajectories.append(traj_np)
                        print(f"   Sample {i+1}: {len(traj_np)} points")
                        
                    except Exception as e:
                        print(f"   ❌ Generation failed for sample {i+1}: {e}")
                        # Add empty trajectory
                        trajectories.append(np.zeros((2, 3)))
            
            generated[model_name] = trajectories
        
        return generated
    
    def benchmark_quality(self, 
                         generated: Dict[str, List[np.ndarray]],
                         reference: Optional[List[np.ndarray]] = None) -> Dict:
        """Benchmark trajectory quality for all models"""
        
        print("\nEvaluating trajectory quality...")
        print("="*50)
        
        quality_results = {}
        
        for model_name, trajectories in generated.items():
            print(f"\nEvaluating {model_name}...")
            
            model_metrics = []
            for i, traj in enumerate(trajectories):
                if len(traj) > 2:  # Valid trajectory
                    ref = reference[i] if reference and i < len(reference) else None
                    metrics = self.metrics_evaluator.evaluate_trajectory(
                        traj, 
                        reference=ref,
                        word=f"test_{i}"
                    )
                    model_metrics.append(metrics)
            
            if model_metrics:
                # Aggregate metrics
                aggregate = {}
                numeric_keys = [k for k in model_metrics[0].keys() 
                              if isinstance(model_metrics[0][k], (int, float))]
                
                for key in numeric_keys:
                    values = [m[key] for m in model_metrics]
                    aggregate[key] = {
                        'mean': float(np.mean(values)),
                        'std': float(np.std(values)),
                        'min': float(np.min(values)),
                        'max': float(np.max(values))
                    }
                
                quality_results[model_name] = aggregate
                
                # Print key metrics
                print(f"   Smoothness: {aggregate.get('smoothness_score', {}).get('mean', 0):.3f}")
                print(f"   Avg length: {aggregate.get('trajectory_length', {}).get('mean', 0):.1f}")
                print(f"   Avg points: {aggregate.get('trajectory_points', {}).get('mean', 0):.1f}")
        
        return quality_results
    
    def benchmark_speed(self, models: Dict, words: torch.Tensor, 
                       num_iterations: int = 10) -> Dict:
        """Benchmark generation speed for all models"""
        
        print("\nBenchmarking generation speed...")
        print("="*50)
        
        speed_results = {}
        
        for model_name, model in models.items():
            print(f"\n{model_name}:")
            
            word = words[:1].to(self.device)
            times = []
            
            # Warmup
            with torch.no_grad():
                if model_name == 'attention_rnn':
                    _ = model.generate(word, max_length=50)
                elif model_name == 'wgan_gp':
                    noise = torch.randn(1, model.latent_dim, device=self.device)
                    _ = model(noise, word)
                elif model_name == 'transformer':
                    _ = model.generate(word, max_length=50)
            
            # Benchmark
            for i in range(num_iterations):
                start_time = time.time()
                
                with torch.no_grad():
                    if model_name == 'attention_rnn':
                        _ = model.generate(word, max_length=150)
                    elif model_name == 'wgan_gp':
                        noise = torch.randn(1, model.latent_dim, device=self.device)
                        _ = model(noise, word)
                    elif model_name == 'transformer':
                        _ = model.generate(word, max_length=150)
                
                elapsed = time.time() - start_time
                times.append(elapsed)
            
            speed_results[model_name] = {
                'mean_time': float(np.mean(times)),
                'std_time': float(np.std(times)),
                'min_time': float(np.min(times)),
                'max_time': float(np.max(times)),
                'trajectories_per_second': float(1.0 / np.mean(times))
            }
            
            print(f"   Avg time: {speed_results[model_name]['mean_time']*1000:.2f}ms")
            print(f"   Speed: {speed_results[model_name]['trajectories_per_second']:.1f} traj/sec")
        
        return speed_results
    
    def benchmark_memory(self, models: Dict) -> Dict:
        """Benchmark model memory usage"""
        
        print("\nBenchmarking memory usage...")
        print("="*50)
        
        memory_results = {}
        
        for model_name, model in models.items():
            # Count parameters
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            # Estimate memory (4 bytes per float32 parameter)
            param_memory_mb = (total_params * 4) / (1024 * 1024)
            
            memory_results[model_name] = {
                'total_parameters': total_params,
                'trainable_parameters': trainable_params,
                'parameter_memory_mb': param_memory_mb
            }
            
            print(f"\n{model_name}:")
            print(f"   Total params: {total_params:,}")
            print(f"   Trainable params: {trainable_params:,}")
            print(f"   Memory: {param_memory_mb:.2f} MB")
        
        return memory_results
    
    def generate_comparison_report(self, output_path: str = "benchmark_report.md"):
        """Generate comprehensive comparison report"""
        
        print("\nGenerating benchmark report...")
        print("="*50)
        
        report = []
        report.append("# Model Benchmark Report")
        report.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"\nDevice: {self.device}")
        
        # Quality comparison
        if 'quality' in self.results:
            report.append("\n## Quality Metrics")
            report.append("\n### Smoothness Scores")
            report.append("| Model | Mean | Std | Min | Max |")
            report.append("|-------|------|-----|-----|-----|")
            
            for model, metrics in self.results['quality'].items():
                if 'smoothness_score' in metrics:
                    m = metrics['smoothness_score']
                    report.append(f"| {model} | {m['mean']:.3f} | {m['std']:.3f} | "
                                f"{m['min']:.3f} | {m['max']:.3f} |")
            
            report.append("\n### Trajectory Length")
            report.append("| Model | Mean | Std | Min | Max |")
            report.append("|-------|------|-----|-----|-----|")
            
            for model, metrics in self.results['quality'].items():
                if 'trajectory_length' in metrics:
                    m = metrics['trajectory_length']
                    report.append(f"| {model} | {m['mean']:.1f} | {m['std']:.1f} | "
                                f"{m['min']:.1f} | {m['max']:.1f} |")
        
        # Speed comparison
        if 'speed' in self.results:
            report.append("\n## Speed Performance")
            report.append("| Model | Avg Time (ms) | Std (ms) | Traj/sec |")
            report.append("|-------|---------------|----------|----------|")
            
            for model, metrics in self.results['speed'].items():
                report.append(f"| {model} | {metrics['mean_time']*1000:.2f} | "
                            f"{metrics['std_time']*1000:.2f} | "
                            f"{metrics['trajectories_per_second']:.1f} |")
        
        # Memory comparison
        if 'memory' in self.results:
            report.append("\n## Memory Usage")
            report.append("| Model | Total Params | Trainable | Memory (MB) |")
            report.append("|-------|--------------|-----------|-------------|")
            
            for model, metrics in self.results['memory'].items():
                report.append(f"| {model} | {metrics['total_parameters']:,} | "
                            f"{metrics['trainable_parameters']:,} | "
                            f"{metrics['parameter_memory_mb']:.2f} |")
        
        # Best model summary
        report.append("\n## Summary")
        
        if 'quality' in self.results and 'speed' in self.results:
            # Find best smoothness
            best_smooth = max(self.results['quality'].items(), 
                            key=lambda x: x[1].get('smoothness_score', {}).get('mean', 0))
            report.append(f"- **Best Smoothness**: {best_smooth[0]} "
                        f"({best_smooth[1]['smoothness_score']['mean']:.3f})")
            
            # Find fastest
            fastest = min(self.results['speed'].items(), 
                        key=lambda x: x[1]['mean_time'])
            report.append(f"- **Fastest Generation**: {fastest[0]} "
                        f"({fastest[1]['mean_time']*1000:.2f}ms)")
            
            # Find smallest
            if 'memory' in self.results:
                smallest = min(self.results['memory'].items(), 
                             key=lambda x: x[1]['total_parameters'])
                report.append(f"- **Smallest Model**: {smallest[0]} "
                            f"({smallest[1]['total_parameters']:,} parameters)")
        
        # Save report
        report_text = '\n'.join(report)
        with open(output_path, 'w') as f:
            f.write(report_text)
        
        print(f"✅ Saved benchmark report to {output_path}")
        
        # Also save raw results as JSON
        json_path = output_path.replace('.md', '.json')
        with open(json_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"✅ Saved raw results to {json_path}")
        
        return report_text
    
    def run_full_benchmark(self, num_samples: int = 10):
        """Run complete benchmark suite"""
        
        print("\n" + "="*60)
        print("FULL MODEL BENCHMARK")
        print("="*60)
        
        # Load models
        models = self.load_models()
        
        if not models:
            print("❌ No models to benchmark")
            return
        
        # Create test data
        print("\nPreparing test data...")
        test_words = torch.randint(1, 30, (num_samples, 10))  # Random words
        
        # Generate trajectories
        generated = self.generate_trajectories(models, test_words, num_samples)
        
        # Benchmark quality
        self.results['quality'] = self.benchmark_quality(generated)
        
        # Benchmark speed
        self.results['speed'] = self.benchmark_speed(models, test_words, num_iterations=10)
        
        # Benchmark memory
        self.results['memory'] = self.benchmark_memory(models)
        
        # Generate report
        report = self.generate_comparison_report()
        
        print("\n" + "="*60)
        print("✅ BENCHMARK COMPLETE")
        print("="*60)
        
        return self.results


def main():
    """Main benchmark function"""
    
    # Setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Create benchmark
    benchmark = ModelBenchmark(device)
    
    # Run full benchmark
    results = benchmark.run_full_benchmark(num_samples=5)
    
    # Print summary
    if results:
        print("\n📊 Benchmark Summary:")
        if 'quality' in results:
            for model in results['quality']:
                smooth = results['quality'][model].get('smoothness_score', {}).get('mean', 0)
                print(f"   {model}: smoothness={smooth:.3f}")
    
    return benchmark


if __name__ == "__main__":
    benchmark = main()