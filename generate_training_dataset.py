#!/usr/bin/env python
"""
Generate large-scale training dataset for GAN/RNN models
Using improved generator with realistic jerk values
"""

import numpy as np
import pandas as pd
import json
import pickle
import random
from pathlib import Path
from datetime import datetime
from improved_swype_generator import ImprovedSwypeGenerator
import time

# Expanded vocabulary for training
TRAINING_VOCABULARY = [
    # Top 100 most common English words
    'the', 'be', 'to', 'of', 'and', 'a', 'in', 'that', 'have', 'i',
    'it', 'for', 'not', 'on', 'with', 'he', 'as', 'you', 'do', 'at',
    'this', 'but', 'his', 'by', 'from', 'they', 'we', 'say', 'her', 'she',
    'or', 'an', 'will', 'my', 'one', 'all', 'would', 'there', 'their', 'what',
    'so', 'up', 'out', 'if', 'about', 'who', 'get', 'which', 'go', 'me',
    'when', 'make', 'can', 'like', 'time', 'no', 'just', 'him', 'know', 'take',
    'people', 'into', 'year', 'your', 'good', 'some', 'could', 'them', 'see', 'other',
    'than', 'then', 'now', 'look', 'only', 'come', 'its', 'over', 'think', 'also',
    'back', 'after', 'use', 'two', 'how', 'our', 'work', 'first', 'well', 'way',
    'even', 'new', 'want', 'because', 'any', 'these', 'give', 'day', 'most', 'us',
    
    # Common tech/computing words
    'computer', 'software', 'hardware', 'internet', 'network', 'database', 'system', 'program',
    'security', 'password', 'email', 'website', 'server', 'client', 'data', 'file',
    'code', 'debug', 'compile', 'execute', 'function', 'variable', 'algorithm', 'interface',
    
    # Common mobile/texting words
    'hello', 'thanks', 'please', 'sorry', 'okay', 'yes', 'maybe', 'later',
    'tomorrow', 'today', 'yesterday', 'morning', 'night', 'love', 'friend', 'call',
    'text', 'send', 'reply', 'message', 'phone', 'meeting', 'home', 'office',
    
    # Action words
    'run', 'walk', 'talk', 'eat', 'sleep', 'read', 'write', 'play',
    'watch', 'listen', 'learn', 'teach', 'buy', 'sell', 'help', 'wait',
    'start', 'stop', 'continue', 'finish', 'begin', 'end', 'open', 'close'
]

def create_balanced_dataset(generator, num_samples_per_config=10):
    """
    Create a balanced dataset with multiple samples per word/style combination
    
    Args:
        generator: ImprovedSwypeGenerator instance
        num_samples_per_config: Number of traces per word/style combination
    
    Returns:
        List of trace dictionaries
    """
    
    styles = ['precise', 'natural', 'fast', 'sloppy']
    dataset = []
    
    total_configs = len(TRAINING_VOCABULARY) * len(styles)
    total_samples = total_configs * num_samples_per_config
    
    print(f"Generating balanced dataset...")
    print(f"  Words: {len(TRAINING_VOCABULARY)}")
    print(f"  Styles: {len(styles)}")
    print(f"  Samples per config: {num_samples_per_config}")
    print(f"  Total samples: {total_samples}")
    print("-" * 60)
    
    start_time = time.time()
    sample_count = 0
    
    for word_idx, word in enumerate(TRAINING_VOCABULARY):
        for style in styles:
            for sample_idx in range(num_samples_per_config):
                # Vary parameters slightly for diversity
                user_speed = np.random.uniform(0.8, 1.2)
                precision = np.random.uniform(0.6, 0.9) if style != 'precise' else np.random.uniform(0.8, 0.95)
                
                try:
                    # Generate trace
                    trace = generator.generate_trajectory(
                        word=word,
                        style=style,
                        user_speed=user_speed,
                        precision=precision
                    )
                    
                    # Create training sample
                    sample = {
                        'id': f"{word}_{style}_{sample_idx}",
                        'word': word,
                        'style': style,
                        'user_speed': user_speed,
                        'precision': precision,
                        'trajectory': trace['trajectory'].tolist(),
                        'duration': trace['duration'],
                        'num_points': len(trace['trajectory']),
                        'mean_jerk': float(trace['metadata']['mean_jerk']),
                        'max_jerk': float(trace['metadata']['max_jerk']),
                        'timestamp': datetime.now().isoformat()
                    }
                    
                    dataset.append(sample)
                    sample_count += 1
                    
                    # Progress update
                    if sample_count % 100 == 0:
                        elapsed = time.time() - start_time
                        rate = sample_count / elapsed
                        remaining = (total_samples - sample_count) / rate
                        print(f"  Generated {sample_count}/{total_samples} samples "
                              f"({sample_count/total_samples*100:.1f}%) "
                              f"- ETA: {remaining:.0f}s")
                    
                except Exception as e:
                    print(f"  Error generating trace for '{word}' ({style}): {e}")
    
    elapsed_total = time.time() - start_time
    print(f"\nDataset generation complete!")
    print(f"  Total time: {elapsed_total:.1f} seconds")
    print(f"  Samples generated: {len(dataset)}")
    print(f"  Success rate: {len(dataset)/total_samples*100:.1f}%")
    
    return dataset

def prepare_for_rnn(dataset):
    """
    Prepare dataset for RNN training format
    
    Args:
        dataset: List of trace dictionaries
    
    Returns:
        Dictionary with RNN-ready data
    """
    
    print("\nPreparing data for RNN training...")
    
    # Convert to stroke format (dx, dy, pen_state)
    strokes_data = []
    labels = []
    metadata = []
    
    for sample in dataset:
        trajectory = np.array(sample['trajectory'])
        
        # Calculate deltas
        dx = np.diff(trajectory[:, 0])
        dy = np.diff(trajectory[:, 1])
        dt = np.diff(trajectory[:, 2])
        
        # Normalize by time
        dx = dx / (dt + 1e-8)
        dy = dy / (dt + 1e-8)
        
        # Pen state (1 = down for all points except last)
        pen_state = np.ones(len(dx))
        pen_state[-1] = 0  # Pen up at end
        
        # Combine into stroke format
        strokes = np.column_stack([dx, dy, pen_state])
        
        strokes_data.append(strokes)
        labels.append(sample['word'])
        metadata.append({
            'style': sample['style'],
            'user_speed': sample['user_speed'],
            'precision': sample['precision']
        })
    
    rnn_data = {
        'strokes': strokes_data,
        'labels': labels,
        'metadata': metadata,
        'vocabulary': list(set(labels)),
        'num_samples': len(strokes_data)
    }
    
    print(f"  Prepared {len(strokes_data)} samples for RNN")
    print(f"  Vocabulary size: {len(rnn_data['vocabulary'])}")
    
    return rnn_data

def prepare_for_gan(dataset):
    """
    Prepare dataset for GAN training format
    
    Args:
        dataset: List of trace dictionaries
    
    Returns:
        Dictionary with GAN-ready data
    """
    
    print("\nPreparing data for GAN training...")
    
    # Normalize trajectories to fixed length for GAN
    target_length = 100  # Fixed sequence length
    
    real_traces = []
    fake_traces = []  # Simple center-connected traces
    styles = []
    words = []
    
    for sample in dataset:
        trajectory = np.array(sample['trajectory'])
        
        # Resample to fixed length
        indices = np.linspace(0, len(trajectory)-1, target_length).astype(int)
        resampled = trajectory[indices]
        
        # Normalize to [-1, 1] range
        x_norm = (resampled[:, 0] - resampled[:, 0].mean()) / (resampled[:, 0].std() + 1e-8)
        y_norm = (resampled[:, 1] - resampled[:, 1].mean()) / (resampled[:, 1].std() + 1e-8)
        t_norm = resampled[:, 2] / resampled[:, 2].max()
        
        normalized = np.column_stack([x_norm, y_norm, t_norm])
        real_traces.append(normalized)
        
        # Create simple fake trace (straight lines between keys)
        # This would be replaced with actual key positions in practice
        fake = np.copy(normalized)
        # Simplify by reducing curvature
        for i in range(1, len(fake)-1):
            fake[i] = fake[i-1] * 0.3 + fake[i] * 0.4 + fake[i+1] * 0.3
        fake_traces.append(fake)
        
        styles.append(sample['style'])
        words.append(sample['word'])
    
    gan_data = {
        'real_traces': np.array(real_traces),
        'fake_traces': np.array(fake_traces),
        'styles': styles,
        'words': words,
        'sequence_length': target_length,
        'num_samples': len(real_traces)
    }
    
    print(f"  Prepared {len(real_traces)} samples for GAN")
    print(f"  Sequence length: {target_length}")
    print(f"  Real traces shape: {gan_data['real_traces'].shape}")
    
    return gan_data

def analyze_dataset(dataset):
    """
    Analyze dataset statistics
    
    Args:
        dataset: List of trace dictionaries
    
    Returns:
        Dictionary of statistics
    """
    
    print("\nAnalyzing dataset statistics...")
    
    # Extract metrics
    words = [s['word'] for s in dataset]
    styles = [s['style'] for s in dataset]
    jerks = [s['mean_jerk'] for s in dataset]
    durations = [s['duration'] for s in dataset]
    num_points = [s['num_points'] for s in dataset]
    
    # Calculate statistics
    stats = {
        'total_samples': len(dataset),
        'unique_words': len(set(words)),
        'style_distribution': {s: styles.count(s) for s in set(styles)},
        'jerk_statistics': {
            'mean': np.mean(jerks),
            'std': np.std(jerks),
            'min': np.min(jerks),
            'max': np.max(jerks),
            'median': np.median(jerks),
            'under_1M': sum(j < 1000000 for j in jerks),
            'under_1M_pct': sum(j < 1000000 for j in jerks) / len(jerks) * 100
        },
        'duration_statistics': {
            'mean': np.mean(durations),
            'std': np.std(durations),
            'min': np.min(durations),
            'max': np.max(durations)
        },
        'trajectory_length': {
            'mean': np.mean(num_points),
            'std': np.std(num_points),
            'min': np.min(num_points),
            'max': np.max(num_points)
        },
        'word_length_distribution': {}
    }
    
    # Word length distribution
    for word in set(words):
        length = len(word)
        if length not in stats['word_length_distribution']:
            stats['word_length_distribution'][length] = 0
        stats['word_length_distribution'][length] += words.count(word)
    
    return stats

def save_dataset(dataset, rnn_data, gan_data, stats, output_dir='training_data'):
    """
    Save dataset in multiple formats
    """
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    print(f"\nSaving dataset to {output_dir}/...")
    
    # Save raw dataset as JSON
    with open(output_path / 'raw_dataset.json', 'w') as f:
        json.dump(dataset, f, indent=2)
    print(f"  Saved raw dataset (JSON)")
    
    # Save RNN data as pickle
    with open(output_path / 'rnn_data.pkl', 'wb') as f:
        pickle.dump(rnn_data, f)
    print(f"  Saved RNN data (Pickle)")
    
    # Save GAN data as NPZ
    np.savez_compressed(
        output_path / 'gan_data.npz',
        real_traces=gan_data['real_traces'],
        fake_traces=gan_data['fake_traces'],
        styles=gan_data['styles'],
        words=gan_data['words']
    )
    print(f"  Saved GAN data (NPZ)")
    
    # Save statistics (convert numpy types to Python types)
    def convert_to_python_types(obj):
        if isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_to_python_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_python_types(item) for item in obj]
        return obj
    
    stats_serializable = convert_to_python_types(stats)
    with open(output_path / 'dataset_stats.json', 'w') as f:
        json.dump(stats_serializable, f, indent=2)
    print(f"  Saved dataset statistics")
    
    # Create summary report
    report = f"""
TRAINING DATASET SUMMARY
========================
Generated: {datetime.now().isoformat()}

Dataset Size:
- Total samples: {stats['total_samples']}
- Unique words: {stats['unique_words']}

Style Distribution:
{json.dumps(stats['style_distribution'], indent=2)}

Jerk Quality:
- Mean: {stats['jerk_statistics']['mean']:,.0f} units/s³
- Samples under 1M: {stats['jerk_statistics']['under_1M']} ({stats['jerk_statistics']['under_1M_pct']:.1f}%)

Duration Statistics:
- Mean: {stats['duration_statistics']['mean']:.2f}s
- Range: {stats['duration_statistics']['min']:.2f}s - {stats['duration_statistics']['max']:.2f}s

Files Generated:
- raw_dataset.json: Complete dataset with all metadata
- rnn_data.pkl: Stroke format for RNN training
- gan_data.npz: Normalized traces for GAN training
- dataset_stats.json: Detailed statistics
"""
    
    with open(output_path / 'README.txt', 'w') as f:
        f.write(report)
    
    print(f"\n{report}")
    
    return output_path

def main():
    """Generate complete training dataset"""
    
    print("="*60)
    print("TRAINING DATASET GENERATION")
    print("="*60)
    
    # Initialize generator
    print("\nInitializing improved swype generator...")
    generator = ImprovedSwypeGenerator()
    
    # Generate balanced dataset
    dataset = create_balanced_dataset(generator, num_samples_per_config=5)
    
    # Prepare for different model types
    rnn_data = prepare_for_rnn(dataset)
    gan_data = prepare_for_gan(dataset)
    
    # Analyze statistics
    stats = analyze_dataset(dataset)
    
    # Save everything
    output_path = save_dataset(dataset, rnn_data, gan_data, stats)
    
    print("="*60)
    print("DATASET GENERATION COMPLETE")
    print("="*60)
    print(f"\nDataset saved to: {output_path}/")
    print(f"Ready for model training!")
    
    return dataset, stats

if __name__ == "__main__":
    dataset, stats = main()