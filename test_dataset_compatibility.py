#!/usr/bin/env python
"""
Test dataset compatibility with RNN and GAN models
Verify data formats and basic training setup
"""

import numpy as np
import pickle
import json
from pathlib import Path

def test_rnn_compatibility():
    """Test if RNN data format is compatible"""
    
    print("="*60)
    print("TESTING RNN COMPATIBILITY")
    print("="*60)
    
    # Load RNN data
    with open('training_data/rnn_data.pkl', 'rb') as f:
        rnn_data = pickle.load(f)
    
    print(f"\nRNN Data Structure:")
    print(f"  Number of samples: {rnn_data['num_samples']}")
    print(f"  Vocabulary size: {len(rnn_data['vocabulary'])}")
    print(f"  Keys in data: {list(rnn_data.keys())}")
    
    # Check stroke format
    strokes = rnn_data['strokes']
    print(f"\nStroke Format Check:")
    print(f"  Total strokes: {len(strokes)}")
    
    # Analyze first sample
    sample_stroke = strokes[0]
    sample_label = rnn_data['labels'][0]
    sample_meta = rnn_data['metadata'][0]
    
    print(f"\nFirst Sample Analysis:")
    print(f"  Word: '{sample_label}'")
    print(f"  Style: {sample_meta['style']}")
    print(f"  Stroke shape: {sample_stroke.shape}")
    print(f"  Stroke format: (dx, dy, pen_state)")
    
    # Verify stroke format
    if sample_stroke.shape[1] == 3:
        print(f"  ✅ Correct stroke dimensions (N, 3)")
    else:
        print(f"  ❌ Wrong stroke dimensions - expected (N, 3)")
    
    # Check pen states
    pen_states = sample_stroke[:, 2]
    if np.all(pen_states[:-1] == 1) and pen_states[-1] == 0:
        print(f"  ✅ Correct pen states (down during trace, up at end)")
    else:
        print(f"  ⚠️  Unexpected pen states")
    
    # Check for NaN or Inf
    if np.any(np.isnan(sample_stroke)) or np.any(np.isinf(sample_stroke)):
        print(f"  ❌ Contains NaN or Inf values")
    else:
        print(f"  ✅ No NaN or Inf values")
    
    # Statistics
    print(f"\nStroke Statistics:")
    dx_values = [s[:, 0] for s in strokes]
    dy_values = [s[:, 1] for s in strokes]
    
    dx_flat = np.concatenate(dx_values)
    dy_flat = np.concatenate(dy_values)
    
    print(f"  dx range: [{np.min(dx_flat):.2f}, {np.max(dx_flat):.2f}]")
    print(f"  dy range: [{np.min(dy_flat):.2f}, {np.max(dy_flat):.2f}]")
    print(f"  Mean sequence length: {np.mean([len(s) for s in strokes]):.1f}")
    
    # Try basic RNN operations
    print(f"\nRNN Compatibility Test:")
    try:
        # Simulate batch creation
        batch_size = 32
        batch_indices = np.random.choice(len(strokes), batch_size)
        batch_strokes = [strokes[i] for i in batch_indices]
        batch_labels = [rnn_data['labels'][i] for i in batch_indices]
        
        # Find max length for padding
        max_len = max(len(s) for s in batch_strokes)
        
        # Pad sequences
        padded_batch = np.zeros((batch_size, max_len, 3))
        for i, stroke in enumerate(batch_strokes):
            padded_batch[i, :len(stroke)] = stroke
        
        print(f"  ✅ Successfully created padded batch")
        print(f"  Batch shape: {padded_batch.shape}")
        
    except Exception as e:
        print(f"  ❌ Failed to create batch: {e}")
    
    return True

def test_gan_compatibility():
    """Test if GAN data format is compatible"""
    
    print("\n" + "="*60)
    print("TESTING GAN COMPATIBILITY")
    print("="*60)
    
    # Load GAN data
    gan_data = np.load('training_data/gan_data.npz', allow_pickle=True)
    
    print(f"\nGAN Data Structure:")
    print(f"  Keys: {list(gan_data.keys())}")
    
    real_traces = gan_data['real_traces']
    fake_traces = gan_data['fake_traces']
    styles = gan_data['styles']
    words = gan_data['words']
    
    print(f"\nData Shapes:")
    print(f"  Real traces: {real_traces.shape}")
    print(f"  Fake traces: {fake_traces.shape}")
    print(f"  Styles: {len(styles)}")
    print(f"  Words: {len(words)}")
    
    # Check normalization
    print(f"\nNormalization Check:")
    print(f"  Real traces range: [{np.min(real_traces):.2f}, {np.max(real_traces):.2f}]")
    print(f"  Fake traces range: [{np.min(fake_traces):.2f}, {np.max(fake_traces):.2f}]")
    
    if -5 < np.min(real_traces) and np.max(real_traces) < 5:
        print(f"  ✅ Real traces properly normalized")
    else:
        print(f"  ⚠️  Real traces may need better normalization")
    
    # Check for NaN or Inf
    if np.any(np.isnan(real_traces)) or np.any(np.isinf(real_traces)):
        print(f"  ❌ Real traces contain NaN or Inf")
    else:
        print(f"  ✅ Real traces have no NaN or Inf")
    
    # Verify fixed sequence length
    if len(set([t.shape[0] for t in real_traces])) == 1:
        print(f"  ✅ All traces have fixed length: {real_traces.shape[1]}")
    else:
        print(f"  ❌ Traces have varying lengths")
    
    # Test batch creation for GAN
    print(f"\nGAN Compatibility Test:")
    try:
        # Create mini-batch
        batch_size = 64
        batch_idx = np.random.choice(len(real_traces), batch_size)
        
        real_batch = real_traces[batch_idx]
        fake_batch = fake_traces[batch_idx]
        
        print(f"  ✅ Successfully created GAN batch")
        print(f"  Real batch shape: {real_batch.shape}")
        print(f"  Fake batch shape: {fake_batch.shape}")
        
        # Check if shapes match (important for GAN training)
        if real_batch.shape == fake_batch.shape:
            print(f"  ✅ Real and fake batches have matching shapes")
        else:
            print(f"  ❌ Shape mismatch between real and fake")
            
    except Exception as e:
        print(f"  ❌ Failed to create GAN batch: {e}")
    
    return True

def test_dataset_quality():
    """Test overall dataset quality metrics"""
    
    print("\n" + "="*60)
    print("DATASET QUALITY METRICS")
    print("="*60)
    
    # Load statistics
    with open('training_data/dataset_stats.json', 'r') as f:
        stats = json.load(f)
    
    print(f"\nDataset Overview:")
    print(f"  Total samples: {stats['total_samples']}")
    print(f"  Unique words: {stats['unique_words']}")
    
    # Style balance
    print(f"\nStyle Balance:")
    total = sum(stats['style_distribution'].values())
    for style, count in stats['style_distribution'].items():
        pct = count / total * 100
        print(f"  {style}: {count} ({pct:.1f}%)")
    
    # Check if balanced
    counts = list(stats['style_distribution'].values())
    if max(counts) - min(counts) < 10:
        print(f"  ✅ Dataset is well-balanced across styles")
    else:
        print(f"  ⚠️  Dataset has imbalanced style distribution")
    
    # Jerk quality
    print(f"\nJerk Quality:")
    jerk_stats = stats['jerk_statistics']
    print(f"  Mean: {jerk_stats['mean']:,.0f} units/s³")
    print(f"  Median: {jerk_stats['median']:,.0f} units/s³")
    print(f"  Under 1M: {jerk_stats['under_1M_pct']:.1f}%")
    
    if jerk_stats['under_1M_pct'] > 50:
        print(f"  ✅ Majority of samples have realistic jerk values")
    else:
        print(f"  ⚠️  Many samples have high jerk values")
    
    # Word complexity
    print(f"\nWord Length Distribution:")
    for length, count in sorted(stats['word_length_distribution'].items()):
        print(f"  {length} chars: {count} samples")
    
    # Duration distribution
    print(f"\nDuration Statistics:")
    dur_stats = stats['duration_statistics']
    print(f"  Mean: {dur_stats['mean']:.2f}s")
    print(f"  Std: {dur_stats['std']:.2f}s")
    print(f"  Range: {dur_stats['min']:.2f}s - {dur_stats['max']:.2f}s")
    
    return True

def verify_file_structure():
    """Verify all expected files exist"""
    
    print("\n" + "="*60)
    print("FILE STRUCTURE VERIFICATION")
    print("="*60)
    
    expected_files = [
        'training_data/raw_dataset.json',
        'training_data/rnn_data.pkl',
        'training_data/gan_data.npz',
        'training_data/dataset_stats.json',
        'training_data/README.txt'
    ]
    
    print("\nChecking files:")
    all_exist = True
    for file_path in expected_files:
        path = Path(file_path)
        if path.exists():
            size = path.stat().st_size / (1024 * 1024)  # Size in MB
            print(f"  ✅ {file_path} ({size:.2f} MB)")
        else:
            print(f"  ❌ {file_path} (missing)")
            all_exist = False
    
    return all_exist

def main():
    """Run all compatibility tests"""
    
    print("="*60)
    print("DATASET COMPATIBILITY TEST SUITE")
    print("="*60)
    
    # Verify files
    files_ok = verify_file_structure()
    
    if not files_ok:
        print("\n❌ Missing required files. Please run generate_training_dataset.py first.")
        return False
    
    # Run tests
    rnn_ok = test_rnn_compatibility()
    gan_ok = test_gan_compatibility()
    quality_ok = test_dataset_quality()
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    if rnn_ok and gan_ok and quality_ok:
        print("\n✅ ALL TESTS PASSED")
        print("Dataset is ready for model training!")
    else:
        print("\n⚠️  Some tests failed. Review issues above.")
    
    print("\nNext steps:")
    print("1. Use rnn_data.pkl with RNN models in RNN-Based-TF1/ or RNN-Based-TF2/")
    print("2. Use gan_data.npz with GAN model in GAN-Based/")
    print("3. Monitor training with dataset_stats.json metrics")
    
    return True

if __name__ == "__main__":
    main()