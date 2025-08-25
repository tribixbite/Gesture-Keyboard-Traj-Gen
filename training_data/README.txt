
TRAINING DATASET SUMMARY
========================
Generated: 2025-08-25T17:27:43.611440

Dataset Size:
- Total samples: 3440
- Unique words: 172

Style Distribution:
{
  "precise": 860,
  "natural": 860,
  "fast": 860,
  "sloppy": 860
}

Jerk Quality:
- Mean: 920,444 units/s³
- Samples under 1M: 1850 (53.8%)

Duration Statistics:
- Mean: 0.64s
- Range: 0.30s - 2.16s

Files Generated:
- raw_dataset.json: Complete dataset with all metadata
- rnn_data.pkl: Stroke format for RNN training
- gan_data.npz: Normalized traces for GAN training
- dataset_stats.json: Detailed statistics
