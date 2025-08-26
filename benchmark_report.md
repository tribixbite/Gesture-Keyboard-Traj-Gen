# Model Benchmark Report

Generated: 2025-08-26 14:07:59

Device: cpu

## Quality Metrics

### Smoothness Scores
| Model | Mean | Std | Min | Max |
|-------|------|-----|-----|-----|
| attention_rnn | 0.500 | 0.500 | 0.000 | 1.000 |
| wgan_gp | 0.975 | 0.010 | 0.956 | 0.986 |
| transformer | 1.000 | 0.000 | 1.000 | 1.000 |

### Trajectory Length
| Model | Mean | Std | Min | Max |
|-------|------|-----|-----|-----|
| attention_rnn | 4.0 | 1.7 | 2.3 | 5.7 |
| wgan_gp | 0.0 | 0.0 | 0.0 | 0.0 |
| transformer | 5.7 | 0.0 | 5.7 | 5.7 |

## Speed Performance
| Model | Avg Time (ms) | Std (ms) | Traj/sec |
|-------|---------------|----------|----------|
| attention_rnn | 1.40 | 0.35 | 712.3 |
| wgan_gp | 38.31 | 5.46 | 26.1 |
| transformer | 7.08 | 3.29 | 141.3 |

## Memory Usage
| Model | Total Params | Trainable | Memory (MB) |
|-------|--------------|-----------|-------------|
| attention_rnn | 1,762,289 | 1,762,289 | 6.72 |
| wgan_gp | 1,451,139 | 1,451,139 | 5.54 |
| transformer | 5,575,545 | 5,575,545 | 21.27 |

## Summary
- **Best Smoothness**: transformer (1.000)
- **Fastest Generation**: attention_rnn (1.40ms)
- **Smallest Model**: wgan_gp (1,451,139 parameters)