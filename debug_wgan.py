#!/usr/bin/env python
"""Debug WGAN dimension issue"""

import torch
from pytorch_wgan_gp import Discriminator

vocab_size = 30
max_length = 50
batch_size = 4

disc = Discriminator(vocab_size, max_length)

# Test trajectory
traj = torch.randn(batch_size, max_length, 3)
word = torch.randint(0, vocab_size, (batch_size, 10))

# Debug forward pass
traj_t = traj.transpose(1, 2)  # (B, 3, L)
print(f"Input shape: {traj_t.shape}")

# Pass through conv layers
x = traj_t
for i, layer in enumerate(disc.traj_conv):
    x = layer(x)
    if hasattr(layer, '__class__') and 'Conv' in layer.__class__.__name__:
        print(f"After conv {i}: {x.shape}")

x_flat = x.view(x.size(0), -1)
print(f"Flattened shape: {x_flat.shape}")
print(f"Flattened features: {x_flat.shape[1]}")

# Word embedding
word_emb = disc.word_embedding(word).mean(dim=1)
print(f"Word embedding shape: {word_emb.shape}")

# Combine
combined = torch.cat([x_flat, word_emb], dim=1)
print(f"Combined shape: {combined.shape}")
print(f"Expected input to critic: {combined.shape[1]}")

# Calculate expected size
conv_output_len = max_length
for _ in range(3):
    conv_output_len = (conv_output_len + 1) // 2
expected = 256 * conv_output_len + 64
print(f"Calculated expected size: {expected}")