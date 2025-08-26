#!/usr/bin/env python
"""Quick training test with minimal data"""

import torch
import numpy as np
from pathlib import Path

# Test imports
print("Testing imports...")
try:
    from pytorch_attention_rnn import AttentionRNN, AttentionTrainer
    print("✅ Attention RNN imported")
except Exception as e:
    print(f"❌ Attention RNN import failed: {e}")

try:
    from pytorch_wgan_gp import Generator, Discriminator, WGAN_GP_Trainer
    print("✅ WGAN-GP imported")
except Exception as e:
    print(f"❌ WGAN-GP import failed: {e}")

try:
    from pytorch_transformer import TransformerTrajectoryModel, TransformerTrainer
    print("✅ Transformer imported")
except Exception as e:
    print(f"❌ Transformer import failed: {e}")

# Create minimal synthetic data
print("\nCreating minimal test data...")
batch_size = 4
seq_len = 50
vocab_size = 30

# Random test data
traces = torch.randn(batch_size, seq_len, 3)
words = torch.randint(0, vocab_size, (batch_size, 10))
word_lengths = torch.tensor([5, 4, 6, 3])

device = torch.device("cpu")
print(f"Device: {device}")

# Test each model
print("\n" + "="*50)
print("TESTING MODELS")
print("="*50)

# 1. Attention RNN
print("\n1. Testing Attention RNN...")
try:
    model = AttentionRNN(vocab_size, embedding_size=64, hidden_size=128)
    trainer = AttentionTrainer(model, device)
    
    # One training step
    loss = trainer.train_step(words, traces, word_lengths)
    print(f"   ✅ Training step completed, loss: {loss:.4f}")
    
    # Generate sample
    generated = model.generate(words[:1], max_length=50)
    print(f"   ✅ Generation works, shape: {generated.shape}")
except Exception as e:
    print(f"   ❌ Attention RNN failed: {e}")

# 2. WGAN-GP
print("\n2. Testing WGAN-GP...")
try:
    generator = Generator(vocab_size, latent_dim=64, max_length=seq_len, hidden_size=128)
    discriminator = Discriminator(vocab_size, max_length=seq_len)
    trainer = WGAN_GP_Trainer(generator, discriminator, device)
    
    # One training step
    losses = trainer.train_step(traces, words)
    print(f"   ✅ Training step completed")
    print(f"      D Loss: {losses['d_loss']:.4f}")
    print(f"      G Loss: {losses['g_loss']:.4f}")
    
    # Generate sample
    noise = torch.randn(1, 64)
    generated = generator(noise, words[:1])
    print(f"   ✅ Generation works, shape: {generated.shape}")
except Exception as e:
    print(f"   ❌ WGAN-GP failed: {e}")

# 3. Transformer
print("\n3. Testing Transformer...")
try:
    model = TransformerTrajectoryModel(
        vocab_size, d_model=128, nhead=4, 
        num_encoder_layers=2, num_decoder_layers=2
    )
    trainer = TransformerTrainer(model, device)
    
    # One training step
    loss = trainer.train_step(words, traces)
    print(f"   ✅ Training step completed, loss: {loss:.4f}")
    
    # Generate sample
    generated = model.generate(words[:1], max_length=50)
    print(f"   ✅ Generation works, shape: {generated.shape}")
except Exception as e:
    print(f"   ❌ Transformer failed: {e}")

print("\n" + "="*50)
print("✅ All models tested successfully!")
print("="*50)