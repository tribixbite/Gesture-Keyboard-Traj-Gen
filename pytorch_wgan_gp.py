#!/usr/bin/env python
"""
PyTorch implementation of WGAN-GP for trajectory generation
Complete training pipeline with checkpointing
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
import json
import time
from typing import Dict, List, Tuple, Optional
import pandas as pd

class TrajectoryDataset(Dataset):
    """Dataset for trajectory data"""
    
    def __init__(self, data_dir: str, max_length: int = 150):
        self.data_dir = Path(data_dir)
        self.max_length = max_length
        self.traces = []
        self.words = []
        
        # Load traces
        self._load_traces()
        
        # Build vocabulary
        self._build_vocab()
    
    def _load_traces(self):
        """Load traces from swipelog files"""
        for log_file in list(self.data_dir.glob("*.log"))[:1000]:  # Limit for testing
            try:
                df = pd.read_csv(log_file, sep=r'\s+', engine='python', 
                               on_bad_lines='skip', header=0)
                if 'word' in df.columns and 'x_pos' in df.columns:
                    for word in df['word'].unique():
                        if pd.isna(word) or word == 'word':
                            continue
                        word_df = df[df['word'] == word]
                        trace = word_df[['x_pos', 'y_pos']].values
                        
                        if len(trace) > 2:
                            # Normalize
                            trace = (trace - trace.mean(0)) / (trace.std(0) + 1e-6)
                            self.traces.append(trace)
                            self.words.append(word.lower())
            except:
                continue
    
    def _build_vocab(self):
        """Build character vocabulary"""
        chars = set()
        for word in self.words:
            chars.update(word)
        self.char_to_idx = {char: idx for idx, char in enumerate(sorted(chars))}
        self.idx_to_char = {idx: char for char, idx in self.char_to_idx.items()}
        self.vocab_size = len(self.char_to_idx)
    
    def __len__(self):
        return len(self.traces)
    
    def __getitem__(self, idx):
        trace = self.traces[idx]
        word = self.words[idx]
        
        # Pad/truncate trace
        if len(trace) > self.max_length:
            indices = np.linspace(0, len(trace)-1, self.max_length, dtype=int)
            trace = trace[indices]
        
        # Add end marker
        trace_tensor = torch.zeros(self.max_length, 3)
        trace_tensor[:len(trace), :2] = torch.FloatTensor(trace)
        trace_tensor[len(trace)-1, 2] = 1  # End marker
        
        # Encode word
        word_encoded = torch.zeros(10, dtype=torch.long)  # Max word length
        for i, char in enumerate(word[:10]):
            if char in self.char_to_idx:
                word_encoded[i] = self.char_to_idx[char]
        
        return trace_tensor, word_encoded


class Generator(nn.Module):
    """WGAN-GP Generator"""
    
    def __init__(self, vocab_size: int, latent_dim: int = 128, 
                 max_length: int = 150, hidden_size: int = 256):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.max_length = max_length
        
        # Word embedding
        self.word_embedding = nn.Embedding(vocab_size, 64)
        
        # Initial projection
        self.project = nn.Sequential(
            nn.Linear(latent_dim + 64, hidden_size * 2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size * 2)
        )
        
        # LSTM for sequence generation
        self.lstm = nn.LSTM(
            hidden_size * 2, 
            hidden_size,
            num_layers=2,
            batch_first=True,
            dropout=0.2
        )
        
        # Output head
        self.output = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Linear(128, 3),  # x, y, pen
            nn.Tanh()
        )
    
    def forward(self, noise: torch.Tensor, word: torch.Tensor) -> torch.Tensor:
        batch_size = noise.size(0)
        
        # Embed word (average pooling)
        word_emb = self.word_embedding(word).mean(dim=1)
        
        # Combine with noise
        x = torch.cat([noise, word_emb], dim=1)
        x = self.project(x)
        
        # Expand for sequence
        x = x.unsqueeze(1).repeat(1, self.max_length, 1)
        
        # Generate sequence
        x, _ = self.lstm(x)
        x = self.output(x)
        
        return x


class Discriminator(nn.Module):
    """WGAN-GP Discriminator (Critic)"""
    
    def __init__(self, vocab_size: int, max_length: int = 150):
        super().__init__()
        
        self.max_length = max_length
        
        # Word embedding
        self.word_embedding = nn.Embedding(vocab_size, 64)
        
        # Trajectory encoder
        self.traj_conv = nn.Sequential(
            nn.Conv1d(3, 64, kernel_size=5, stride=2, padding=2),
            nn.LeakyReLU(0.2),
            nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2),
            nn.Conv1d(128, 256, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2)
        )
        
        # Calculate actual output size by running a dummy input
        with torch.no_grad():
            dummy_input = torch.zeros(1, 3, max_length)
            dummy_output = self.traj_conv(dummy_input)
            conv_output_size = dummy_output.reshape(1, -1).shape[1]
        
        # Final critic layers
        self.critic = nn.Sequential(
            nn.Linear(conv_output_size + 64, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(256, 1)
        )
    
    def forward(self, trajectory: torch.Tensor, word: torch.Tensor) -> torch.Tensor:
        # Encode trajectory
        traj = trajectory.transpose(1, 2)  # (B, 3, L)
        traj_features = self.traj_conv(traj)
        traj_features = traj_features.reshape(traj_features.size(0), -1)
        
        # Embed word
        word_emb = self.word_embedding(word).mean(dim=1)
        
        # Combine and critique
        x = torch.cat([traj_features, word_emb], dim=1)
        validity = self.critic(x)
        
        return validity


class WGAN_GP_Trainer:
    """WGAN-GP training pipeline"""
    
    def __init__(self, 
                 generator: Generator,
                 discriminator: Discriminator,
                 device: str = 'cpu',
                 checkpoint_dir: str = 'checkpoints'):
        
        self.G = generator.to(device)
        self.D = discriminator.to(device)
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        # Optimizers
        self.g_optimizer = optim.Adam(self.G.parameters(), lr=0.0001, betas=(0.5, 0.9))
        self.d_optimizer = optim.Adam(self.D.parameters(), lr=0.0001, betas=(0.5, 0.9))
        
        # Training config
        self.lambda_gp = 10
        self.d_steps = 5
        
        # History
        self.history = {'d_loss': [], 'g_loss': [], 'gp': []}
    
    def gradient_penalty(self, real: torch.Tensor, fake: torch.Tensor, 
                        word: torch.Tensor) -> torch.Tensor:
        """Calculate gradient penalty"""
        
        batch_size = real.size(0)
        alpha = torch.rand(batch_size, 1, 1, device=self.device)
        interpolated = alpha * real + (1 - alpha) * fake
        interpolated.requires_grad_(True)
        
        validity = self.D(interpolated, word)
        
        gradients = torch.autograd.grad(
            outputs=validity,
            inputs=interpolated,
            grad_outputs=torch.ones_like(validity),
            create_graph=True,
            retain_graph=True
        )[0]
        
        gradients = gradients.reshape(batch_size, -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        
        return gradient_penalty
    
    def train_step(self, real_traj: torch.Tensor, word: torch.Tensor) -> Dict:
        """Single training step"""
        
        batch_size = real_traj.size(0)
        
        # Train Discriminator
        d_losses = []
        gp_values = []
        
        for _ in range(self.d_steps):
            self.d_optimizer.zero_grad()
            
            # Generate fake
            noise = torch.randn(batch_size, self.G.latent_dim, device=self.device)
            fake_traj = self.G(noise, word).detach()
            
            # Critic scores
            real_validity = self.D(real_traj, word)
            fake_validity = self.D(fake_traj, word)
            
            # Gradient penalty
            gp = self.gradient_penalty(real_traj, fake_traj, word)
            
            # Wasserstein loss
            d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + self.lambda_gp * gp
            
            d_loss.backward()
            self.d_optimizer.step()
            
            d_losses.append(d_loss.item())
            gp_values.append(gp.item())
        
        # Train Generator
        self.g_optimizer.zero_grad()
        
        noise = torch.randn(batch_size, self.G.latent_dim, device=self.device)
        fake_traj = self.G(noise, word)
        fake_validity = self.D(fake_traj, word)
        
        g_loss = -torch.mean(fake_validity)
        
        g_loss.backward()
        self.g_optimizer.step()
        
        return {
            'd_loss': np.mean(d_losses),
            'g_loss': g_loss.item(),
            'gp': np.mean(gp_values)
        }
    
    def train(self, dataloader: DataLoader, epochs: int = 100):
        """Train WGAN-GP"""
        
        print("Starting WGAN-GP Training")
        print("="*50)
        
        for epoch in range(epochs):
            epoch_d_loss = []
            epoch_g_loss = []
            epoch_gp = []
            
            for batch_idx, (real_traj, word) in enumerate(dataloader):
                real_traj = real_traj.to(self.device)
                word = word.to(self.device)
                
                losses = self.train_step(real_traj, word)
                
                epoch_d_loss.append(losses['d_loss'])
                epoch_g_loss.append(losses['g_loss'])
                epoch_gp.append(losses['gp'])
                
                if batch_idx % 10 == 0:
                    print(f"  Epoch {epoch+1}/{epochs} [{batch_idx}/{len(dataloader)}] "
                          f"D: {losses['d_loss']:.4f} G: {losses['g_loss']:.4f} "
                          f"GP: {losses['gp']:.4f}")
            
            # Record epoch statistics
            self.history['d_loss'].append(np.mean(epoch_d_loss))
            self.history['g_loss'].append(np.mean(epoch_g_loss))
            self.history['gp'].append(np.mean(epoch_gp))
            
            print(f"Epoch {epoch+1} - D Loss: {self.history['d_loss'][-1]:.4f} "
                  f"G Loss: {self.history['g_loss'][-1]:.4f} "
                  f"GP: {self.history['gp'][-1]:.4f}")
            
            # Checkpoint every 10 epochs
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(epoch + 1)
                self.generate_samples(word[:3])
    
    def save_checkpoint(self, epoch: int):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'generator_state': self.G.state_dict(),
            'discriminator_state': self.D.state_dict(),
            'g_optimizer': self.g_optimizer.state_dict(),
            'd_optimizer': self.d_optimizer.state_dict(),
            'history': self.history
        }
        
        path = self.checkpoint_dir / f"wgan_gp_epoch_{epoch}.pt"
        torch.save(checkpoint, path)
        print(f"  ✅ Saved checkpoint: {path}")
    
    def load_checkpoint(self, path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.G.load_state_dict(checkpoint['generator_state'])
        self.D.load_state_dict(checkpoint['discriminator_state'])
        self.g_optimizer.load_state_dict(checkpoint['g_optimizer'])
        self.d_optimizer.load_state_dict(checkpoint['d_optimizer'])
        self.history = checkpoint['history']
        
        print(f"✅ Loaded checkpoint from epoch {checkpoint['epoch']}")
        
        return checkpoint['epoch']
    
    def generate_samples(self, words: torch.Tensor, num_samples: int = 1):
        """Generate sample trajectories"""
        
        self.G.eval()
        with torch.no_grad():
            for word in words:
                word = word.unsqueeze(0)
                noise = torch.randn(num_samples, self.G.latent_dim, device=self.device)
                trajectories = self.G(noise, word)
                
                # Process first trajectory
                traj = trajectories[0].cpu().numpy()
                
                # Find end of stroke
                end_idx = np.where(traj[:, 2] > 0.5)[0]
                if len(end_idx) > 0:
                    traj = traj[:end_idx[0] + 1]
                
                print(f"  Generated trajectory: {len(traj)} points")
        
        self.G.train()


def main():
    """Main training function"""
    
    print("="*60)
    print("WGAN-GP TRAINING PIPELINE")
    print("="*60)
    
    # Check for GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load dataset
    print("\nLoading dataset...")
    dataset = TrajectoryDataset("datasets/swipelogs")
    print(f"Loaded {len(dataset)} traces")
    print(f"Vocabulary size: {dataset.vocab_size}")
    
    if len(dataset) == 0:
        print("❌ No data loaded")
        return
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=True,
        num_workers=0
    )
    
    # Create models
    print("\nCreating models...")
    generator = Generator(dataset.vocab_size, latent_dim=128)
    discriminator = Discriminator(dataset.vocab_size)
    
    # Count parameters
    g_params = sum(p.numel() for p in generator.parameters())
    d_params = sum(p.numel() for p in discriminator.parameters())
    print(f"Generator parameters: {g_params:,}")
    print(f"Discriminator parameters: {d_params:,}")
    
    # Create trainer
    trainer = WGAN_GP_Trainer(generator, discriminator, device)
    
    # Train
    print("\nStarting training...")
    trainer.train(dataloader, epochs=10)  # Short training for testing
    
    # Save final model
    trainer.save_checkpoint(999)
    
    print("\n✅ Training complete!")
    print(f"Final D Loss: {trainer.history['d_loss'][-1]:.4f}")
    print(f"Final G Loss: {trainer.history['g_loss'][-1]:.4f}")
    
    return trainer


if __name__ == "__main__":
    trainer = main()