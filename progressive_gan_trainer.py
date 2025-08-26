#!/usr/bin/env python
"""
Progressive GAN training improvements for trajectory generation
Implements progressive growing, spectral normalization, and advanced training techniques
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math
import time
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path
import json
from tensorboard_logger import TensorboardTrainingMixin

class SpectralNorm(nn.Module):
    """Spectral normalization for improved GAN training stability"""
    
    def __init__(self, module: nn.Module, name: str = 'weight', n_power_iterations: int = 1, eps: float = 1e-12):
        super().__init__()
        self.module = module
        self.name = name
        self.n_power_iterations = n_power_iterations
        self.eps = eps
        
        # Get weight
        if not hasattr(module, name):
            raise ValueError(f"Module {module} has no parameter {name}")
        
        weight = getattr(module, name)
        
        # Remove weight from parameter list
        delattr(module, name)
        
        # Add as buffer
        self.register_buffer(name, weight)
        
        # Initialize u and v for power iteration
        with torch.no_grad():
            h, w = weight.size(0), weight.view(weight.size(0), -1).size(1)
            self.register_buffer('u', torch.randn(h).normal_(0, 1))
            self.register_buffer('v', torch.randn(w).normal_(0, 1))
    
    def forward(self, *args, **kwargs):
        self.compute_spectral_norm()
        return self.module(*args, **kwargs)
    
    def compute_spectral_norm(self):
        weight = getattr(self, self.name)
        weight_mat = weight.view(weight.size(0), -1)
        
        with torch.no_grad():
            for _ in range(self.n_power_iterations):
                # Power iteration
                self.v = F.normalize(torch.mv(weight_mat.t(), self.u), eps=self.eps)
                self.u = F.normalize(torch.mv(weight_mat, self.v), eps=self.eps)
        
        # Compute spectral norm
        sigma = torch.dot(self.u, torch.mv(weight_mat, self.v))
        
        # Normalize weight
        setattr(self.module, self.name, weight / sigma)

def spectral_norm(module: nn.Module, name: str = 'weight', n_power_iterations: int = 1) -> nn.Module:
    """Apply spectral normalization to a module"""
    SpectralNorm(module, name, n_power_iterations)
    return module

class SelfAttention(nn.Module):
    """Self-attention module for improved feature quality"""
    
    def __init__(self, in_channels: int):
        super().__init__()
        self.in_channels = in_channels
        
        self.query = nn.Conv1d(in_channels, in_channels // 8, 1)
        self.key = nn.Conv1d(in_channels, in_channels // 8, 1)
        self.value = nn.Conv1d(in_channels, in_channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, channels, length = x.size()
        
        # Compute query, key, value
        query = self.query(x).view(batch_size, -1, length).permute(0, 2, 1)  # [B, L, C']
        key = self.key(x).view(batch_size, -1, length)  # [B, C', L]
        value = self.value(x).view(batch_size, -1, length)  # [B, C, L]
        
        # Compute attention weights
        attention = torch.bmm(query, key)  # [B, L, L]
        attention = F.softmax(attention, dim=2)
        
        # Apply attention to values
        out = torch.bmm(value, attention.permute(0, 2, 1))  # [B, C, L]
        out = out.view(batch_size, channels, length)
        
        # Apply learnable scaling and residual connection
        out = self.gamma * out + x
        
        return out

class ProgressiveGenerator(nn.Module):
    """Progressive generator with growing architecture"""
    
    def __init__(self, 
                 vocab_size: int,
                 latent_dim: int = 128,
                 max_trajectory_length: int = 200,
                 hidden_channels: int = 256):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.latent_dim = latent_dim
        self.max_trajectory_length = max_trajectory_length
        self.hidden_channels = hidden_channels
        
        # Word embedding
        self.word_embedding = nn.Embedding(vocab_size, 64)
        
        # Progressive stages (each stage doubles the trajectory length)
        self.num_stages = int(math.log2(max_trajectory_length // 4)) + 1  # Start from length 4
        self.current_stage = 0
        self.alpha = 1.0  # Blending factor for progressive growing
        
        # Initial dense layers
        self.initial = nn.Sequential(
            nn.Linear(latent_dim + 64, hidden_channels * 4),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(hidden_channels * 4),
            nn.Linear(hidden_channels * 4, hidden_channels * 4),
            nn.ReLU(inplace=True)
        )
        
        # Progressive blocks
        self.progressive_blocks = nn.ModuleList()
        self.to_traj_layers = nn.ModuleList()
        
        current_channels = hidden_channels
        current_length = 4
        
        for stage in range(self.num_stages):
            # Upsampling block
            if stage == 0:
                # First stage: reshape to initial trajectory
                block = nn.Sequential(
                    nn.Linear(hidden_channels * 4, current_channels * current_length),
                )
            else:
                # Subsequent stages: 1D convolutions for upsampling
                block = nn.Sequential(
                    nn.Upsample(scale_factor=2, mode='linear', align_corners=False),
                    nn.Conv1d(current_channels, current_channels // 2, kernel_size=3, padding=1),
                    nn.BatchNorm1d(current_channels // 2),
                    nn.ReLU(inplace=True),
                    nn.Conv1d(current_channels // 2, current_channels // 2, kernel_size=3, padding=1),
                    nn.BatchNorm1d(current_channels // 2),
                    nn.ReLU(inplace=True)
                )
                
                # Add self-attention at higher resolutions
                if current_length >= 32:
                    block.add_module('self_attention', SelfAttention(current_channels // 2))
            
            self.progressive_blocks.append(block)
            
            # Output layer for this stage
            if stage == 0:
                to_traj = nn.Sequential(
                    nn.Linear(current_channels, 3),
                    nn.Tanh()
                )
                current_channels = current_channels // 2
            else:
                to_traj = nn.Sequential(
                    nn.Conv1d(current_channels // 2, 3, kernel_size=1),
                    nn.Tanh()
                )
                current_channels = current_channels // 2
            
            self.to_traj_layers.append(to_traj)
            current_length *= 2
    
    def set_stage(self, stage: int, alpha: float = 1.0):
        """Set current training stage and blending factor"""
        self.current_stage = min(stage, self.num_stages - 1)
        self.alpha = alpha
    
    def forward(self, noise: torch.Tensor, word: torch.Tensor) -> torch.Tensor:
        batch_size = noise.size(0)
        
        # Embed word
        word_emb = self.word_embedding(word).mean(dim=1)  # Average pooling
        
        # Concatenate noise and word embedding
        x = torch.cat([noise, word_emb], dim=1)
        
        # Initial processing
        x = self.initial(x)
        
        # Progressive generation
        for stage in range(self.current_stage + 1):
            if stage == 0:
                # First stage: reshape to trajectory
                x = self.progressive_blocks[stage](x)
                x = x.view(batch_size, -1, 4).transpose(1, 2)  # [B, 4, C]
            else:
                # Subsequent stages: upsampling
                x = self.progressive_blocks[stage](x)
            
            # Generate trajectory for this stage
            if stage == self.current_stage:
                # Final output
                traj = self.to_traj_layers[stage](x)
                if stage == 0:
                    traj = traj.transpose(1, 2)  # [B, L, 3]
                else:
                    traj = traj.transpose(1, 2)  # [B, L, 3]
                
                # If we're in transition phase, blend with previous stage
                if self.alpha < 1.0 and stage > 0:
                    # Generate previous stage output
                    prev_x = F.interpolate(x, scale_factor=0.5, mode='linear', align_corners=False)
                    prev_traj = self.to_traj_layers[stage - 1](prev_x).transpose(1, 2)
                    
                    # Upsample previous trajectory
                    prev_traj_upsampled = F.interpolate(
                        prev_traj.transpose(1, 2), 
                        size=traj.size(1), 
                        mode='linear', 
                        align_corners=False
                    ).transpose(1, 2)
                    
                    # Blend trajectories
                    traj = self.alpha * traj + (1 - self.alpha) * prev_traj_upsampled
                
                return traj
        
        return traj

class ProgressiveDiscriminator(nn.Module):
    """Progressive discriminator with growing architecture"""
    
    def __init__(self, 
                 vocab_size: int,
                 max_trajectory_length: int = 200,
                 hidden_channels: int = 256):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.max_trajectory_length = max_trajectory_length
        self.hidden_channels = hidden_channels
        
        # Word embedding
        self.word_embedding = nn.Embedding(vocab_size, 64)
        
        # Progressive stages
        self.num_stages = int(math.log2(max_trajectory_length // 4)) + 1
        self.current_stage = 0
        self.alpha = 1.0
        
        # From trajectory layers
        self.from_traj_layers = nn.ModuleList()
        self.progressive_blocks = nn.ModuleList()
        
        current_channels = 16
        current_length = max_trajectory_length
        
        # Build progressive layers (reverse order from generator)
        for stage in range(self.num_stages):
            # Input layer for this stage
            if stage == 0:
                # Highest resolution input
                from_traj = spectral_norm(nn.Conv1d(3, current_channels, kernel_size=1))
            else:
                from_traj = spectral_norm(nn.Conv1d(3, current_channels, kernel_size=1))
            
            self.from_traj_layers.append(from_traj)
            
            # Progressive block
            if stage == self.num_stages - 1:
                # Final stage: to classification
                block = nn.Sequential(
                    spectral_norm(nn.Conv1d(current_channels, current_channels * 2, 3, padding=1)),
                    nn.LeakyReLU(0.2, inplace=True),
                    spectral_norm(nn.Conv1d(current_channels * 2, current_channels * 4, 3, padding=1)),
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.AdaptiveAvgPool1d(1)
                )
            else:
                block = nn.Sequential(
                    spectral_norm(nn.Conv1d(current_channels, current_channels * 2, 3, padding=1)),
                    nn.LeakyReLU(0.2, inplace=True),
                    spectral_norm(nn.Conv1d(current_channels * 2, current_channels * 2, 3, padding=1)),
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.AvgPool1d(2)
                )
                
                # Add self-attention at appropriate stages
                if current_length >= 32:
                    block.add_module('self_attention', SelfAttention(current_channels * 2))
            
            self.progressive_blocks.append(block)
            current_channels *= 2
            current_length //= 2
        
        # Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(current_channels + 64, 512),  # +64 for word embedding
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            spectral_norm(nn.Linear(512, 1))
        )
    
    def set_stage(self, stage: int, alpha: float = 1.0):
        """Set current training stage and blending factor"""
        self.current_stage = min(stage, self.num_stages - 1)
        self.alpha = alpha
    
    def forward(self, trajectory: torch.Tensor, word: torch.Tensor) -> torch.Tensor:
        batch_size = trajectory.size(0)
        
        # Embed word
        word_emb = self.word_embedding(word).mean(dim=1)
        
        # Convert trajectory to appropriate format
        x = trajectory.transpose(1, 2)  # [B, 3, L]
        
        # Progressive discrimination
        stage_outputs = []
        
        for stage in range(self.current_stage + 1):
            if stage == 0:
                # Process at current resolution
                stage_x = self.from_traj_layers[stage](x)
                stage_outputs.append(stage_x)
            else:
                # Downsample for lower resolution processing
                downsampled_x = F.avg_pool1d(x, kernel_size=2**stage)
                stage_x = self.from_traj_layers[stage](downsampled_x)
                stage_outputs.append(stage_x)
        
        # Start from the lowest resolution
        current_x = stage_outputs[-1]
        
        # Process through progressive blocks (in reverse order)
        for stage in range(self.current_stage, -1, -1):
            block_idx = self.num_stages - 1 - (self.current_stage - stage)
            
            if stage == self.current_stage:
                # Process current stage
                current_x = self.progressive_blocks[block_idx](current_x)
                
                # If in transition, blend with upsampled lower resolution
                if self.alpha < 1.0 and stage > 0:
                    # Process next lower stage
                    lower_x = self.progressive_blocks[block_idx + 1](stage_outputs[stage - 1])
                    
                    # Upsample lower resolution features
                    if lower_x.size(2) != current_x.size(2):
                        lower_x = F.interpolate(lower_x, size=current_x.size(2), mode='linear', align_corners=False)
                    
                    # Blend features
                    current_x = self.alpha * current_x + (1 - self.alpha) * lower_x
            elif stage == self.current_stage - 1 and self.alpha < 1.0:
                # Handle transition blending
                continue
            else:
                # Process intermediate stages
                current_x = self.progressive_blocks[block_idx](current_x)
        
        # Global average pooling and classification
        if current_x.dim() == 3:
            current_x = F.adaptive_avg_pool1d(current_x, 1).squeeze(2)
        else:
            current_x = current_x.view(batch_size, -1)
        
        # Combine with word embedding
        combined = torch.cat([current_x, word_emb], dim=1)
        
        # Final classification
        output = self.classifier(combined)
        
        return output

class ProgressiveGANTrainer(TensorboardTrainingMixin):
    """Progressive GAN training with advanced techniques"""
    
    def __init__(self,
                 generator: ProgressiveGenerator,
                 discriminator: ProgressiveDiscriminator,
                 device: str = 'cpu',
                 learning_rate: float = 0.0001,
                 use_tensorboard: bool = True):
        
        self.generator = generator.to(device)
        self.discriminator = discriminator.to(device)
        self.device = device
        
        # Separate optimizers for generator and discriminator
        self.g_optimizer = optim.Adam(
            generator.parameters(),
            lr=learning_rate,
            betas=(0.0, 0.99)  # Progressive GAN uses different betas
        )
        
        self.d_optimizer = optim.Adam(
            discriminator.parameters(),
            lr=learning_rate,
            betas=(0.0, 0.99)
        )
        
        # Training parameters
        self.gradient_penalty_lambda = 10.0
        self.drift_loss_lambda = 0.001
        self.current_stage = 0
        self.transition_phase = False
        self.transition_alpha = 0.0
        
        # Progressive training schedule
        self.stage_lengths = [800000, 800000, 800000, 800000, 800000]  # Steps per stage
        self.transition_lengths = [800000, 800000, 800000, 800000]     # Transition steps
        
        # Training history
        self.history = {
            'stage': [],
            'alpha': [],
            'd_loss': [],
            'g_loss': [],
            'gp_loss': []
        }
        
        # Initialize Tensorboard if requested
        if use_tensorboard:
            self.init_tensorboard(
                experiment_name=f"progressive_gan_{int(time.time())}",
                model_name="Progressive_GAN"
            )
    
    def gradient_penalty(self,
                        real_data: torch.Tensor,
                        fake_data: torch.Tensor,
                        word: torch.Tensor) -> torch.Tensor:
        """Compute gradient penalty for WGAN-GP"""
        
        batch_size = real_data.size(0)
        
        # Random interpolation factor
        epsilon = torch.rand(batch_size, 1, 1, device=self.device)
        
        # Interpolated data
        interpolated = epsilon * real_data + (1 - epsilon) * fake_data
        interpolated.requires_grad_(True)
        
        # Discriminator output on interpolated data
        d_interpolated = self.discriminator(interpolated, word)
        
        # Compute gradients
        gradients = torch.autograd.grad(
            outputs=d_interpolated,
            inputs=interpolated,
            grad_outputs=torch.ones_like(d_interpolated),
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]
        
        # Compute gradient penalty
        gradients = gradients.view(batch_size, -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        
        return gradient_penalty
    
    def drift_loss(self, real_output: torch.Tensor) -> torch.Tensor:
        """Compute drift loss to prevent discriminator drift"""
        return 0.5 * torch.mean(real_output ** 2)
    
    def set_stage(self, stage: int, alpha: float = 1.0):
        """Set training stage for both generator and discriminator"""
        self.current_stage = stage
        self.transition_alpha = alpha
        
        self.generator.set_stage(stage, alpha)
        self.discriminator.set_stage(stage, alpha)
    
    def train_step(self,
                   real_trajectories: torch.Tensor,
                   words: torch.Tensor,
                   n_critic: int = 5) -> Dict[str, float]:
        """Single training step with progressive training"""
        
        batch_size = real_trajectories.size(0)
        
        # Train discriminator n_critic times
        d_losses = []
        gp_losses = []
        
        for _ in range(n_critic):
            self.d_optimizer.zero_grad()
            
            # Real data
            real_output = self.discriminator(real_trajectories, words)
            
            # Fake data
            noise = torch.randn(batch_size, self.generator.latent_dim, device=self.device)
            fake_trajectories = self.generator(noise, words).detach()
            fake_output = self.discriminator(fake_trajectories, words)
            
            # WGAN loss
            wgan_loss = torch.mean(fake_output) - torch.mean(real_output)
            
            # Gradient penalty
            gp_loss = self.gradient_penalty(real_trajectories, fake_trajectories, words)
            
            # Drift loss
            drift = self.drift_loss(real_output)
            
            # Total discriminator loss
            d_loss = wgan_loss + self.gradient_penalty_lambda * gp_loss + self.drift_loss_lambda * drift
            
            d_loss.backward()
            self.d_optimizer.step()
            
            d_losses.append(d_loss.item())
            gp_losses.append(gp_loss.item())
        
        # Train generator
        self.g_optimizer.zero_grad()
        
        noise = torch.randn(batch_size, self.generator.latent_dim, device=self.device)
        fake_trajectories = self.generator(noise, words)
        fake_output = self.discriminator(fake_trajectories, words)
        
        # Generator loss
        g_loss = -torch.mean(fake_output)
        
        g_loss.backward()
        self.g_optimizer.step()
        
        return {
            'd_loss': np.mean(d_losses),
            'g_loss': g_loss.item(),
            'gp_loss': np.mean(gp_losses)
        }
    
    def train_progressive(self,
                         dataloader: DataLoader,
                         total_steps: int = 5000000):
        """Train with progressive growing"""
        
        print("Starting Progressive GAN Training")
        print("=" * 50)
        
        step = 0
        current_stage = 0
        stage_step = 0
        in_transition = False
        
        while step < total_steps and current_stage < self.generator.num_stages:
            for batch in dataloader:
                if step >= total_steps:
                    break
                
                real_trajectories, words = batch
                real_trajectories = real_trajectories.to(self.device)
                words = words.to(self.device)
                
                # Resize trajectories for current stage
                current_length = 4 * (2 ** current_stage)
                if real_trajectories.size(1) != current_length:
                    real_trajectories = F.interpolate(
                        real_trajectories.transpose(1, 2),
                        size=current_length,
                        mode='linear',
                        align_corners=False
                    ).transpose(1, 2)
                
                # Set progressive stage
                alpha = 1.0
                if in_transition:
                    transition_progress = stage_step / self.transition_lengths[current_stage - 1]
                    alpha = min(1.0, transition_progress)
                
                self.set_stage(current_stage, alpha)
                
                # Training step
                losses = self.train_step(real_trajectories, words)
                
                # Record history
                self.history['stage'].append(current_stage)
                self.history['alpha'].append(alpha)
                self.history['d_loss'].append(losses['d_loss'])
                self.history['g_loss'].append(losses['g_loss'])
                self.history['gp_loss'].append(losses['gp_loss'])
                
                # Progress logging
                if step % 100 == 0:
                    stage_info = f"Stage {current_stage}"
                    if in_transition:
                        stage_info += f" (transition α={alpha:.3f})"
                    
                    print(f"Step {step}: {stage_info} | "
                          f"D: {losses['d_loss']:.4f} | "
                          f"G: {losses['g_loss']:.4f} | "
                          f"GP: {losses['gp_loss']:.4f}")
                    
                    # Log to Tensorboard
                    if hasattr(self, 'tb_logger'):
                        self.tb_logger.log_scalars('Losses', {
                            'discriminator': losses['d_loss'],
                            'generator': losses['g_loss'],
                            'gradient_penalty': losses['gp_loss']
                        }, step)
                        
                        self.tb_logger.log_scalars('Progressive_Training', {
                            'current_stage': current_stage,
                            'alpha': alpha,
                            'trajectory_length': current_length
                        }, step)
                
                # Log model parameters every 1000 steps
                if step % 1000 == 0 and hasattr(self, 'tb_logger'):
                    self.tb_logger.log_model_parameters(self.generator, step)
                    self.tb_logger.log_model_parameters(self.discriminator, step)
                
                # Generate and log sample trajectories every 5000 steps
                if step % 5000 == 0 and step > 0 and hasattr(self, 'tb_logger'):
                    try:
                        with torch.no_grad():
                            sample_words = ['hello', 'world', 'test']
                            sample_trajectories = []
                            
                            for word in sample_words:
                                # Create word encoding
                                word_tensor = torch.zeros(1, len(word), dtype=torch.long, device=self.device)
                                for i, char in enumerate(word.lower()):
                                    if ord(char) >= ord('a') and ord(char) <= ord('z'):
                                        word_tensor[0, i] = ord(char) - ord('a') + 1
                                
                                # Generate trajectory
                                noise = torch.randn(1, self.generator.latent_dim, device=self.device)
                                fake_traj = self.generator(noise, word_tensor)
                                
                                # Convert to trajectory format
                                trajectory = []
                                traj_np = fake_traj[0].cpu().numpy()
                                for i, point in enumerate(traj_np):
                                    trajectory.append({
                                        'x': float(point[0]) * 400 + 200,
                                        'y': float(point[1]) * 300 + 150,
                                        't': i * 16.67,
                                        'p': 1 if point[2] > 0.5 else 0
                                    })
                                
                                sample_trajectories.append(trajectory)
                            
                            self.tb_logger.log_trajectory_visualization(
                                sample_trajectories,
                                sample_words,
                                step,
                                f"Generated_Trajectories_Stage_{current_stage}"
                            )
                            
                    except Exception as e:
                        print(f"Failed to log sample trajectories: {e}")
                
                step += 1
                stage_step += 1
                
                # Check for stage progression
                if not in_transition:
                    # Check if current stage is complete
                    if stage_step >= self.stage_lengths[current_stage]:
                        if current_stage < len(self.transition_lengths):
                            print(f"Starting transition from stage {current_stage} to {current_stage + 1}")
                            in_transition = True
                            stage_step = 0
                        else:
                            # Final stage completed
                            break
                else:
                    # Check if transition is complete
                    if stage_step >= self.transition_lengths[current_stage]:
                        print(f"Completed transition to stage {current_stage + 1}")
                        current_stage += 1
                        in_transition = False
                        stage_step = 0
        
        print("Progressive GAN training completed!")
        
        # Close Tensorboard logger
        if hasattr(self, 'tb_logger'):
            # Log final hyperparameters and metrics
            hparams = {
                'learning_rate': self.g_optimizer.param_groups[0]['lr'],
                'total_steps': step,
                'final_stage': current_stage,
                'gradient_penalty_lambda': self.gradient_penalty_lambda,
                'drift_loss_lambda': self.drift_loss_lambda
            }
            
            final_metrics = {
                'final_d_loss': np.mean(self.history['d_loss'][-100:]) if self.history['d_loss'] else 0,
                'final_g_loss': np.mean(self.history['g_loss'][-100:]) if self.history['g_loss'] else 0,
                'final_gp_loss': np.mean(self.history['gp_loss'][-100:]) if self.history['gp_loss'] else 0,
                'stages_completed': current_stage + 1
            }
            
            self.tb_logger.log_hyperparameters(hparams, final_metrics)
            self.tb_logger.close()
        
        return self.history

def test_progressive_gan():
    """Test progressive GAN implementation"""
    print("Testing Progressive GAN...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Model parameters
    vocab_size = 50
    latent_dim = 128
    max_length = 32  # Smaller for testing
    
    # Create models
    generator = ProgressiveGenerator(
        vocab_size=vocab_size,
        latent_dim=latent_dim,
        max_trajectory_length=max_length
    )
    
    discriminator = ProgressiveDiscriminator(
        vocab_size=vocab_size,
        max_trajectory_length=max_length
    )
    
    print(f"Generator parameters: {sum(p.numel() for p in generator.parameters()):,}")
    print(f"Discriminator parameters: {sum(p.numel() for p in discriminator.parameters()):,}")
    
    # Test at different stages
    batch_size = 4
    word = torch.randint(0, vocab_size, (batch_size, 5))
    
    for stage in range(generator.num_stages):
        current_length = 4 * (2 ** stage)
        if current_length > max_length:
            break
        
        print(f"\nTesting stage {stage} (length {current_length}):")
        
        # Set stage
        generator.set_stage(stage, alpha=1.0)
        discriminator.set_stage(stage, alpha=1.0)
        
        # Test generation
        noise = torch.randn(batch_size, latent_dim)
        fake_traj = generator(noise, word)
        print(f"  Generated shape: {fake_traj.shape}")
        
        # Test discrimination
        real_traj = torch.randn(batch_size, current_length, 3)
        real_output = discriminator(real_traj, word)
        fake_output = discriminator(fake_traj, word)
        print(f"  Discriminator outputs: real={real_output.mean():.3f}, fake={fake_output.mean():.3f}")
        
        # Test trainer
        trainer = ProgressiveGANTrainer(generator, discriminator, str(device))
        losses = trainer.train_step(real_traj, word, n_critic=1)
        print(f"  Training losses: D={losses['d_loss']:.4f}, G={losses['g_loss']:.4f}")
    
    print("\n✅ Progressive GAN test completed!")
    return generator, discriminator

if __name__ == "__main__":
    generator, discriminator = test_progressive_gan()