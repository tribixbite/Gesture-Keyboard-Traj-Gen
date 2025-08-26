#!/usr/bin/env python
"""
Transformer-based trajectory generation model
Using modern transformer architecture for gesture generation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Optional, Tuple

class PositionalEncoding(nn.Module):
    """Positional encoding for transformer"""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]


class TransformerTrajectoryModel(nn.Module):
    """Transformer model for trajectory generation"""
    
    def __init__(self,
                 vocab_size: int,
                 d_model: int = 256,
                 nhead: int = 8,
                 num_encoder_layers: int = 3,
                 num_decoder_layers: int = 3,
                 dim_feedforward: int = 1024,
                 dropout: float = 0.1,
                 max_seq_length: int = 200,
                 num_mixture_components: int = 20):
        super().__init__()
        
        self.d_model = d_model
        self.num_mixture_components = num_mixture_components
        
        # Word embedding
        self.word_embedding = nn.Embedding(vocab_size, d_model)
        self.word_pos_encoding = PositionalEncoding(d_model)
        
        # Trajectory embedding
        self.traj_projection = nn.Linear(3, d_model)
        self.traj_pos_encoding = PositionalEncoding(d_model)
        
        # Transformer
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        
        # Output head for mixture density network
        self.output_projection = nn.Linear(
            d_model,
            6 * num_mixture_components + 1  # GMM params + end-of-stroke
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self,
                src_words: torch.Tensor,
                tgt_trajectories: torch.Tensor,
                src_mask: Optional[torch.Tensor] = None,
                tgt_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            src_words: Source word indices (batch, word_len)
            tgt_trajectories: Target trajectories (batch, seq_len, 3)
            src_mask: Source attention mask
            tgt_mask: Target attention mask
            
        Returns:
            mixture_params: Parameters for mixture density network
        """
        
        # Encode words
        word_emb = self.word_embedding(src_words) * math.sqrt(self.d_model)
        word_emb = self.word_pos_encoding(word_emb)
        
        # Encode trajectories
        traj_emb = self.traj_projection(tgt_trajectories)
        traj_emb = self.traj_pos_encoding(traj_emb)
        
        # Generate target mask (causal)
        if tgt_mask is None:
            seq_len = tgt_trajectories.size(1)
            tgt_mask = self.generate_square_subsequent_mask(seq_len).to(tgt_trajectories.device)
        
        # Transformer
        output = self.transformer(
            src=word_emb,
            tgt=traj_emb,
            src_mask=src_mask,
            tgt_mask=tgt_mask
        )
        
        # Project to mixture parameters
        mixture_params = self.output_projection(output)
        
        return mixture_params
    
    def generate_square_subsequent_mask(self, sz: int) -> torch.Tensor:
        """Generate causal mask for decoder"""
        mask = torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)
        return mask
    
    def generate(self,
                 words: torch.Tensor,
                 max_length: int = 200,
                 temperature: float = 1.0) -> torch.Tensor:
        """
        Generate trajectory for given words
        
        Args:
            words: Word indices (batch, word_len)
            max_length: Maximum trajectory length
            temperature: Sampling temperature
            
        Returns:
            trajectory: Generated trajectory
        """
        
        self.eval()
        with torch.no_grad():
            batch_size = words.size(0)
            device = words.device
            
            # Encode words
            word_emb = self.word_embedding(words) * math.sqrt(self.d_model)
            word_emb = self.word_pos_encoding(word_emb)
            
            # Start with zeros
            trajectory = torch.zeros(batch_size, 1, 3, device=device)
            
            for _ in range(max_length - 1):
                # Encode trajectory so far
                traj_emb = self.traj_projection(trajectory)
                traj_emb = self.traj_pos_encoding(traj_emb)
                
                # Generate next point
                output = self.transformer(
                    src=word_emb,
                    tgt=traj_emb
                )
                
                # Get last timestep
                mixture_params = self.output_projection(output[:, -1:])
                
                # Sample from mixture
                next_point = self.sample_from_mixture(mixture_params, temperature)
                
                # Append
                trajectory = torch.cat([trajectory, next_point], dim=1)
                
                # Check for end of stroke
                if next_point[0, 0, 2] > 0.5:  # End marker
                    break
            
        return trajectory
    
    def sample_from_mixture(self,
                           params: torch.Tensor,
                           temperature: float = 1.0) -> torch.Tensor:
        """Sample point from mixture of Gaussians"""
        
        batch_size = params.size(0)
        m = self.num_mixture_components
        
        # Parse parameters
        pi = params[..., :m]
        mu_x = params[..., m:2*m]
        mu_y = params[..., 2*m:3*m]
        sigma_x = params[..., 3*m:4*m]
        sigma_y = params[..., 4*m:5*m]
        rho = params[..., 5*m:6*m]
        eos = params[..., -1:]
        
        # Apply activations
        pi = F.softmax(pi / temperature, dim=-1)
        sigma_x = F.softplus(sigma_x) * temperature
        sigma_y = F.softplus(sigma_y) * temperature
        rho = torch.tanh(rho)
        eos = torch.sigmoid(eos)
        
        # Sample component
        component = torch.multinomial(pi.squeeze(1), 1)
        
        # Extract selected component parameters
        batch_idx = torch.arange(batch_size, device=params.device)
        mu_x_selected = mu_x.squeeze(1)[batch_idx, component.squeeze()]
        mu_y_selected = mu_y.squeeze(1)[batch_idx, component.squeeze()]
        sigma_x_selected = sigma_x.squeeze(1)[batch_idx, component.squeeze()]
        sigma_y_selected = sigma_y.squeeze(1)[batch_idx, component.squeeze()]
        rho_selected = rho.squeeze(1)[batch_idx, component.squeeze()]
        
        # Sample from 2D Gaussian
        z1 = torch.randn(batch_size, device=params.device)
        z2 = torch.randn(batch_size, device=params.device)
        
        x = mu_x_selected + sigma_x_selected * z1
        y = mu_y_selected + sigma_y_selected * (rho_selected * z1 + 
                                                torch.sqrt(1 - rho_selected**2) * z2)
        
        # Sample pen state
        pen = (torch.rand(batch_size, device=params.device) < eos.squeeze()).float()
        
        return torch.stack([x, y, pen], dim=-1).unsqueeze(1)


class TransformerTrainer:
    """Trainer for transformer model"""
    
    def __init__(self, model: TransformerTrajectoryModel, 
                 device: str = 'cpu',
                 learning_rate: float = 0.0001):
        
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, patience=5, factor=0.5
        )
        
    def mixture_loss(self, params: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Negative log-likelihood loss for mixture density network"""
        
        batch_size, seq_len, _ = targets.size()
        m = self.model.num_mixture_components
        
        # Parse parameters
        pi = params[..., :m]
        mu_x = params[..., m:2*m]
        mu_y = params[..., 2*m:3*m]
        sigma_x = params[..., 3*m:4*m]
        sigma_y = params[..., 4*m:5*m]
        rho = params[..., 5*m:6*m]
        eos = params[..., -1:]
        
        # Apply activations
        pi = F.softmax(pi, dim=-1)
        sigma_x = F.softplus(sigma_x) + 1e-6
        sigma_y = F.softplus(sigma_y) + 1e-6
        rho = torch.tanh(rho)
        eos = torch.sigmoid(eos)
        
        # Target coordinates
        target_x = targets[..., 0:1]
        target_y = targets[..., 1:2]
        target_pen = targets[..., 2:3]
        
        # Compute 2D Gaussian log-likelihood
        z_x = (target_x - mu_x) / sigma_x
        z_y = (target_y - mu_y) / sigma_y
        
        z = z_x**2 + z_y**2 - 2*rho*z_x*z_y
        
        norm = 2 * math.pi * sigma_x * sigma_y * torch.sqrt(1 - rho**2 + 1e-6)
        
        log_probs = -z / (2 * (1 - rho**2 + 1e-6)) - torch.log(norm)
        
        # Mixture log-likelihood
        log_mixture = torch.logsumexp(torch.log(pi + 1e-6) + log_probs, dim=-1)
        
        # Pen state loss (binary cross-entropy)
        target_pen = torch.clamp(target_pen, 0, 1)  # Ensure valid range
        pen_loss = F.binary_cross_entropy(eos, target_pen, reduction='none').squeeze(-1)
        
        # Total loss
        total_loss = -log_mixture + pen_loss
        
        return total_loss.mean()
    
    def train_step(self, words: torch.Tensor, trajectories: torch.Tensor) -> float:
        """Single training step"""
        
        self.model.train()
        self.optimizer.zero_grad()
        
        # Shift trajectories for teacher forcing
        traj_input = trajectories[:, :-1]
        traj_target = trajectories[:, 1:]
        
        # Forward pass
        mixture_params = self.model(words, traj_input)
        
        # Compute loss
        loss = self.mixture_loss(mixture_params, traj_target)
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
        self.optimizer.step()
        
        return loss.item()
    
    def validate(self, val_loader) -> float:
        """Validate model"""
        
        self.model.eval()
        val_losses = []
        
        with torch.no_grad():
            for words, trajectories in val_loader:
                words = words.to(self.device)
                trajectories = trajectories.to(self.device)
                
                traj_input = trajectories[:, :-1]
                traj_target = trajectories[:, 1:]
                
                mixture_params = self.model(words, traj_input)
                loss = self.mixture_loss(mixture_params, traj_target)
                
                val_losses.append(loss.item())
        
        return np.mean(val_losses)


def test_transformer():
    """Test transformer implementation"""
    
    print("Testing Transformer Model")
    print("="*50)
    
    # Create model
    model = TransformerTrajectoryModel(
        vocab_size=50,
        d_model=256,
        nhead=8,
        num_encoder_layers=3,
        num_decoder_layers=3
    )
    
    # Test data
    batch_size = 4
    word_len = 5
    seq_len = 100
    
    words = torch.randint(0, 50, (batch_size, word_len))
    trajectories = torch.randn(batch_size, seq_len, 3)
    
    # Forward pass
    output = model(words, trajectories)
    print(f"✅ Forward pass: output shape {output.shape}")
    
    # Generation
    generated = model.generate(words, max_length=150)
    print(f"✅ Generation: trajectory shape {generated.shape}")
    
    # Parameter count
    total_params = sum(p.numel() for p in model.parameters())
    print(f"✅ Total parameters: {total_params:,}")
    
    # Test trainer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainer = TransformerTrainer(model, device)
    
    loss = trainer.train_step(words, trajectories)
    print(f"✅ Training step: loss = {loss:.4f}")
    
    print("\n✨ Transformer model successfully implemented!")
    
    return model


if __name__ == "__main__":
    model = test_transformer()