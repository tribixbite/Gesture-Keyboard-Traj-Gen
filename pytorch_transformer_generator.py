#!/usr/bin/env python
"""
PyTorch Transformer implementation for gesture trajectory generation
Based on "Attention is All You Need" with adaptations for trajectory synthesis
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math
import time
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import pandas as pd

class PositionalEncoding(nn.Module):
    """Positional encoding for transformer sequences"""
    
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 1000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input embeddings"""
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class MultiHeadSelfAttention(nn.Module):
    """Multi-head self-attention mechanism"""
    
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def scaled_dot_product_attention(self, query: torch.Tensor, 
                                   key: torch.Tensor,
                                   value: torch.Tensor,
                                   mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute scaled dot-product attention"""
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        output = torch.matmul(attention_weights, value)
        return output, attention_weights
    
    def forward(self, x: torch.Tensor, 
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass through multi-head attention"""
        batch_size, seq_len, d_model = x.size()
        
        # Linear projections
        Q = self.w_q(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        # Apply attention
        attention_output, _ = self.scaled_dot_product_attention(Q, K, V, mask)
        
        # Concatenate heads
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, d_model
        )
        
        # Final linear projection
        output = self.w_o(attention_output)
        return output

class TransformerEncoderLayer(nn.Module):
    """Single transformer encoder layer"""
    
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        
        self.self_attention = MultiHeadSelfAttention(d_model, n_heads, dropout)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, 
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass through encoder layer"""
        # Self-attention with residual connection
        attention_output = self.self_attention(x, mask)
        x = self.norm1(x + self.dropout(attention_output))
        
        # Feed-forward with residual connection
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x

class TransformerDecoderLayer(nn.Module):
    """Single transformer decoder layer"""
    
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        
        self.self_attention = MultiHeadSelfAttention(d_model, n_heads, dropout)
        self.cross_attention = MultiHeadSelfAttention(d_model, n_heads, dropout)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, 
                encoder_output: torch.Tensor,
                tgt_mask: Optional[torch.Tensor] = None,
                memory_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass through decoder layer"""
        # Self-attention with causal mask
        self_attn_output = self.self_attention(x, tgt_mask)
        x = self.norm1(x + self.dropout(self_attn_output))
        
        # Cross-attention to encoder output
        cross_attn_output = self.cross_attention_forward(x, encoder_output, memory_mask)
        x = self.norm2(x + self.dropout(cross_attn_output))
        
        # Feed-forward
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        
        return x
    
    def cross_attention_forward(self, query: torch.Tensor,
                              key_value: torch.Tensor,
                              mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Cross-attention between decoder and encoder"""
        batch_size, seq_len, d_model = query.size()
        n_heads = self.cross_attention.n_heads
        d_k = self.cross_attention.d_k
        
        Q = self.cross_attention.w_q(query).view(batch_size, seq_len, n_heads, d_k).transpose(1, 2)
        K = self.cross_attention.w_k(key_value).view(batch_size, -1, n_heads, d_k).transpose(1, 2)
        V = self.cross_attention.w_v(key_value).view(batch_size, -1, n_heads, d_k).transpose(1, 2)
        
        attention_output, _ = self.cross_attention.scaled_dot_product_attention(Q, K, V, mask)
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, d_model
        )
        
        return self.cross_attention.w_o(attention_output)

class TrajectoryTransformer(nn.Module):
    """Complete transformer model for trajectory generation"""
    
    def __init__(self, 
                 vocab_size: int,
                 d_model: int = 256,
                 n_heads: int = 8,
                 n_encoder_layers: int = 3,
                 n_decoder_layers: int = 3,
                 d_ff: int = 1024,
                 max_seq_len: int = 200,
                 num_mixture_components: int = 20,
                 dropout: float = 0.1):
        super().__init__()
        
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.num_mixture_components = num_mixture_components
        
        # Character embedding
        self.char_embedding = nn.Embedding(vocab_size, d_model)
        
        # Trajectory input projection
        self.traj_projection = nn.Linear(3, d_model)  # x, y, pen_state
        
        # Positional encodings
        self.pos_encoding = PositionalEncoding(d_model, dropout, max_seq_len)
        
        # Encoder layers
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_encoder_layers)
        ])
        
        # Decoder layers
        self.decoder_layers = nn.ModuleList([
            TransformerDecoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_decoder_layers)
        ])
        
        # Output projection to mixture parameters
        self.output_projection = nn.Linear(d_model, 6 * num_mixture_components + 1)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.xavier_uniform_(module.weight)
    
    def create_causal_mask(self, size: int, device: torch.device) -> torch.Tensor:
        """Create causal (upper triangular) mask"""
        mask = torch.triu(torch.ones(size, size, device=device), diagonal=1)
        return mask == 0  # Convert to boolean mask (True where attention is allowed)
    
    def forward(self, 
                char_sequence: torch.Tensor,
                trajectory: torch.Tensor,
                char_mask: Optional[torch.Tensor] = None,
                traj_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through transformer
        
        Args:
            char_sequence: Character indices [batch, char_len]
            trajectory: Input trajectory points [batch, traj_len, 3]
            char_mask: Character sequence mask [batch, char_len]
            traj_mask: Trajectory mask [batch, traj_len]
            
        Returns:
            Mixture parameters [batch, traj_len, 6*components+1]
        """
        batch_size, char_len = char_sequence.size()
        traj_len = trajectory.size(1)
        device = char_sequence.device
        
        # Encode character sequence
        char_emb = self.char_embedding(char_sequence) * math.sqrt(self.d_model)
        char_emb = self.pos_encoding(char_emb.transpose(0, 1)).transpose(0, 1)
        
        # Pass through encoder layers
        encoder_output = char_emb
        for encoder_layer in self.encoder_layers:
            encoder_output = encoder_layer(encoder_output, char_mask)
        
        # Prepare trajectory input
        traj_emb = self.traj_projection(trajectory) * math.sqrt(self.d_model)
        traj_emb = self.pos_encoding(traj_emb.transpose(0, 1)).transpose(0, 1)
        
        # Create causal mask for decoder
        causal_mask = self.create_causal_mask(traj_len, device)
        if traj_mask is not None:
            causal_mask = causal_mask & traj_mask.unsqueeze(1) & traj_mask.unsqueeze(2)
        
        # Pass through decoder layers
        decoder_output = traj_emb
        for decoder_layer in self.decoder_layers:
            decoder_output = decoder_layer(
                decoder_output, encoder_output, causal_mask, char_mask
            )
        
        # Project to mixture parameters
        mixture_params = self.output_projection(decoder_output)
        
        return mixture_params
    
    def generate(self, 
                 char_sequence: torch.Tensor,
                 max_length: int = 150,
                 temperature: float = 1.0,
                 start_token: torch.Tensor = None) -> torch.Tensor:
        """
        Generate trajectory autoregressively
        
        Args:
            char_sequence: Character indices [1, char_len]
            max_length: Maximum trajectory length
            temperature: Sampling temperature
            start_token: Initial trajectory point [1, 1, 3]
            
        Returns:
            Generated trajectory [1, gen_len, 3]
        """
        self.eval()
        with torch.no_grad():
            device = char_sequence.device
            batch_size = 1
            
            # Encode character sequence
            char_emb = self.char_embedding(char_sequence) * math.sqrt(self.d_model)
            char_emb = self.pos_encoding(char_emb.transpose(0, 1)).transpose(0, 1)
            
            encoder_output = char_emb
            for encoder_layer in self.encoder_layers:
                encoder_output = encoder_layer(encoder_output)
            
            # Initialize trajectory
            if start_token is None:
                generated = torch.zeros(batch_size, 1, 3, device=device)
                generated[0, 0, 2] = 1.0  # Pen down
            else:
                generated = start_token
            
            for step in range(max_length - 1):
                # Prepare current trajectory input
                traj_emb = self.traj_projection(generated) * math.sqrt(self.d_model)
                traj_emb = self.pos_encoding(traj_emb.transpose(0, 1)).transpose(0, 1)
                
                # Create causal mask
                traj_len = generated.size(1)
                causal_mask = self.create_causal_mask(traj_len, device)
                
                # Pass through decoder
                decoder_output = traj_emb
                for decoder_layer in self.decoder_layers:
                    decoder_output = decoder_layer(
                        decoder_output, encoder_output, causal_mask
                    )
                
                # Get mixture parameters for last position
                mixture_params = self.output_projection(decoder_output[:, -1:, :])
                
                # Sample from mixture
                next_point = self.sample_from_mixture(mixture_params, temperature)
                
                # Append to generated sequence
                generated = torch.cat([generated, next_point], dim=1)
                
                # Check for end of stroke
                if next_point[0, 0, 2] > 0.5:  # Pen up
                    break
            
            return generated
    
    def sample_from_mixture(self, 
                           mixture_params: torch.Tensor,
                           temperature: float = 1.0) -> torch.Tensor:
        """Sample point from mixture distribution"""
        batch_size = mixture_params.size(0)
        device = mixture_params.device
        
        # Parse mixture parameters
        n_components = self.num_mixture_components
        pi_params = mixture_params[:, :, :n_components]
        mu_params = mixture_params[:, :, n_components:3*n_components]
        sigma_params = mixture_params[:, :, 3*n_components:5*n_components]
        rho_params = mixture_params[:, :, 5*n_components:6*n_components]
        eos_params = mixture_params[:, :, -1:]
        
        # Apply activations
        pi = F.softmax(pi_params / temperature, dim=-1)
        mu_x, mu_y = torch.chunk(mu_params, 2, dim=-1)
        sigma_x, sigma_y = torch.chunk(sigma_params, 2, dim=-1)
        sigma_x = torch.exp(sigma_x) * temperature
        sigma_y = torch.exp(sigma_y) * temperature
        rho = torch.tanh(rho_params)
        eos = torch.sigmoid(eos_params / temperature)
        
        # Sample component index
        component_dist = torch.distributions.Categorical(pi.squeeze(1))
        component_idx = component_dist.sample()
        
        # Get selected parameters
        batch_indices = torch.arange(batch_size, device=device)
        selected_mu_x = mu_x.squeeze(1)[batch_indices, component_idx]
        selected_mu_y = mu_y.squeeze(1)[batch_indices, component_idx]
        selected_sigma_x = sigma_x.squeeze(1)[batch_indices, component_idx]
        selected_sigma_y = sigma_y.squeeze(1)[batch_indices, component_idx]
        selected_rho = rho.squeeze(1)[batch_indices, component_idx]
        
        # Sample from bivariate normal
        z1 = torch.randn(batch_size, device=device)
        z2 = torch.randn(batch_size, device=device)
        
        x = selected_mu_x + selected_sigma_x * z1
        y = selected_mu_y + selected_sigma_y * (
            selected_rho * z1 + torch.sqrt(1 - selected_rho**2) * z2
        )
        
        # Sample pen state
        pen_dist = torch.distributions.Bernoulli(eos.squeeze())
        pen = pen_dist.sample().float()
        
        # Ensure all tensors have same shape
        if pen.dim() == 0:
            pen = pen.unsqueeze(0)
        
        return torch.stack([x, y, pen], dim=-1).unsqueeze(1)

class TransformerTrainer:
    """Training pipeline for transformer model"""
    
    def __init__(self, 
                 model: TrajectoryTransformer,
                 device: str = 'cpu',
                 learning_rate: float = 0.0001):
        
        self.model = model.to(device)
        self.device = device
        
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            betas=(0.9, 0.98),
            eps=1e-9,
            weight_decay=0.01
        )
        
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=3
        )
        
    def mixture_loss(self, 
                    predictions: torch.Tensor,
                    targets: torch.Tensor,
                    target_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute negative log-likelihood loss for mixture density network"""
        
        batch_size, seq_len, _ = targets.size()
        n_components = self.model.num_mixture_components
        
        # Parse predictions
        pi_logits = predictions[:, :, :n_components]
        mu_params = predictions[:, :, n_components:3*n_components]
        sigma_params = predictions[:, :, 3*n_components:5*n_components]
        rho_params = predictions[:, :, 5*n_components:6*n_components]
        eos_logits = predictions[:, :, -1:]
        
        # Apply activations
        pi = F.softmax(pi_logits, dim=-1)
        mu_x, mu_y = torch.chunk(mu_params, 2, dim=-1)
        sigma_x = torch.exp(sigma_params[:, :, :n_components]) + 1e-6
        sigma_y = torch.exp(sigma_params[:, :, n_components:]) + 1e-6
        rho = torch.tanh(rho_params)
        eos_probs = torch.sigmoid(eos_logits)
        
        # Target coordinates
        target_x = targets[:, :, 0].unsqueeze(-1)  # [batch, seq, 1]
        target_y = targets[:, :, 1].unsqueeze(-1)
        target_eos = targets[:, :, 2].unsqueeze(-1)
        
        # Compute 2D Gaussian log-likelihood for each component
        dx = target_x - mu_x
        dy = target_y - mu_y
        
        z = (dx**2)/(sigma_x**2) + (dy**2)/(sigma_y**2) - 2*rho*dx*dy/(sigma_x*sigma_y)
        norm_factor = 2*math.pi*sigma_x*sigma_y*torch.sqrt(1 - rho**2 + 1e-6)
        log_prob_gauss = -0.5*z/(1 - rho**2 + 1e-6) - torch.log(norm_factor)
        
        # Mixture log-likelihood
        log_mixture = torch.logsumexp(torch.log(pi + 1e-8) + log_prob_gauss, dim=-1)
        
        # End-of-stroke loss (ensure targets are in [0, 1])
        target_eos_clamped = torch.clamp(target_eos.squeeze(-1), 0.0, 1.0)
        eos_loss = F.binary_cross_entropy(eos_probs.squeeze(-1), target_eos_clamped, reduction='none')
        
        # Combined loss
        total_loss = -log_mixture + eos_loss
        
        # Apply mask if provided
        if target_mask is not None:
            total_loss = total_loss * target_mask
            return total_loss.sum() / target_mask.sum()
        else:
            return total_loss.mean()
    
    def train_step(self, 
                   char_sequence: torch.Tensor,
                   trajectory: torch.Tensor,
                   char_mask: Optional[torch.Tensor] = None,
                   traj_mask: Optional[torch.Tensor] = None) -> float:
        """Single training step"""
        
        self.model.train()
        self.optimizer.zero_grad()
        
        # Teacher forcing: use ground truth as input, predict next step
        input_traj = trajectory[:, :-1, :]
        target_traj = trajectory[:, 1:, :]
        
        if traj_mask is not None:
            input_mask = traj_mask[:, :-1]
            target_mask = traj_mask[:, 1:]
        else:
            input_mask = None
            target_mask = None
        
        # Forward pass
        predictions = self.model(
            char_sequence, input_traj, char_mask, input_mask
        )
        
        # Compute loss
        loss = self.mixture_loss(predictions, target_traj, target_mask)
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        
        return loss.item()

# Dataset class for transformer training
class TransformerDataset(Dataset):
    """Dataset for transformer trajectory generation"""
    
    def __init__(self, data_dir: str, max_traj_length: int = 150, max_word_length: int = 20):
        self.data_dir = Path(data_dir)
        self.max_traj_length = max_traj_length
        self.max_word_length = max_word_length
        
        self.trajectories = []
        self.words = []
        self.char_to_idx = {}
        
        self._load_data()
        self._build_vocab()
    
    def _load_data(self):
        """Load trajectory data from swipelog files"""
        for log_file in self.data_dir.glob("*.log"):
            try:
                df = pd.read_csv(log_file, sep=r'\s+', engine='python',
                               on_bad_lines='skip', header=0)
                
                if 'word' in df.columns and 'x_pos' in df.columns:
                    for word in df['word'].unique():
                        if pd.isna(word) or word == 'word':
                            continue
                        
                        word_df = df[df['word'] == word]
                        traj = word_df[['x_pos', 'y_pos']].values
                        
                        if len(traj) > 2 and len(word) <= self.max_word_length:
                            # Normalize trajectory
                            traj = (traj - traj.mean(0)) / (traj.std(0) + 1e-6)
                            self.trajectories.append(traj)
                            self.words.append(word.lower())
            except:
                continue
    
    def _build_vocab(self):
        """Build character vocabulary"""
        chars = set()
        for word in self.words:
            chars.update(word)
        
        # Add special tokens
        self.char_to_idx = {'<PAD>': 0, '<SOS>': 1, '<EOS>': 2}
        for i, char in enumerate(sorted(chars)):
            self.char_to_idx[char] = i + 3
        
        self.idx_to_char = {idx: char for char, idx in self.char_to_idx.items()}
        self.vocab_size = len(self.char_to_idx)
    
    def __len__(self):
        return len(self.trajectories)
    
    def __getitem__(self, idx):
        trajectory = self.trajectories[idx]
        word = self.words[idx]
        
        # Prepare trajectory with pen states
        traj_len = min(len(trajectory), self.max_traj_length - 1)
        traj_tensor = torch.zeros(self.max_traj_length, 3)
        
        # Add trajectory points
        traj_tensor[:traj_len, :2] = torch.FloatTensor(trajectory[:traj_len])
        traj_tensor[:traj_len, 2] = 1.0  # Pen down
        traj_tensor[traj_len, 2] = 0.0   # Pen up (end marker)
        
        # Prepare character sequence
        char_seq = torch.full((self.max_word_length,), self.char_to_idx['<PAD>'], dtype=torch.long)
        char_seq[0] = self.char_to_idx['<SOS>']
        
        for i, char in enumerate(word):
            if i + 1 < self.max_word_length - 1:
                char_seq[i + 1] = self.char_to_idx.get(char, self.char_to_idx['<PAD>'])
        
        if len(word) + 2 < self.max_word_length:
            char_seq[len(word) + 1] = self.char_to_idx['<EOS>']
        
        return traj_tensor, char_seq

def test_transformer():
    """Test transformer implementation"""
    print("Testing Transformer Model...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Model parameters
    vocab_size = 50
    batch_size = 4
    char_len = 10
    traj_len = 80
    
    # Create model
    model = TrajectoryTransformer(
        vocab_size=vocab_size,
        d_model=128,
        n_heads=4,
        n_encoder_layers=2,
        n_decoder_layers=2,
        d_ff=256,
        num_mixture_components=10
    ).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params:,}")
    
    # Test data
    char_seq = torch.randint(0, vocab_size, (batch_size, char_len)).to(device)
    trajectory = torch.randn(batch_size, traj_len, 3).to(device)
    
    # Forward pass
    start_time = time.time()
    with torch.no_grad():
        output = model(char_seq, trajectory)
    forward_time = time.time() - start_time
    
    print(f"Forward pass time: {forward_time:.3f}s")
    print(f"Output shape: {output.shape}")
    print(f"Expected shape: ({batch_size}, {traj_len}, {6*10+1})")
    
    # Test generation
    print("\nTesting generation...")
    test_char_seq = torch.randint(0, vocab_size, (1, char_len)).to(device)
    
    start_time = time.time()
    generated = model.generate(test_char_seq, max_length=50)
    gen_time = time.time() - start_time
    
    print(f"Generation time: {gen_time:.3f}s")
    print(f"Generated trajectory shape: {generated.shape}")
    
    # Test trainer
    print("\nTesting trainer...")
    trainer = TransformerTrainer(model, str(device))
    
    loss = trainer.train_step(char_seq, trajectory)
    print(f"Training loss: {loss:.4f}")
    
    print("✅ Transformer test completed successfully!")
    
    return model, trainer

if __name__ == "__main__":
    model, trainer = test_transformer()