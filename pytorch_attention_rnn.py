#!/usr/bin/env python
"""
PyTorch implementation of RNN with attention mechanism
Extracted and adapted from RNN-Based TF2 implementation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, NamedTuple

class AttentionState(NamedTuple):
    """State for LSTM with attention"""
    h1: torch.Tensor
    c1: torch.Tensor
    h2: torch.Tensor
    c2: torch.Tensor
    h3: torch.Tensor
    c3: torch.Tensor
    alpha: torch.Tensor
    beta: torch.Tensor
    kappa: torch.Tensor
    w: torch.Tensor
    phi: torch.Tensor

class LSTMAttentionCell(nn.Module):
    """LSTM cell with attention mechanism for gesture generation"""
    
    def __init__(self, 
                 input_size: int,
                 hidden_size: int = 256,
                 num_attn_mixtures: int = 10,
                 num_output_mixtures: int = 20,
                 window_size: int = 128):
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_attn_mixtures = num_attn_mixtures
        self.num_output_mixtures = num_output_mixtures
        self.window_size = window_size
        
        # LSTM cells
        self.lstm1 = nn.LSTMCell(input_size + window_size, hidden_size)
        self.lstm2 = nn.LSTMCell(hidden_size + window_size, hidden_size)
        self.lstm3 = nn.LSTMCell(hidden_size + window_size, hidden_size)
        
        # Attention parameters
        self.attention_layer = nn.Linear(
            input_size + hidden_size + window_size,
            3 * num_attn_mixtures
        )
        
        # Output mixture parameters
        self.output_layer = nn.Linear(
            hidden_size,
            6 * num_output_mixtures + 1  # pi, mu_x, mu_y, sigma_x, sigma_y, rho, eos
        )
        
    def forward(self, 
                x: torch.Tensor,
                state: Optional[AttentionState] = None,
                attention_values: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, AttentionState]:
        """
        Forward pass through attention LSTM
        
        Args:
            x: Input tensor (batch_size, input_size)
            state: Previous state
            attention_values: Character encodings (batch_size, seq_len, encoding_size)
            attention_mask: Mask for attention values
            
        Returns:
            output: Mixture parameters
            new_state: Updated state
        """
        batch_size = x.size(0)
        device = x.device
        
        # Initialize state if needed
        if state is None:
            state = self.init_state(batch_size, device)
        
        # LSTM 1
        lstm1_input = torch.cat([state.w, x], dim=1)
        h1, c1 = self.lstm1(lstm1_input, (state.h1, state.c1))
        
        # Attention mechanism
        if attention_values is not None:
            attention_out, alpha, beta, kappa, phi, w = self.compute_attention(
                x, h1, state.w, state.kappa,
                attention_values, attention_mask
            )
        else:
            w = state.w
            alpha, beta, kappa = state.alpha, state.beta, state.kappa
            phi = state.phi
        
        # LSTM 2
        lstm2_input = torch.cat([h1, w], dim=1)
        h2, c2 = self.lstm2(lstm2_input, (state.h2, state.c2))
        
        # LSTM 3
        lstm3_input = torch.cat([h2, w], dim=1)
        h3, c3 = self.lstm3(lstm3_input, (state.h3, state.c3))
        
        # Output mixture parameters
        output = self.output_layer(h3)
        
        # Create new state
        new_state = AttentionState(
            h1=h1, c1=c1,
            h2=h2, c2=c2,
            h3=h3, c3=c3,
            alpha=alpha, beta=beta, kappa=kappa,
            w=w, phi=phi
        )
        
        return output, new_state
    
    def compute_attention(self, 
                         x: torch.Tensor,
                         h: torch.Tensor,
                         prev_w: torch.Tensor,
                         prev_kappa: torch.Tensor,
                         attention_values: torch.Tensor,
                         attention_mask: Optional[torch.Tensor] = None):
        """Compute attention weights and context vector"""
        
        batch_size = x.size(0)
        seq_len = attention_values.size(1)
        
        # Compute attention parameters
        attention_input = torch.cat([prev_w, x, h], dim=1)
        attention_params = self.attention_layer(attention_input)
        
        # Split and process parameters
        alpha, beta, kappa_increment = torch.split(
            attention_params, self.num_attn_mixtures, dim=1
        )
        
        # Apply transformations
        alpha = F.softplus(alpha)
        beta = F.softplus(beta).clamp(min=0.01)
        kappa = prev_kappa + F.softplus(kappa_increment) / 25.0
        
        # Expand for broadcasting
        alpha = alpha.unsqueeze(2)  # (batch, mixtures, 1)
        beta = beta.unsqueeze(2)
        kappa_expanded = kappa.unsqueeze(2)
        
        # Create position grid
        positions = torch.arange(seq_len, dtype=torch.float32, device=x.device)
        positions = positions.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len)
        positions = positions.expand(batch_size, self.num_attn_mixtures, -1)
        
        # Compute Gaussian attention weights
        phi = alpha * torch.exp(-torch.square(kappa_expanded - positions) / beta)
        phi = phi.sum(dim=1)  # (batch, seq_len)
        
        # Apply mask if provided
        if attention_mask is not None:
            phi = phi * attention_mask
        
        # Normalize
        phi = phi / (phi.sum(dim=1, keepdim=True) + 1e-8)
        
        # Compute context vector
        phi_expanded = phi.unsqueeze(2)  # (batch, seq_len, 1)
        w = (phi_expanded * attention_values).sum(dim=1)
        
        return attention_params, alpha.squeeze(2), beta.squeeze(2), kappa, phi, w
    
    def init_state(self, batch_size: int, device: torch.device) -> AttentionState:
        """Initialize attention state"""
        return AttentionState(
            h1=torch.zeros(batch_size, self.hidden_size, device=device),
            c1=torch.zeros(batch_size, self.hidden_size, device=device),
            h2=torch.zeros(batch_size, self.hidden_size, device=device),
            c2=torch.zeros(batch_size, self.hidden_size, device=device),
            h3=torch.zeros(batch_size, self.hidden_size, device=device),
            c3=torch.zeros(batch_size, self.hidden_size, device=device),
            alpha=torch.zeros(batch_size, self.num_attn_mixtures, device=device),
            beta=torch.zeros(batch_size, self.num_attn_mixtures, device=device),
            kappa=torch.zeros(batch_size, self.num_attn_mixtures, device=device),
            w=torch.zeros(batch_size, self.window_size, device=device),
            phi=torch.zeros(batch_size, 1, device=device)  # Placeholder
        )


class AttentionRNN(nn.Module):
    """Full RNN model with attention for trajectory generation"""
    
    def __init__(self,
                 vocab_size: int,
                 embedding_size: int = 128,
                 hidden_size: int = 256,
                 num_attn_mixtures: int = 10,
                 num_output_mixtures: int = 20,
                 dropout: float = 0.2):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.num_output_mixtures = num_output_mixtures
        
        # Character embedding
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.dropout = nn.Dropout(dropout)
        
        # Attention LSTM cell
        self.attention_cell = LSTMAttentionCell(
            input_size=3,  # x, y, pen_state
            hidden_size=hidden_size,
            num_attn_mixtures=num_attn_mixtures,
            num_output_mixtures=num_output_mixtures,
            window_size=embedding_size
        )
        
    def forward(self, 
                trajectories: torch.Tensor,
                words: torch.Tensor,
                word_lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through attention RNN
        
        Args:
            trajectories: Input trajectories (batch, seq_len, 3)
            words: Word indices (batch, word_len)
            word_lengths: Actual lengths of words
            
        Returns:
            mixture_params: Parameters for mixture density network
        """
        batch_size, seq_len, _ = trajectories.size()
        
        # Embed words for attention
        word_embeddings = self.embedding(words)
        word_embeddings = self.dropout(word_embeddings)
        
        # Create attention mask if word lengths provided
        if word_lengths is not None:
            max_word_len = words.size(1)
            mask = torch.arange(max_word_len, device=words.device)[None, :] < word_lengths[:, None]
            mask = mask.float()
        else:
            mask = None
        
        # Process sequence through attention LSTM
        outputs = []
        state = None
        
        for t in range(seq_len):
            x_t = trajectories[:, t, :]
            output, state = self.attention_cell(
                x_t, state, word_embeddings, mask
            )
            outputs.append(output)
        
        # Stack outputs
        outputs = torch.stack(outputs, dim=1)
        
        return outputs
    
    def generate(self,
                 word: torch.Tensor,
                 max_length: int = 200,
                 temperature: float = 1.0) -> torch.Tensor:
        """Generate trajectory for given word"""
        
        self.eval()
        with torch.no_grad():
            batch_size = word.size(0)
            device = word.device
            
            # Embed word
            word_embedding = self.embedding(word)
            
            # Initialize
            trajectory = []
            state = None
            x = torch.zeros(batch_size, 3, device=device)
            
            for _ in range(max_length):
                # Forward through attention cell
                output, state = self.attention_cell(
                    x, state, word_embedding
                )
                
                # Sample from mixture
                x = self.sample_from_mixture(output, temperature)
                trajectory.append(x)
                
                # Check for end of stroke
                if x[:, 2].mean() > 0.5:  # Average pen-up
                    break
            
            trajectory = torch.stack(trajectory, dim=1)
            
        return trajectory
    
    def sample_from_mixture(self, 
                           params: torch.Tensor,
                           temperature: float = 1.0) -> torch.Tensor:
        """Sample point from mixture of Gaussians"""
        
        batch_size = params.size(0)
        num_mixtures = self.num_output_mixtures
        
        # Parse parameters
        pi, mu_x, mu_y, sigma_x, sigma_y, rho, eos = torch.split(
            params,
            [num_mixtures, num_mixtures, num_mixtures, 
             num_mixtures, num_mixtures, num_mixtures, 1],
            dim=-1
        )
        
        # Apply activations
        pi = F.softmax(pi / temperature, dim=-1)
        sigma_x = torch.exp(sigma_x) * temperature
        sigma_y = torch.exp(sigma_y) * temperature
        rho = torch.tanh(rho)
        eos = torch.sigmoid(eos)
        
        # Sample component
        component = torch.multinomial(pi, 1).squeeze(-1)
        
        # Get selected component parameters
        indices = torch.arange(batch_size, device=params.device)
        selected_mu_x = mu_x[indices, component]
        selected_mu_y = mu_y[indices, component]
        selected_sigma_x = sigma_x[indices, component]
        selected_sigma_y = sigma_y[indices, component]
        selected_rho = rho[indices, component]
        
        # Sample from 2D Gaussian
        z1 = torch.randn(batch_size, device=params.device)
        z2 = torch.randn(batch_size, device=params.device)
        
        x = selected_mu_x + selected_sigma_x * z1
        y = selected_mu_y + selected_sigma_y * (selected_rho * z1 + 
                                                 torch.sqrt(1 - selected_rho**2) * z2)
        
        # Sample pen state
        pen = (torch.rand(batch_size, device=params.device) < eos.squeeze()).float()
        
        return torch.stack([x, y, pen], dim=-1)


def test_attention_rnn():
    """Test the attention RNN implementation"""
    
    print("Testing Attention RNN Implementation")
    print("="*50)
    
    # Create model
    model = AttentionRNN(
        vocab_size=50,
        embedding_size=128,
        hidden_size=256,
        num_attn_mixtures=10,
        num_output_mixtures=20
    )
    
    # Test data
    batch_size = 4
    seq_len = 100
    word_len = 5
    
    trajectories = torch.randn(batch_size, seq_len, 3)
    words = torch.randint(0, 50, (batch_size, word_len))
    word_lengths = torch.tensor([3, 4, 5, 2])
    
    # Forward pass
    output = model(trajectories, words, word_lengths)
    print(f"✅ Forward pass: output shape {output.shape}")
    
    # Generation
    test_word = torch.tensor([[1, 2, 3, 0, 0]])  # "hello" encoded
    generated = model.generate(test_word, max_length=150)
    print(f"✅ Generation: trajectory shape {generated.shape}")
    
    # Parameter count
    total_params = sum(p.numel() for p in model.parameters())
    print(f"✅ Total parameters: {total_params:,}")
    
    print("\n✨ Attention RNN successfully implemented!")
    
    return model


if __name__ == "__main__":
    model = test_attention_rnn()