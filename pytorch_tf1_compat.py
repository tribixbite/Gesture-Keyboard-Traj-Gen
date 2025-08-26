#!/usr/bin/env python
"""
PyTorch compatibility layer for TensorFlow 1.x RNN model
Provides equivalent functionality to the TF1 handwriting synthesis model
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math
from typing import Dict, List, Tuple, Optional, NamedTuple
import json
import time

class LSTMState(NamedTuple):
    """LSTM cell state tuple"""
    hidden: torch.Tensor
    cell: torch.Tensor

class AttentionParams(NamedTuple):
    """Attention mechanism parameters"""
    alpha: torch.Tensor  # Mixture weights
    beta: torch.Tensor   # Mixture precisions  
    kappa: torch.Tensor  # Mixture positions
    phi: torch.Tensor    # Attention alignment
    window: torch.Tensor # Attended character vector

class MDNParams(NamedTuple):
    """Mixture Density Network parameters"""
    eos: torch.Tensor      # End of stroke probability
    pi: torch.Tensor       # Mixture weights
    mu_x: torch.Tensor     # X means
    mu_y: torch.Tensor     # Y means
    sigma_x: torch.Tensor  # X standard deviations
    sigma_y: torch.Tensor  # Y standard deviations
    rho: torch.Tensor      # Correlations

class TF1CompatModel(nn.Module):
    """
    PyTorch implementation compatible with TF1 handwriting synthesis model
    Maintains same architecture and parameter structure
    """
    
    def __init__(self, 
                 rnn_size: int = 400,
                 num_mixtures: int = 20,
                 num_window_mixtures: int = 10,
                 alphabet_size: int = 80,
                 dropout: float = 0.9):
        super().__init__()
        
        self.rnn_size = rnn_size
        self.num_mixtures = num_mixtures
        self.num_window_mixtures = num_window_mixtures
        self.alphabet_size = alphabet_size
        self.dropout_prob = 1.0 - dropout
        
        # Three LSTM cells (matching TF1 model)
        self.lstm1 = nn.LSTMCell(3, rnn_size)  # Input: x, y, eos
        self.lstm2 = nn.LSTMCell(rnn_size + alphabet_size + 3, rnn_size)
        self.lstm3 = nn.LSTMCell(rnn_size + alphabet_size + 3, rnn_size)
        
        # Attention window parameters
        self.window_linear = nn.Linear(rnn_size, 3 * num_window_mixtures)
        
        # Mixture Density Network output
        mdn_output_size = 1 + num_mixtures * 6  # eos + (pi, mu_x, mu_y, sigma_x, sigma_y, rho) per mixture
        self.mdn_linear = nn.Linear(rnn_size, mdn_output_size)
        
        # Dropout layers
        self.dropout = nn.Dropout(self.dropout_prob)
        
        # Initialize weights similar to Graves initializer
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights with Graves-style initialization"""
        for name, param in self.named_parameters():
            if 'weight' in name:
                if 'lstm' in name:
                    # LSTM weights
                    nn.init.normal_(param, mean=0.0, std=0.075)
                else:
                    # Linear layer weights
                    nn.init.normal_(param, mean=0.0, std=0.075)
            elif 'bias' in name:
                if 'window_linear' in name:
                    # Special initialization for window bias (kappa bias)
                    nn.init.normal_(param, mean=-3.0, std=0.25)
                else:
                    nn.init.normal_(param, mean=0.0, std=0.075)
    
    def init_states(self, batch_size: int, device: torch.device) -> Tuple[LSTMState, LSTMState, LSTMState]:
        """Initialize LSTM states"""
        zero_state = lambda: LSTMState(
            hidden=torch.zeros(batch_size, self.rnn_size, device=device),
            cell=torch.zeros(batch_size, self.rnn_size, device=device)
        )
        return zero_state(), zero_state(), zero_state()
    
    def init_attention(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Initialize attention kappa"""
        return torch.zeros(batch_size, self.num_window_mixtures, device=device)
    
    def compute_attention(self, 
                         lstm1_output: torch.Tensor,
                         char_sequence: torch.Tensor,
                         prev_kappa: torch.Tensor) -> AttentionParams:
        """
        Compute attention mechanism (Gaussian window)
        
        Args:
            lstm1_output: Output from first LSTM [batch, rnn_size]
            char_sequence: Character sequence [batch, seq_len, alphabet_size]
            prev_kappa: Previous attention position [batch, num_mixtures]
            
        Returns:
            AttentionParams with alpha, beta, kappa, phi, window
        """
        batch_size, seq_len, _ = char_sequence.size()
        
        # Compute attention parameters
        window_params = self.window_linear(lstm1_output)  # [batch, 3*num_mixtures]
        window_params = torch.exp(window_params.view(batch_size, 3, self.num_window_mixtures))
        
        alpha, beta, kappa_increment = window_params.unbind(dim=1)
        
        # Update kappa (cumulative attention position)
        kappa = prev_kappa + kappa_increment
        
        # Compute attention alignment (phi)
        # Create position grid
        u = torch.arange(seq_len, device=char_sequence.device, dtype=torch.float32)
        u = u.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len]
        u = u.expand(batch_size, self.num_window_mixtures, seq_len)
        
        # Expand attention parameters
        alpha_exp = alpha.unsqueeze(2)  # [batch, num_mixtures, 1]
        beta_exp = beta.unsqueeze(2)
        kappa_exp = kappa.unsqueeze(2)
        
        # Compute Gaussian attention weights
        phi_k = alpha_exp * torch.exp(-beta_exp * torch.square(kappa_exp - u))
        phi = torch.sum(phi_k, dim=1)  # [batch, seq_len]
        
        # Compute attended window
        phi_expanded = phi.unsqueeze(2)  # [batch, seq_len, 1]
        window = torch.sum(phi_expanded * char_sequence, dim=1)  # [batch, alphabet_size]
        
        return AttentionParams(
            alpha=alpha,
            beta=beta,
            kappa=kappa,
            phi=phi,
            window=window
        )
    
    def forward(self, 
                trajectory: torch.Tensor,
                char_sequence: torch.Tensor,
                initial_states: Optional[Tuple] = None,
                initial_kappa: Optional[torch.Tensor] = None,
                training: bool = True) -> Tuple[torch.Tensor, Dict]:
        """
        Forward pass through the model
        
        Args:
            trajectory: Input trajectory [batch, seq_len, 3]
            char_sequence: Character sequence [batch, char_len, alphabet_size]
            initial_states: Initial LSTM states
            initial_kappa: Initial attention position
            training: Training mode flag
            
        Returns:
            MDN parameters [batch, seq_len, mdn_output_size]
            Additional outputs (states, attention, etc.)
        """
        batch_size, seq_len, _ = trajectory.size()
        device = trajectory.device
        
        # Initialize states if not provided
        if initial_states is None:
            state1, state2, state3 = self.init_states(batch_size, device)
        else:
            state1, state2, state3 = initial_states
        
        if initial_kappa is None:
            kappa = self.init_attention(batch_size, device)
        else:
            kappa = initial_kappa
        
        outputs = []
        attention_history = []
        
        for t in range(seq_len):
            # Current input
            x_t = trajectory[:, t, :]  # [batch, 3]
            
            # LSTM 1
            if training and self.dropout_prob > 0:
                lstm1_input = self.dropout(x_t)
            else:
                lstm1_input = x_t
                
            hidden1, cell1 = self.lstm1(lstm1_input, (state1.hidden, state1.cell))
            state1 = LSTMState(hidden1, cell1)
            
            if training and self.dropout_prob > 0:
                hidden1 = self.dropout(hidden1)
            
            # Attention mechanism
            attention = self.compute_attention(hidden1, char_sequence, kappa)
            kappa = attention.kappa
            attention_history.append(attention)
            
            # LSTM 2 (with attention window and input)
            lstm2_input = torch.cat([hidden1, attention.window, x_t], dim=1)
            if training and self.dropout_prob > 0:
                lstm2_input = self.dropout(lstm2_input)
                
            hidden2, cell2 = self.lstm2(lstm2_input, (state2.hidden, state2.cell))
            state2 = LSTMState(hidden2, cell2)
            
            if training and self.dropout_prob > 0:
                hidden2 = self.dropout(hidden2)
            
            # LSTM 3 (with attention window and input)
            lstm3_input = torch.cat([hidden2, attention.window, x_t], dim=1)
            if training and self.dropout_prob > 0:
                lstm3_input = self.dropout(lstm3_input)
                
            hidden3, cell3 = self.lstm3(lstm3_input, (state3.hidden, state3.cell))
            state3 = LSTMState(hidden3, cell3)
            
            # MDN output
            mdn_output = self.mdn_linear(hidden3)
            outputs.append(mdn_output)
        
        # Stack outputs
        outputs = torch.stack(outputs, dim=1)  # [batch, seq_len, mdn_output_size]
        
        # Additional outputs for analysis/sampling
        extras = {
            'final_states': (state1, state2, state3),
            'final_kappa': kappa,
            'attention_history': attention_history,
            'last_attention': attention_history[-1] if attention_history else None
        }
        
        return outputs, extras
    
    def parse_mdn_params(self, mdn_output: torch.Tensor) -> MDNParams:
        """
        Parse MDN output into mixture parameters
        
        Args:
            mdn_output: Raw MDN output [batch, seq_len, output_size]
            
        Returns:
            MDNParams with all mixture components
        """
        batch_size, seq_len, _ = mdn_output.size()
        
        # Split parameters
        eos_hat = mdn_output[:, :, 0:1]
        mixture_params = mdn_output[:, :, 1:]
        
        n_mix = self.num_mixtures
        pi_hat, mu_x_hat, mu_y_hat, sigma_x_hat, sigma_y_hat, rho_hat = torch.split(
            mixture_params, [n_mix, n_mix, n_mix, n_mix, n_mix, n_mix], dim=2
        )
        
        # Apply activations (matching TF1 model exactly)
        eos = torch.sigmoid(-eos_hat)  # Note: negative sign as in TF1 model
        pi = F.softmax(pi_hat, dim=2)
        mu_x = mu_x_hat  # Leave as-is
        mu_y = mu_y_hat  # Leave as-is
        sigma_x = torch.exp(sigma_x_hat)
        sigma_y = torch.exp(sigma_y_hat)
        rho = torch.tanh(rho_hat)
        
        return MDNParams(
            eos=eos,
            pi=pi,
            mu_x=mu_x,
            mu_y=mu_y,
            sigma_x=sigma_x,
            sigma_y=sigma_y,
            rho=rho
        )
    
    def mdn_loss(self, 
                 mdn_params: MDNParams,
                 target_trajectory: torch.Tensor) -> torch.Tensor:
        """
        Compute MDN loss (negative log-likelihood)
        
        Args:
            mdn_params: Parsed MDN parameters
            target_trajectory: Target trajectory [batch, seq_len, 3]
            
        Returns:
            Scalar loss value
        """
        # Target values
        x_target = target_trajectory[:, :, 0:1]
        y_target = target_trajectory[:, :, 1:2]
        eos_target = target_trajectory[:, :, 2:3]
        
        # Compute 2D Gaussian likelihood for each mixture component
        dx = x_target - mdn_params.mu_x
        dy = y_target - mdn_params.mu_y
        
        # Bivariate Gaussian (Eq. 24-25 from Graves paper)
        z = (torch.square(dx / mdn_params.sigma_x) + 
             torch.square(dy / mdn_params.sigma_y) - 
             2 * mdn_params.rho * dx * dy / (mdn_params.sigma_x * mdn_params.sigma_y))
        
        rho_square_term = 1 - torch.square(mdn_params.rho)
        exp_term = torch.exp(-z / (2 * rho_square_term))
        normalization = (2 * math.pi * mdn_params.sigma_x * mdn_params.sigma_y * 
                        torch.sqrt(rho_square_term))
        
        gaussian = exp_term / normalization
        
        # Mixture likelihood
        weighted_gaussian = mdn_params.pi * gaussian
        mixture_likelihood = torch.sum(weighted_gaussian, dim=2, keepdim=True)
        
        # Clamp to avoid log(0)
        mixture_likelihood = torch.clamp(mixture_likelihood, min=1e-20)
        
        # Gaussian loss term
        gaussian_loss = -torch.log(mixture_likelihood)
        
        # End-of-stroke Bernoulli loss
        eos_loss = -(eos_target * torch.log(mdn_params.eos + 1e-20) + 
                     (1 - eos_target) * torch.log(1 - mdn_params.eos + 1e-20))
        
        # Combined loss
        total_loss = gaussian_loss + eos_loss
        
        return torch.mean(total_loss)
    
    def sample(self,
               char_sequence: torch.Tensor,
               max_length: int = 800,
               temperature: float = 1.0,
               bias: float = 0.0) -> torch.Tensor:
        """
        Sample trajectory from the model
        
        Args:
            char_sequence: Character sequence [1, char_len, alphabet_size]
            max_length: Maximum trajectory length
            temperature: Sampling temperature
            bias: Bias for mixture selection
            
        Returns:
            Sampled trajectory [1, generated_len, 3]
        """
        self.eval()
        with torch.no_grad():
            batch_size = 1
            device = char_sequence.device
            
            # Initialize states
            state1, state2, state3 = self.init_states(batch_size, device)
            kappa = self.init_attention(batch_size, device)
            
            # Start with pen down
            x_t = torch.zeros(batch_size, 3, device=device)
            x_t[:, 2] = 1.0  # Pen down initially
            
            trajectory = [x_t.clone()]
            
            for step in range(max_length):
                # Forward pass with current states
                mdn_output, extras = self.forward(
                    x_t.unsqueeze(1),  # Add seq_len dimension
                    char_sequence,
                    (state1, state2, state3),
                    kappa,
                    training=False
                )
                
                # Update states
                state1, state2, state3 = extras['final_states']
                kappa = extras['final_kappa']
                
                # Parse MDN parameters
                mdn_params = self.parse_mdn_params(mdn_output)
                
                # Apply temperature and bias
                pi_biased = mdn_params.pi.squeeze(1).clone()
                if bias != 0.0:
                    pi_biased = F.softmax(torch.log(pi_biased + 1e-8) + bias, dim=1)
                
                # Sample mixture component
                component_dist = torch.distributions.Categorical(pi_biased)
                component_idx = component_dist.sample()
                
                # Get parameters for selected component
                selected_mu_x = mdn_params.mu_x.squeeze(1).gather(1, component_idx.unsqueeze(1))
                selected_mu_y = mdn_params.mu_y.squeeze(1).gather(1, component_idx.unsqueeze(1))
                selected_sigma_x = mdn_params.sigma_x.squeeze(1).gather(1, component_idx.unsqueeze(1))
                selected_sigma_y = mdn_params.sigma_y.squeeze(1).gather(1, component_idx.unsqueeze(1))
                selected_rho = mdn_params.rho.squeeze(1).gather(1, component_idx.unsqueeze(1))
                
                # Sample from selected 2D Gaussian
                z1 = torch.randn(batch_size, 1, device=device) * temperature
                z2 = torch.randn(batch_size, 1, device=device) * temperature
                
                x = selected_mu_x + selected_sigma_x * z1
                y = selected_mu_y + selected_sigma_y * (
                    selected_rho * z1 + torch.sqrt(1 - selected_rho**2) * z2
                )
                
                # Sample end-of-stroke
                eos_prob = mdn_params.eos.squeeze()
                if temperature != 1.0:
                    eos_prob = torch.sigmoid(torch.log(eos_prob / (1 - eos_prob + 1e-8)) / temperature)
                
                eos = (torch.rand(batch_size, device=device) < eos_prob).float()
                
                # Create next point
                x_t = torch.cat([x, y, eos.unsqueeze(1)], dim=1)
                trajectory.append(x_t.clone())
                
                # Check for end of writing
                if eos.item() > 0.5:
                    break
            
            return torch.stack(trajectory, dim=1)

class TF1CompatTrainer:
    """Training pipeline for TF1 compatible model"""
    
    def __init__(self,
                 model: TF1CompatModel,
                 device: str = 'cpu',
                 learning_rate: float = 0.0008,
                 grad_clip: float = 10.0):
        
        self.model = model.to(device)
        self.device = device
        self.grad_clip = grad_clip
        
        # Use RMSprop optimizer (as in TF1 model)
        self.optimizer = optim.RMSprop(
            model.parameters(),
            lr=learning_rate,
            momentum=0.9,
            weight_decay=0.0
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ExponentialLR(
            self.optimizer, 
            gamma=0.97  # Decay factor
        )
    
    def train_step(self,
                   trajectory: torch.Tensor,
                   char_sequence: torch.Tensor,
                   target_trajectory: torch.Tensor) -> float:
        """Single training step"""
        
        self.model.train()
        self.optimizer.zero_grad()
        
        # Forward pass
        mdn_output, extras = self.model(
            trajectory, char_sequence, training=True
        )
        
        # Parse MDN parameters
        mdn_params = self.model.parse_mdn_params(mdn_output)
        
        # Compute loss
        loss = self.model.mdn_loss(mdn_params, target_trajectory)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
        
        # Update weights
        self.optimizer.step()
        
        return loss.item()
    
    def save_checkpoint(self, filepath: str, epoch: int, loss: float):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'loss': loss,
            'model_config': {
                'rnn_size': self.model.rnn_size,
                'num_mixtures': self.model.num_mixtures,
                'num_window_mixtures': self.model.num_window_mixtures,
                'alphabet_size': self.model.alphabet_size
            }
        }
        torch.save(checkpoint, filepath)
    
    def load_checkpoint(self, filepath: str) -> Tuple[int, float]:
        """Load model checkpoint"""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        return checkpoint['epoch'], checkpoint['loss']

def test_tf1_compat():
    """Test TF1 compatibility model"""
    print("Testing TF1 Compatibility Model...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Model parameters
    rnn_size = 256
    num_mixtures = 10
    num_window_mixtures = 5
    alphabet_size = 50
    
    # Create model
    model = TF1CompatModel(
        rnn_size=rnn_size,
        num_mixtures=num_mixtures,
        num_window_mixtures=num_window_mixtures,
        alphabet_size=alphabet_size
    ).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params:,}")
    
    # Test data
    batch_size, seq_len, char_len = 2, 30, 15
    trajectory = torch.randn(batch_size, seq_len, 3).to(device)
    char_sequence = torch.randn(batch_size, char_len, alphabet_size).to(device)
    
    # Test forward pass
    print("Testing forward pass...")
    start_time = time.time()
    mdn_output, extras = model(trajectory, char_sequence)
    forward_time = time.time() - start_time
    
    expected_output_size = 1 + num_mixtures * 6
    print(f"Forward pass time: {forward_time:.3f}s")
    print(f"Output shape: {mdn_output.shape}")
    print(f"Expected shape: ({batch_size}, {seq_len}, {expected_output_size})")
    
    # Test MDN parameter parsing
    print("Testing MDN parameter parsing...")
    mdn_params = model.parse_mdn_params(mdn_output)
    print(f"MDN params shapes: pi={mdn_params.pi.shape}, mu_x={mdn_params.mu_x.shape}")
    
    # Test loss computation
    print("Testing loss computation...")
    target = torch.randn(batch_size, seq_len, 3).to(device)
    loss = model.mdn_loss(mdn_params, target)
    print(f"Loss: {loss.item():.4f}")
    
    # Test sampling
    print("Testing sampling...")
    test_char_seq = torch.randn(1, char_len, alphabet_size).to(device)
    start_time = time.time()
    generated = model.sample(test_char_seq, max_length=50)
    sample_time = time.time() - start_time
    
    print(f"Sampling time: {sample_time:.3f}s")
    print(f"Generated trajectory shape: {generated.shape}")
    
    # Test trainer
    print("Testing trainer...")
    trainer = TF1CompatTrainer(model, str(device))
    
    loss = trainer.train_step(trajectory[:, :-1], char_sequence, trajectory[:, 1:])
    print(f"Training loss: {loss:.4f}")
    
    print("✅ TF1 compatibility test completed successfully!")
    
    return model, trainer

if __name__ == "__main__":
    model, trainer = test_tf1_compat()