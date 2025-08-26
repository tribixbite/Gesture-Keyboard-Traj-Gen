#!/usr/bin/env python
"""
PyTorch-based RNN trainer for gesture keyboard trajectory generation
"""

import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import time
from pytorch_attention_rnn import AttentionRNN, AttentionTrainer
from model_checkpoint_manager import CheckpointManager

class SwipelogDataset(Dataset):
    """PyTorch Dataset for swipelog data"""
    
    def __init__(self, traces, labels, char_to_idx, max_seq_len=150):
        self.traces = traces
        self.labels = labels
        self.char_to_idx = char_to_idx
        self.max_seq_len = max_seq_len
        
    def __len__(self):
        return len(self.traces)
    
    def __getitem__(self, idx):
        # Get trace and normalize
        trace = self.traces[idx]
        if len(trace) > self.max_seq_len:
            indices = np.linspace(0, len(trace)-1, self.max_seq_len, dtype=int)
            trace = trace[indices]
        
        # Normalize
        trace = np.array(trace, dtype=np.float32)
        if len(trace) > 0:
            trace[:, 0] = (trace[:, 0] - trace[:, 0].min()) / (trace[:, 0].max() - trace[:, 0].min() + 1e-6)
            trace[:, 1] = (trace[:, 1] - trace[:, 1].min()) / (trace[:, 1].max() - trace[:, 1].min() + 1e-6)
        
        # Add end-of-stroke marker
        eos = np.zeros((len(trace), 1))
        if len(trace) > 0:
            eos[-1] = 1
        trace = np.concatenate([trace, eos], axis=1)
        
        # Pad if necessary
        if len(trace) < self.max_seq_len:
            padding = np.zeros((self.max_seq_len - len(trace), 3))
            trace = np.concatenate([trace, padding], axis=0)
        
        # Encode label
        label = self.labels[idx]
        encoded = [self.char_to_idx.get('<START>', 0)]
        for char in label:
            if char in self.char_to_idx:
                encoded.append(self.char_to_idx[char])
        encoded.append(self.char_to_idx.get('<END>', 0))
        
        return torch.FloatTensor(trace), torch.LongTensor(encoded)

class TrajectoryRNN(nn.Module):
    """RNN model for trajectory generation"""
    
    def __init__(self, vocab_size, hidden_size=256, num_layers=2, 
                 mixture_components=20, dropout=0.2):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.mixture_components = mixture_components
        
        # Trajectory encoder
        self.traj_encoder = nn.LSTM(
            input_size=3,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        # Word decoder
        self.word_embedding = nn.Embedding(vocab_size, 128)
        self.word_decoder = nn.LSTM(
            input_size=128,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        self.word_output = nn.Linear(hidden_size, vocab_size)
        
        # Trajectory generator
        self.traj_decoder = nn.LSTM(
            input_size=3,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        # Mixture density network output
        self.mixture_output = nn.Linear(
            hidden_size,
            mixture_components * 6 + 1  # pi, mu_x, mu_y, sigma_x, sigma_y, rho, eos
        )
        
    def forward(self, trajectories, words=None):
        batch_size = trajectories.size(0)
        
        # Encode trajectory
        traj_out, (h_n, c_n) = self.traj_encoder(trajectories)
        
        # If words provided, decode to words
        word_logits = None
        if words is not None:
            word_embeds = self.word_embedding(words)
            word_out, _ = self.word_decoder(word_embeds, (h_n, c_n))
            word_logits = self.word_output(word_out)
        
        # Generate trajectory parameters
        traj_gen_out, _ = self.traj_decoder(trajectories, (h_n, c_n))
        mixture_params = self.mixture_output(traj_gen_out)
        
        return word_logits, mixture_params
    
    def generate(self, word_encoded, max_length=150):
        """Generate trajectory from word encoding"""
        self.eval()
        with torch.no_grad():
            # Start with zeros
            trajectory = torch.zeros(1, max_length, 3)
            
            # Encode word
            word_tensor = torch.LongTensor([word_encoded]).unsqueeze(0)
            word_embeds = self.word_embedding(word_tensor)
            
            # Initialize hidden state from word
            h_0 = torch.zeros(self.num_layers, 1, self.hidden_size)
            c_0 = torch.zeros(self.num_layers, 1, self.hidden_size)
            _, (h_n, c_n) = self.word_decoder(word_embeds, (h_0, c_0))
            
            # Generate trajectory
            points = []
            hidden = (h_n, c_n)
            
            for t in range(max_length):
                # Get next point
                if t == 0:
                    x_t = torch.zeros(1, 1, 3)
                else:
                    x_t = torch.FloatTensor(points[-1]).unsqueeze(0).unsqueeze(0)
                
                output, hidden = self.traj_decoder(x_t, hidden)
                mixture_params = self.mixture_output(output)
                
                # Sample from mixture
                point = self.sample_from_mixture(mixture_params[0, 0])
                points.append(point)
                
                # Check for end of stroke
                if point[2] > 0.5:
                    break
            
            return np.array(points)
    
    def sample_from_mixture(self, params):
        """Sample point from mixture parameters"""
        m = self.mixture_components
        
        # Parse parameters
        pi = torch.softmax(params[:m], dim=0)
        mu_x = params[m:2*m]
        mu_y = params[2*m:3*m]
        sigma_x = torch.exp(params[3*m:4*m])
        sigma_y = torch.exp(params[4*m:5*m])
        rho = torch.tanh(params[5*m:6*m])
        eos = torch.sigmoid(params[-1])
        
        # Sample component
        component = torch.multinomial(pi, 1).item()
        
        # Sample from component
        x = torch.randn(1) * sigma_x[component] + mu_x[component]
        y = torch.randn(1) * sigma_y[component] + mu_y[component]
        e = 1.0 if torch.rand(1) < eos else 0.0
        
        return [x.item(), y.item(), e]

def train_model(model, train_loader, val_loader, epochs=30, lr=0.001, device='cpu'):
    """Train the RNN model"""
    
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    word_criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding
    
    history = {'train_loss': [], 'val_loss': []}
    
    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        
        # Training
        model.train()
        train_losses = []
        
        for batch_idx, (trajectories, words) in enumerate(train_loader):
            trajectories = trajectories.to(device)
            words = words.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            word_logits, mixture_params = model(trajectories, words)
            
            # Calculate losses
            word_loss = 0
            if word_logits is not None:
                # Reshape for loss calculation
                word_logits_flat = word_logits[:, :-1].contiguous().view(-1, word_logits.size(-1))
                words_flat = words[:, 1:].contiguous().view(-1)
                word_loss = word_criterion(word_logits_flat, words_flat)
            
            # Reconstruction loss using mixture mean positions
            # Extract mu_x and mu_y from first component (indices 20-40)
            m = model.mixture_components
            mu_x = mixture_params[:, :, m:2*m]  # mu_x components
            mu_y = mixture_params[:, :, 2*m:3*m]  # mu_y components
            
            # Use first component as prediction for simplicity
            traj_pred_x = mu_x[:, :, 0]  # First component mu_x
            traj_pred_y = mu_y[:, :, 0]  # First component mu_y
            
            # Reconstruction loss for x and y coordinates
            traj_loss = torch.mean((traj_pred_x - trajectories[:, :, 0]) ** 2) + \
                       torch.mean((traj_pred_y - trajectories[:, :, 1]) ** 2)
            
            # Combined loss
            total_loss = word_loss + 0.5 * traj_loss
            
            # Backward pass
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            
            train_losses.append(total_loss.item())
            
            if batch_idx % 10 == 0:
                print(f"  Batch {batch_idx}: loss={total_loss.item():.4f}")
        
        # Validation
        model.eval()
        val_losses = []
        with torch.no_grad():
            for trajectories, words in val_loader:
                trajectories = trajectories.to(device)
                words = words.to(device)
                
                word_logits, mixture_params = model(trajectories, words)
                
                word_loss = 0
                if word_logits is not None:
                    word_logits_flat = word_logits[:, :-1].contiguous().view(-1, word_logits.size(-1))
                    words_flat = words[:, 1:].contiguous().view(-1)
                    word_loss = word_criterion(word_logits_flat, words_flat)
                
                # Reconstruction loss using mixture mean positions (same as training)
                m = model.mixture_components
                mu_x = mixture_params[:, :, m:2*m]  # mu_x components
                mu_y = mixture_params[:, :, 2*m:3*m]  # mu_y components
                
                # Use first component as prediction
                traj_pred_x = mu_x[:, :, 0]
                traj_pred_y = mu_y[:, :, 0]
                
                traj_loss = torch.mean((traj_pred_x - trajectories[:, :, 0]) ** 2) + \
                           torch.mean((traj_pred_y - trajectories[:, :, 1]) ** 2)
                total_loss = word_loss + 0.5 * traj_loss
                
                val_losses.append(total_loss.item())
        
        avg_train_loss = np.mean(train_losses)
        avg_val_loss = np.mean(val_losses)
        
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Val Loss: {avg_val_loss:.4f}")
        
        # Early stopping
        if epoch > 5 and history['val_loss'][-1] > history['val_loss'][-3]:
            print("Early stopping triggered")
            break
    
    return history

def train_attention_model(trainer, train_loader, val_loader, epochs=10):
    """Training function for attention-based model"""
    
    history = {'train_loss': [], 'val_loss': []}
    
    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        
        # Training
        train_losses = []
        trainer.model.train()
        
        for batch_idx, (trajectories, words) in enumerate(train_loader):
            trajectories = trajectories.to(trainer.device)
            words = words.to(trainer.device)
            
            # Convert to appropriate format for attention model
            # Words should be padded to same length for batch processing
            max_word_len = words.size(1)
            word_lengths = torch.sum(words != 0, dim=1)
            
            try:
                loss = trainer.train_step(words, trajectories, word_lengths)
                train_losses.append(loss)
                
                if batch_idx % 10 == 0:
                    print(f"  Batch {batch_idx}: loss={loss:.4f}")
                    
            except Exception as e:
                print(f"  Batch {batch_idx} failed: {e}")
                continue
        
        if not train_losses:
            print("  No successful training batches")
            continue
            
        # Validation
        val_losses = []
        trainer.model.eval()
        with torch.no_grad():
            for trajectories, words in val_loader:
                trajectories = trajectories.to(trainer.device)
                words = words.to(trainer.device)
                
                word_lengths = torch.sum(words != 0, dim=1)
                
                try:
                    # Simple validation loss computation
                    traj_input = trajectories[:, :-1]
                    traj_target = trajectories[:, 1:]
                    
                    mixture_params = trainer.model(traj_input, words, word_lengths)
                    loss = trainer.mixture_loss(mixture_params[:, :traj_target.size(1)], traj_target)
                    val_losses.append(loss.item())
                except Exception as e:
                    continue
        
        avg_train_loss = np.mean(train_losses) if train_losses else float('inf')
        avg_val_loss = np.mean(val_losses) if val_losses else float('inf')
        
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Val Loss: {avg_val_loss:.4f}")
        
        # Update learning rate
        trainer.scheduler.step(avg_val_loss)
        
        # Early stopping
        if epoch > 5 and len(history['val_loss']) > 3:
            if history['val_loss'][-1] > history['val_loss'][-3]:
                print("Early stopping triggered")
                break
    
    return history

def load_data():
    """Load and process swipelog data"""
    print("Loading swipelog data...")
    
    traces = []
    labels = []
    
    # Load from both real and synthetic
    for data_dir in [Path("datasets/swipelogs"), Path("datasets/synthetic_swipelogs")]:
        if not data_dir.exists():
            continue
        
        count = 0
        for log_file in list(data_dir.glob("*.log"))[:500]:  # Limit for testing
            try:
                df = pd.read_csv(log_file, sep=r'\s+', engine='python', 
                                on_bad_lines='skip', header=0)
                
                if 'word' in df.columns and 'x_pos' in df.columns:
                    for word in df['word'].unique():
                        if pd.isna(word) or word == 'word':
                            continue
                        
                        word_df = df[df['word'] == word]
                        traj = word_df[['x_pos', 'y_pos']].values
                        
                        if len(traj) > 2 and len(traj) < 150:
                            # Ensure we always add both trace and label together
                            traces.append(traj)
                            labels.append(word.lower())
                            count += 1
                            
                            if count >= 500:  # Limit per directory
                                break
            except Exception as e:
                continue
    
    # Ensure traces and labels have same length
    min_len = min(len(traces), len(labels))
    traces = traces[:min_len]
    labels = labels[:min_len]
    
    print(f"Loaded {len(traces)} traces")
    
    # Build vocabulary
    chars = set('<PAD><START><END>')
    for word in labels:
        chars.update(word)
    
    char_to_idx = {char: idx for idx, char in enumerate(sorted(chars))}
    idx_to_char = {idx: char for char, idx in char_to_idx.items()}
    
    return traces, labels, char_to_idx, idx_to_char

def create_attention_model(vocab_size, use_attention=True):
    """Create either attention-based or simple RNN model"""
    if use_attention:
        print("Using Attention-based RNN Model")
        model = AttentionRNN(
            vocab_size=vocab_size,
            embedding_size=128,
            hidden_size=256,
            num_attn_mixtures=10,
            num_output_mixtures=20,
            dropout=0.2
        )
    else:
        print("Using Simple RNN Model")
        model = TrajectoryRNN(
            vocab_size=vocab_size,
            hidden_size=256,
            num_layers=2,
            mixture_components=20,
            dropout=0.2
        )
    return model

def main():
    """Main training function"""
    print("="*60)
    print("PYTORCH RNN TRAINING WITH ATTENTION")
    print("="*60)
    
    # Load data
    traces, labels, char_to_idx, idx_to_char = load_data()
    
    if len(traces) == 0:
        print("❌ No data loaded")
        return
    
    print(f"Vocabulary size: {len(char_to_idx)}")
    
    # Split data (simple split without sklearn)
    split_idx = int(len(traces) * 0.85)
    train_traces = traces[:split_idx]
    val_traces = traces[split_idx:]
    train_labels = labels[:split_idx]
    val_labels = labels[split_idx:]
    
    print(f"Training set: {len(train_traces)} samples")
    print(f"Validation set: {len(val_traces)} samples")
    
    # Create datasets and loaders
    train_dataset = SwipelogDataset(train_traces, train_labels, char_to_idx)
    val_dataset = SwipelogDataset(val_traces, val_labels, char_to_idx)
    
    # Custom collate function to handle variable length sequences
    def collate_fn(batch):
        if not batch:
            return torch.zeros(0, 150, 3), torch.zeros(0, 1, dtype=torch.long)
            
        trajectories, labels = zip(*batch)
        
        # Pad labels to same length
        max_label_len = max(len(l) for l in labels) if labels else 1
        padded_labels = []
        for label in labels:
            padded = torch.zeros(max_label_len, dtype=torch.long)
            if len(label) > 0:
                padded[:len(label)] = label
            padded_labels.append(padded)
        
        # Stack trajectories (already padded to same length)
        traj_tensor = torch.stack(list(trajectories))
        label_tensor = torch.stack(padded_labels) if padded_labels else torch.zeros(0, max_label_len, dtype=torch.long)
        
        return traj_tensor, label_tensor
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)
    
    # Create model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Try attention model first, fallback to simple RNN
    try:
        model = create_attention_model(len(char_to_idx), use_attention=True)
        use_attention_training = True
        print("✅ Attention model created successfully")
    except Exception as e:
        print(f"⚠️ Attention model failed: {e}")
        print("Falling back to simple RNN model")
        model = create_attention_model(len(char_to_idx), use_attention=False)
        use_attention_training = False
    
    # Train with appropriate trainer
    if use_attention_training:
        # Use attention-specific training
        trainer = AttentionTrainer(model, device=str(device), learning_rate=0.001)
        history = train_attention_model(trainer, train_loader, val_loader, epochs=10)
    else:
        # Use original training
        history = train_model(model, train_loader, val_loader, 
                             epochs=10, lr=0.001, device=device)
    
    # Save model with checkpoint manager
    checkpoint_manager = CheckpointManager()
    
    # Determine final epoch and model type
    final_epoch = len(history['train_loss'])
    model_type = "rnn_attention" if use_attention_training else "rnn_simple"
    
    # Save final checkpoint
    checkpoint = checkpoint_manager.save_checkpoint(
        model=model,
        model_type=model_type,
        model_name="pytorch_rnn",
        epoch=final_epoch,
        optimizer=trainer.optimizer if use_attention_training else None,
        metrics={
            'train_loss': history['train_loss'][-1] if history['train_loss'] else 0,
            'val_loss': history['val_loss'][-1] if history['val_loss'] else 0
        },
        config={
            'vocab_size': len(char_to_idx),
            'use_attention': use_attention_training,
            'data_size': len(traces)
        },
        extra_data={
            'char_to_idx': char_to_idx,
            'idx_to_char': idx_to_char,
            'history': history
        }
    )
    
    print(f"✅ Saved checkpoint: {checkpoint.checkpoint_path}")
    
    # Also save to legacy location for compatibility
    os.makedirs('models', exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'char_to_idx': char_to_idx,
        'idx_to_char': idx_to_char,
        'history': history
    }, 'models/pytorch_rnn_model.pt')
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print(f"Final train loss: {history['train_loss'][-1]:.4f}")
    print(f"Final val loss: {history['val_loss'][-1]:.4f}")
    print("Model saved to models/pytorch_rnn_model.pt")
    
    # Test generation
    print("\nTesting generation...")
    model.eval()
    test_words = ['hello', 'world', 'the']
    for word in test_words:
        try:
            encoded = [char_to_idx.get(c, 0) for c in word]
            trajectory = model.generate(encoded)
            print(f"  '{word}': Generated {len(trajectory)} points")
        except Exception as e:
            print(f"  '{word}': Generation failed - {e}")
    
    return history

if __name__ == "__main__":
    history = main()