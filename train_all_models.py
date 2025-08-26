#!/usr/bin/env python
"""
Unified training script for all trajectory generation models
Trains on combined real and synthetic dataset
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
import numpy as np
from pathlib import Path
import json
import time
from datetime import datetime
import pandas as pd
import sys

# Import all models
from pytorch_attention_rnn import AttentionRNN, AttentionTrainer
from pytorch_wgan_gp import Generator, Discriminator, WGAN_GP_Trainer, TrajectoryDataset
from pytorch_transformer import TransformerTrajectoryModel, TransformerTrainer
# Dataset will be unified

class UnifiedDataset(torch.utils.data.Dataset):
    """Unified dataset for all models"""
    
    def __init__(self, data_dir: str = "datasets/swipelogs", max_length: int = 150):
        self.max_length = max_length
        self.traces = []
        self.words = []
        
        # Load real traces
        real_count = self._load_real_traces(Path(data_dir))
        
        # Load synthetic traces
        synthetic_count = self._load_synthetic_traces(Path("datasets/synthetic_missing_words"))
        
        # Build vocabulary
        self._build_vocab()
        
        print(f"✅ Loaded {len(self.traces)} total traces")
        print(f"   - Real traces: {real_count}")
        print(f"   - Synthetic traces: {synthetic_count}")
        print(f"   - Vocabulary size: {self.vocab_size}")
    
    def _load_real_traces(self, data_dir: Path) -> int:
        """Load real swipelog traces"""
        count = 0
        log_files = list(data_dir.glob("*.log"))[:100]  # Limit to 100 files for testing
        
        for log_file in log_files:
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
                            count += 1
                            
                            # Stop if we have enough samples
                            if count >= 500:
                                return count
            except Exception as e:
                continue
        return count
    
    def _load_synthetic_traces(self, data_dir: Path) -> int:
        """Load synthetic traces"""
        count = 0
        if not data_dir.exists():
            return 0
            
        json_files = list(data_dir.glob("*.json"))[:50]  # Limit to 50 files
        for json_file in json_files:
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                
                # Parse synthetic session format
                if 'sessionMetadata' in data and 'swipes' in data:
                    for swipe in data['swipes']:
                        word = swipe['word'].lower()
                        trace = np.array(swipe['trace'])
                        
                        if len(trace) > 2:
                            # Normalize
                            trace = (trace - trace.mean(0)) / (trace.std(0) + 1e-6)
                            self.traces.append(trace)
                            self.words.append(word)
                            count += 1
                            
                            # Stop if we have enough samples
                            if count >= 500:
                                return count
            except Exception as e:
                continue
        return count
    
    def _build_vocab(self):
        """Build character vocabulary"""
        chars = set()
        for word in self.words:
            chars.update(word)
        self.char_to_idx = {char: idx + 1 for idx, char in enumerate(sorted(chars))}
        self.char_to_idx['<pad>'] = 0
        self.idx_to_char = {idx: char for char, idx in self.char_to_idx.items()}
        self.vocab_size = len(self.char_to_idx)
    
    def __len__(self):
        return len(self.traces)
    
    def __getitem__(self, idx):
        trace = self.traces[idx]
        word = self.words[idx]
        
        # Process trace
        if len(trace) > self.max_length:
            indices = np.linspace(0, len(trace)-1, self.max_length, dtype=int)
            trace = trace[indices]
        
        # Pad trace
        trace_tensor = torch.zeros(self.max_length, 3)
        trace_tensor[:len(trace), :2] = torch.FloatTensor(trace)
        trace_tensor[len(trace)-1, 2] = 1  # End marker
        
        # Encode word
        word_encoded = torch.zeros(10, dtype=torch.long)
        for i, char in enumerate(word[:10]):
            if char in self.char_to_idx:
                word_encoded[i] = self.char_to_idx[char]
        
        return trace_tensor, word_encoded, len(word)


def train_attention_rnn(dataset, device, epochs=20):
    """Train Attention RNN model"""
    print("\n" + "="*60)
    print("TRAINING ATTENTION RNN")
    print("="*60)
    
    # Create model
    model = AttentionRNN(
        vocab_size=dataset.vocab_size,
        embedding_size=128,
        hidden_size=256,
        num_attn_mixtures=10,
        num_output_mixtures=20
    )
    
    # Create data loader
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=0)
    
    # Create trainer
    trainer = AttentionTrainer(model, device, learning_rate=0.0001)
    
    # Training loop
    best_loss = float('inf')
    for epoch in range(epochs):
        epoch_losses = []
        
        for batch_idx, (traces, words, word_lengths) in enumerate(dataloader):
            traces = traces.to(device)
            words = words.to(device)
            word_lengths = word_lengths.to(device)
            
            loss = trainer.train_step(words, traces, word_lengths)
            epoch_losses.append(loss)
            
            if batch_idx % 50 == 0:
                print(f"  Epoch {epoch+1}/{epochs} [{batch_idx}/{len(dataloader)}] "
                      f"Loss: {loss:.4f}")
        
        avg_loss = np.mean(epoch_losses)
        print(f"Epoch {epoch+1} - Average Loss: {avg_loss:.4f}")
        
        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'model_state': model.state_dict(),
                'optimizer_state': trainer.optimizer.state_dict(),
                'epoch': epoch,
                'loss': best_loss
            }, 'checkpoints/attention_rnn_best.pt')
            print(f"  ✅ Saved best model (loss: {best_loss:.4f})")
    
    return model, best_loss


def train_wgan_gp(dataset, device, epochs=20):
    """Train WGAN-GP model"""
    print("\n" + "="*60)
    print("TRAINING WGAN-GP")
    print("="*60)
    
    # Create models
    generator = Generator(dataset.vocab_size, latent_dim=128)
    discriminator = Discriminator(dataset.vocab_size)
    
    # Create data loader
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=0)
    
    # Create trainer
    trainer = WGAN_GP_Trainer(generator, discriminator, device)
    
    # Training loop
    for epoch in range(epochs):
        epoch_d_loss = []
        epoch_g_loss = []
        
        for batch_idx, (real_traj, word, _) in enumerate(dataloader):
            real_traj = real_traj.to(device)
            word = word.to(device)
            
            losses = trainer.train_step(real_traj, word)
            
            epoch_d_loss.append(losses['d_loss'])
            epoch_g_loss.append(losses['g_loss'])
            
            if batch_idx % 50 == 0:
                print(f"  Epoch {epoch+1}/{epochs} [{batch_idx}/{len(dataloader)}] "
                      f"D: {losses['d_loss']:.4f} G: {losses['g_loss']:.4f}")
        
        avg_d_loss = np.mean(epoch_d_loss)
        avg_g_loss = np.mean(epoch_g_loss)
        print(f"Epoch {epoch+1} - D Loss: {avg_d_loss:.4f}, G Loss: {avg_g_loss:.4f}")
        
        # Save checkpoint
        if (epoch + 1) % 5 == 0:
            trainer.save_checkpoint(epoch + 1)
    
    return generator, discriminator, avg_g_loss


def train_transformer(dataset, device, epochs=20):
    """Train Transformer model"""
    print("\n" + "="*60)
    print("TRAINING TRANSFORMER")
    print("="*60)
    
    # Create model
    model = TransformerTrajectoryModel(
        vocab_size=dataset.vocab_size,
        d_model=256,
        nhead=8,
        num_encoder_layers=3,
        num_decoder_layers=3
    )
    
    # Create data loader
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=0)
    
    # Create trainer
    trainer = TransformerTrainer(model, device, learning_rate=0.0001)
    
    # Training loop
    best_loss = float('inf')
    for epoch in range(epochs):
        epoch_losses = []
        
        for batch_idx, (traces, words, _) in enumerate(dataloader):
            traces = traces.to(device)
            words = words.to(device)
            
            loss = trainer.train_step(words, traces)
            epoch_losses.append(loss)
            
            if batch_idx % 50 == 0:
                print(f"  Epoch {epoch+1}/{epochs} [{batch_idx}/{len(dataloader)}] "
                      f"Loss: {loss:.4f}")
        
        avg_loss = np.mean(epoch_losses)
        print(f"Epoch {epoch+1} - Average Loss: {avg_loss:.4f}")
        
        # Update learning rate
        trainer.scheduler.step(avg_loss)
        
        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'model_state': model.state_dict(),
                'optimizer_state': trainer.optimizer.state_dict(),
                'epoch': epoch,
                'loss': best_loss
            }, 'checkpoints/transformer_best.pt')
            print(f"  ✅ Saved best model (loss: {best_loss:.4f})")
    
    return model, best_loss


def evaluate_models(dataset, device):
    """Evaluate all trained models"""
    print("\n" + "="*60)
    print("MODEL EVALUATION")
    print("="*60)
    
    results = {}
    
    # Load and evaluate each model if checkpoint exists
    checkpoint_dir = Path("checkpoints")
    
    # Attention RNN
    if (checkpoint_dir / "attention_rnn_best.pt").exists():
        checkpoint = torch.load("checkpoints/attention_rnn_best.pt", map_location=device)
        print(f"Attention RNN - Best Loss: {checkpoint['loss']:.4f} (Epoch {checkpoint['epoch']+1})")
        results['attention_rnn'] = checkpoint['loss']
    
    # WGAN-GP
    wgan_files = list(checkpoint_dir.glob("wgan_gp_epoch_*.pt"))
    if wgan_files:
        latest = max(wgan_files, key=lambda x: int(x.stem.split('_')[-1]))
        checkpoint = torch.load(latest, map_location=device)
        print(f"WGAN-GP - Trained for {checkpoint['epoch']} epochs")
        if 'history' in checkpoint:
            print(f"  Final D Loss: {checkpoint['history']['d_loss'][-1]:.4f}")
            print(f"  Final G Loss: {checkpoint['history']['g_loss'][-1]:.4f}")
            results['wgan_gp'] = checkpoint['history']['g_loss'][-1]
    
    # Transformer
    if (checkpoint_dir / "transformer_best.pt").exists():
        checkpoint = torch.load("checkpoints/transformer_best.pt", map_location=device)
        print(f"Transformer - Best Loss: {checkpoint['loss']:.4f} (Epoch {checkpoint['epoch']+1})")
        results['transformer'] = checkpoint['loss']
    
    return results


def main():
    """Main training pipeline"""
    
    print("="*60)
    print("UNIFIED MODEL TRAINING PIPELINE")
    print("="*60)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create checkpoint directory
    Path("checkpoints").mkdir(exist_ok=True)
    
    # Load unified dataset
    print("\nLoading dataset...")
    dataset = UnifiedDataset()
    
    if len(dataset) == 0:
        print("❌ No data loaded!")
        return
    
    # Training parameters
    epochs_per_model = 2  # Quick training for testing
    
    # Train each model
    start_time = time.time()
    
    # 1. Attention RNN
    try:
        rnn_model, rnn_loss = train_attention_rnn(dataset, device, epochs_per_model)
    except Exception as e:
        print(f"⚠️ Attention RNN training failed: {e}")
        rnn_loss = None
    
    # 2. WGAN-GP
    try:
        gen_model, disc_model, gan_loss = train_wgan_gp(dataset, device, epochs_per_model)
    except Exception as e:
        print(f"⚠️ WGAN-GP training failed: {e}")
        gan_loss = None
    
    # 3. Transformer
    try:
        transformer_model, transformer_loss = train_transformer(dataset, device, epochs_per_model)
    except Exception as e:
        print(f"⚠️ Transformer training failed: {e}")
        transformer_loss = None
    
    # Evaluate all models
    results = evaluate_models(dataset, device)
    
    # Training summary
    training_time = time.time() - start_time
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print(f"Total training time: {training_time/60:.1f} minutes")
    print(f"Dataset size: {len(dataset)} traces")
    print(f"Models trained: {len([x for x in [rnn_loss, gan_loss, transformer_loss] if x is not None])}/3")
    
    if results:
        print("\nBest performing model:")
        best_model = min(results.items(), key=lambda x: x[1] if x[1] is not None else float('inf'))
        print(f"  🏆 {best_model[0]}: {best_model[1]:.4f}")
    
    # Save training summary
    summary = {
        'timestamp': datetime.now().isoformat(),
        'dataset_size': len(dataset),
        'vocab_size': dataset.vocab_size,
        'device': str(device),
        'training_time_minutes': training_time/60,
        'epochs_per_model': epochs_per_model,
        'results': {k: float(v) if v is not None else None for k, v in results.items()}
    }
    
    with open('training_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("\n✅ Training summary saved to training_summary.json")
    print("✅ Model checkpoints saved to checkpoints/")
    
    return results


if __name__ == "__main__":
    results = main()