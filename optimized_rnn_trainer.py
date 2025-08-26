#!/usr/bin/env python
"""
Optimized RNN trainer for gesture keyboard trajectory generation
Improved to work with real and synthetic swipelog datasets
"""

import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import time
from sklearn.model_selection import train_test_split

class SwipelogDataProcessor:
    """Process swipelog format data for RNN training"""
    
    def __init__(self, real_dir: str = "datasets/swipelogs", 
                 synthetic_dir: str = "datasets/synthetic_swipelogs"):
        self.real_dir = Path(real_dir)
        self.synthetic_dir = Path(synthetic_dir)
        self.max_seq_length = 150  # Maximum sequence length
        self.vocab = {}
        self.char_to_idx = {}
        self.idx_to_char = {}
        
    def load_traces(self, limit: int = None) -> Dict:
        """Load traces from both real and synthetic datasets"""
        print("Loading trace data...")
        
        traces = []
        labels = []
        
        # Load real traces
        real_count = 0
        for log_file in list(self.real_dir.glob("*.log"))[:limit]:
            try:
                df = pd.read_csv(log_file, sep='\s+', engine='python', 
                                on_bad_lines='skip', header=0)
                if 'word' in df.columns and 'x_pos' in df.columns:
                    # Group by word
                    for word in df['word'].unique():
                        if pd.isna(word) or word == 'word':
                            continue
                        word_df = df[df['word'] == word]
                        
                        # Extract trajectory
                        traj = word_df[['x_pos', 'y_pos']].values
                        if len(traj) > 2:
                            traces.append(traj)
                            labels.append(word.lower())
                            real_count += 1
            except Exception as e:
                continue
        
        print(f"Loaded {real_count} real traces")
        
        # Load synthetic traces
        synthetic_count = 0
        for log_file in list(self.synthetic_dir.glob("*.log"))[:limit]:
            try:
                df = pd.read_csv(log_file, sep='\s+', engine='python', 
                                on_bad_lines='skip', header=0)
                if 'word' in df.columns and 'x_pos' in df.columns:
                    # Group by word
                    for word in df['word'].unique():
                        if pd.isna(word) or word == 'word':
                            continue
                        word_df = df[df['word'] == word]
                        
                        # Extract trajectory
                        traj = word_df[['x_pos', 'y_pos']].values
                        if len(traj) > 2:
                            traces.append(traj)
                            labels.append(word.lower())
                            synthetic_count += 1
            except Exception as e:
                continue
        
        print(f"Loaded {synthetic_count} synthetic traces")
        print(f"Total traces: {len(traces)}")
        
        # Build vocabulary
        self.build_vocabulary(labels)
        
        return {'traces': traces, 'labels': labels}
    
    def build_vocabulary(self, words: List[str]):
        """Build character vocabulary from words"""
        chars = set()
        for word in words:
            chars.update(word)
        
        # Add special tokens
        chars.add('<PAD>')
        chars.add('<START>')
        chars.add('<END>')
        
        # Create mappings
        self.char_to_idx = {char: idx for idx, char in enumerate(sorted(chars))}
        self.idx_to_char = {idx: char for char, idx in self.char_to_idx.items()}
        self.vocab_size = len(self.char_to_idx)
        
        print(f"Vocabulary size: {self.vocab_size}")
    
    def preprocess_data(self, traces: List, labels: List) -> Tuple:
        """Preprocess traces and labels for training"""
        
        # Normalize traces
        X = []
        for trace in traces:
            # Normalize to [0, 1]
            trace = np.array(trace, dtype=np.float32)
            if len(trace) > self.max_seq_length:
                # Downsample
                indices = np.linspace(0, len(trace)-1, self.max_seq_length, dtype=int)
                trace = trace[indices]
            
            # Normalize coordinates
            trace[:, 0] = (trace[:, 0] - trace[:, 0].min()) / (trace[:, 0].max() - trace[:, 0].min() + 1e-6)
            trace[:, 1] = (trace[:, 1] - trace[:, 1].min()) / (trace[:, 1].max() - trace[:, 1].min() + 1e-6)
            
            # Add end-of-stroke marker (third dimension)
            eos = np.zeros((len(trace), 1))
            eos[-1] = 1
            trace = np.concatenate([trace, eos], axis=1)
            
            X.append(trace)
        
        # Pad sequences
        X_padded = tf.keras.preprocessing.sequence.pad_sequences(
            X, maxlen=self.max_seq_length, dtype='float32',
            padding='post', value=0.0
        )
        
        # Encode labels
        y = []
        for label in labels:
            encoded = [self.char_to_idx['<START>']]
            for char in label:
                if char in self.char_to_idx:
                    encoded.append(self.char_to_idx[char])
            encoded.append(self.char_to_idx['<END>'])
            y.append(encoded)
        
        # Pad labels
        y_padded = tf.keras.preprocessing.sequence.pad_sequences(
            y, padding='post', value=self.char_to_idx['<PAD>']
        )
        
        return X_padded, y_padded


class OptimizedRNN(keras.Model):
    """Optimized RNN model for trajectory generation"""
    
    def __init__(self, vocab_size: int, lstm_units: int = 256, 
                 mixture_components: int = 20, dropout_rate: float = 0.2):
        super().__init__()
        
        self.lstm_units = lstm_units
        self.mixture_components = mixture_components
        self.output_size = mixture_components * 6 + 1  # GMM parameters + end-of-stroke
        
        # Encoder for trajectories
        self.trajectory_encoder = keras.Sequential([
            layers.LSTM(lstm_units, return_sequences=True, dropout=dropout_rate),
            layers.LSTM(lstm_units // 2, dropout=dropout_rate)
        ])
        
        # Decoder for word prediction
        self.word_embedding = layers.Embedding(vocab_size, 128)
        self.word_decoder = layers.LSTM(lstm_units, return_sequences=True, dropout=dropout_rate)
        self.word_output = layers.Dense(vocab_size)
        
        # Trajectory generator (reverse direction)
        self.trajectory_decoder = layers.LSTM(lstm_units, return_sequences=True, dropout=dropout_rate)
        self.mixture_output = layers.Dense(self.output_size)
        
    def call(self, inputs, training=False):
        trajectories, words = inputs
        
        # Encode trajectory
        traj_features = self.trajectory_encoder(trajectories, training=training)
        
        # Decode to words (for supervision)
        word_embeddings = self.word_embedding(words)
        word_hidden = self.word_decoder(word_embeddings, 
                                        initial_state=[traj_features, traj_features],
                                        training=training)
        word_logits = self.word_output(word_hidden)
        
        # Generate trajectory parameters (for reconstruction)
        traj_hidden = self.trajectory_decoder(trajectories, 
                                             initial_state=[traj_features, traj_features],
                                             training=training)
        mixture_params = self.mixture_output(traj_hidden)
        
        return word_logits, mixture_params
    
    def generate_trajectory(self, word: str, char_to_idx: Dict, max_length: int = 150):
        """Generate trajectory for a given word"""
        
        # Encode word
        word_encoded = [char_to_idx.get(c, 0) for c in word.lower()]
        word_tensor = tf.constant([word_encoded])
        
        # Initialize with zeros for trajectory input
        traj_input = tf.zeros((1, max_length, 3))
        
        # Forward pass
        _, mixture_params = self.call([traj_input, word_tensor], training=False)
        
        # Sample from mixture parameters
        trajectory = self.sample_from_mixture(mixture_params)
        
        return trajectory
    
    def sample_from_mixture(self, params):
        """Sample trajectory from GMM parameters"""
        # Parse mixture parameters
        num_components = self.mixture_components
        
        pis = params[..., :num_components]
        mus = params[..., num_components:3*num_components]
        sigmas = params[..., 3*num_components:5*num_components]
        rhos = params[..., 5*num_components:6*num_components]
        eos = params[..., -1:]
        
        # Apply activations
        pis = tf.nn.softmax(pis, axis=-1)
        sigmas = tf.nn.softplus(sigmas) + 1e-6
        rhos = tf.nn.tanh(rhos)
        eos = tf.nn.sigmoid(eos)
        
        # Sample component
        component = tf.random.categorical(tf.math.log(pis[0]), 1)[0, 0]
        
        # Sample from chosen component
        mu_x = mus[0, :, component]
        mu_y = mus[0, :, component + num_components]
        sigma_x = sigmas[0, :, component]
        sigma_y = sigmas[0, :, component + num_components]
        rho = rhos[0, :, component]
        
        # Generate correlated samples
        trajectory = []
        for t in range(len(mu_x)):
            x = tf.random.normal([1]) * sigma_x[t] + mu_x[t]
            y = tf.random.normal([1]) * sigma_y[t] + mu_y[t]
            e = 1.0 if tf.random.uniform([1]) < eos[0, t] else 0.0
            
            trajectory.append([x.numpy()[0], y.numpy()[0], e])
            
            if e > 0.5:  # End of stroke
                break
        
        return np.array(trajectory)


class RNNTrainer:
    """Trainer for optimized RNN model"""
    
    def __init__(self, model: OptimizedRNN, learning_rate: float = 0.001):
        self.model = model
        self.optimizer = keras.optimizers.Adam(learning_rate)
        self.word_loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        
    def mixture_loss(self, y_true, mixture_params):
        """Negative log-likelihood loss for GMM"""
        # Simplified version - full implementation would parse GMM params
        # and compute proper NLL
        return tf.reduce_mean(tf.square(y_true - mixture_params[..., :3]))
    
    @tf.function
    def train_step(self, trajectories, words):
        """Single training step"""
        with tf.GradientTape() as tape:
            word_logits, mixture_params = self.model([trajectories, words], training=True)
            
            # Word prediction loss
            word_loss = self.word_loss_fn(words[:, 1:], word_logits[:, :-1])
            
            # Trajectory reconstruction loss
            traj_loss = self.mixture_loss(trajectories, mixture_params)
            
            # Combined loss
            total_loss = word_loss + 0.5 * traj_loss
        
        # Compute and apply gradients
        gradients = tape.gradient(total_loss, self.model.trainable_variables)
        gradients = [tf.clip_by_norm(g, 10.0) for g in gradients]  # Gradient clipping
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        
        return total_loss, word_loss, traj_loss
    
    def train(self, X_train, y_train, X_val, y_val, epochs: int = 50, batch_size: int = 32):
        """Train the model"""
        
        train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
        train_dataset = train_dataset.shuffle(1000).batch(batch_size)
        
        val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val))
        val_dataset = val_dataset.batch(batch_size)
        
        history = {'train_loss': [], 'val_loss': []}
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch + 1}/{epochs}")
            
            # Training
            train_losses = []
            for batch_idx, (traj_batch, word_batch) in enumerate(train_dataset):
                total_loss, word_loss, traj_loss = self.train_step(traj_batch, word_batch)
                train_losses.append(total_loss.numpy())
                
                if batch_idx % 10 == 0:
                    print(f"  Batch {batch_idx}: loss={total_loss:.4f}, "
                          f"word_loss={word_loss:.4f}, traj_loss={traj_loss:.4f}")
            
            # Validation
            val_losses = []
            for traj_batch, word_batch in val_dataset:
                word_logits, mixture_params = self.model([traj_batch, word_batch], training=False)
                word_loss = self.word_loss_fn(word_batch[:, 1:], word_logits[:, :-1])
                traj_loss = self.mixture_loss(traj_batch, mixture_params)
                val_losses.append((word_loss + 0.5 * traj_loss).numpy())
            
            avg_train_loss = np.mean(train_losses)
            avg_val_loss = np.mean(val_losses)
            
            history['train_loss'].append(avg_train_loss)
            history['val_loss'].append(avg_val_loss)
            
            print(f"  Train Loss: {avg_train_loss:.4f}")
            print(f"  Val Loss: {avg_val_loss:.4f}")
            
            # Early stopping
            if epoch > 10 and history['val_loss'][-1] > history['val_loss'][-5]:
                print("Early stopping triggered")
                break
        
        return history


def main():
    """Main training function"""
    print("="*60)
    print("OPTIMIZED RNN TRAINING")
    print("="*60)
    
    # Load and preprocess data
    processor = SwipelogDataProcessor()
    data = processor.load_traces(limit=200)  # Start with subset
    
    if not data['traces']:
        print("❌ No traces loaded")
        return
    
    X, y = processor.preprocess_data(data['traces'], data['labels'])
    
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.15, random_state=42
    )
    
    print(f"\nTraining set: {len(X_train)} samples")
    print(f"Validation set: {len(X_val)} samples")
    
    # Create model
    model = OptimizedRNN(
        vocab_size=processor.vocab_size,
        lstm_units=256,
        mixture_components=20,
        dropout_rate=0.2
    )
    
    # Create trainer
    trainer = RNNTrainer(model, learning_rate=0.001)
    
    # Train model
    history = trainer.train(
        X_train, y_train, X_val, y_val,
        epochs=30,
        batch_size=32
    )
    
    # Save model
    model.save_weights('models/optimized_rnn_weights.h5')
    
    # Save processor info
    with open('models/processor_info.json', 'w') as f:
        json.dump({
            'vocab_size': processor.vocab_size,
            'char_to_idx': processor.char_to_idx,
            'max_seq_length': processor.max_seq_length
        }, f)
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print(f"Final train loss: {history['train_loss'][-1]:.4f}")
    print(f"Final val loss: {history['val_loss'][-1]:.4f}")
    print("Model saved to models/optimized_rnn_weights.h5")
    
    # Test generation
    print("\nTesting generation...")
    test_words = ['hello', 'world', 'the']
    for word in test_words:
        try:
            trajectory = model.generate_trajectory(word, processor.char_to_idx)
            print(f"  '{word}': Generated {len(trajectory)} points")
        except Exception as e:
            print(f"  '{word}': Generation failed - {e}")
    
    return history


if __name__ == "__main__":
    history = main()