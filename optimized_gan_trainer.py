#!/usr/bin/env python
"""
Optimized GAN trainer for gesture keyboard trajectory generation
Enhanced with Wasserstein loss and gradient penalty for stable training
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

class TrajectoryGAN:
    """Wasserstein GAN with Gradient Penalty for trajectory generation"""
    
    def __init__(self, vocab_size: int, latent_dim: int = 100, 
                 max_seq_length: int = 150):
        self.vocab_size = vocab_size
        self.latent_dim = latent_dim
        self.max_seq_length = max_seq_length
        self.gp_weight = 10.0  # Gradient penalty weight
        self.d_steps = 5  # Discriminator steps per generator step
        
        # Build models
        self.generator = self.build_generator()
        self.discriminator = self.build_discriminator()
        
        # Optimizers
        self.g_optimizer = keras.optimizers.Adam(0.0001, beta_1=0.5, beta_2=0.9)
        self.d_optimizer = keras.optimizers.Adam(0.0001, beta_1=0.5, beta_2=0.9)
    
    def build_generator(self):
        """Build generator network"""
        
        # Word input
        word_input = layers.Input(shape=(None,), dtype=tf.int32)
        word_embedding = layers.Embedding(self.vocab_size, 128)(word_input)
        
        # Latent noise input
        noise_input = layers.Input(shape=(self.latent_dim,))
        noise_dense = layers.Dense(256)(noise_input)
        noise_expanded = layers.RepeatVector(self.max_seq_length)(noise_dense)
        
        # Combine word and noise
        word_lstm = layers.LSTM(256, return_sequences=True)(word_embedding)
        combined = layers.Concatenate()([word_lstm, noise_expanded])
        
        # Generate trajectory
        x = layers.LSTM(512, return_sequences=True)(combined)
        x = layers.BatchNormalization()(x)
        x = layers.LSTM(256, return_sequences=True)(x)
        x = layers.BatchNormalization()(x)
        
        # Output trajectory points (x, y, end_of_stroke)
        trajectory_output = layers.TimeDistributed(
            layers.Dense(3, activation='tanh')
        )(x)
        
        return keras.Model([word_input, noise_input], trajectory_output, name='generator')
    
    def build_discriminator(self):
        """Build discriminator network (critic for WGAN)"""
        
        # Trajectory input
        traj_input = layers.Input(shape=(self.max_seq_length, 3))
        
        # Word condition input
        word_input = layers.Input(shape=(None,), dtype=tf.int32)
        word_embedding = layers.Embedding(self.vocab_size, 64)(word_input)
        word_features = layers.LSTM(128)(word_embedding)
        word_expanded = layers.RepeatVector(self.max_seq_length)(word_features)
        
        # Combine trajectory and word
        combined = layers.Concatenate()([traj_input, word_expanded])
        
        # Discriminator layers
        x = layers.LSTM(256, return_sequences=True)(combined)
        x = layers.Dropout(0.3)(x)
        x = layers.LSTM(128)(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(64, activation='relu')(x)
        
        # Output (no activation for WGAN)
        validity = layers.Dense(1)(x)
        
        return keras.Model([traj_input, word_input], validity, name='discriminator')
    
    def gradient_penalty(self, real_trajectories, fake_trajectories, word_conditions):
        """Calculate gradient penalty for WGAN-GP"""
        
        batch_size = tf.shape(real_trajectories)[0]
        
        # Random interpolation
        alpha = tf.random.uniform([batch_size, 1, 1], 0., 1.)
        interpolated = alpha * real_trajectories + (1 - alpha) * fake_trajectories
        
        with tf.GradientTape() as tape:
            tape.watch(interpolated)
            pred = self.discriminator([interpolated, word_conditions])
        
        grads = tape.gradient(pred, interpolated)
        norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2]) + 1e-10)
        gradient_penalty = tf.reduce_mean((norm - 1.0) ** 2)
        
        return gradient_penalty
    
    @tf.function
    def train_discriminator(self, real_trajectories, word_conditions):
        """Train discriminator step"""
        
        batch_size = tf.shape(real_trajectories)[0]
        
        # Generate fake trajectories
        noise = tf.random.normal([batch_size, self.latent_dim])
        fake_trajectories = self.generator([word_conditions, noise])
        
        with tf.GradientTape() as tape:
            # Real predictions
            real_pred = self.discriminator([real_trajectories, word_conditions])
            
            # Fake predictions
            fake_pred = self.discriminator([fake_trajectories, word_conditions])
            
            # Wasserstein loss
            d_loss = tf.reduce_mean(fake_pred) - tf.reduce_mean(real_pred)
            
            # Gradient penalty
            gp = self.gradient_penalty(real_trajectories, fake_trajectories, word_conditions)
            
            # Total loss
            total_d_loss = d_loss + self.gp_weight * gp
        
        # Update discriminator
        d_gradients = tape.gradient(total_d_loss, self.discriminator.trainable_variables)
        self.d_optimizer.apply_gradients(zip(d_gradients, self.discriminator.trainable_variables))
        
        return d_loss, gp
    
    @tf.function
    def train_generator(self, word_conditions):
        """Train generator step"""
        
        batch_size = tf.shape(word_conditions)[0]
        noise = tf.random.normal([batch_size, self.latent_dim])
        
        with tf.GradientTape() as tape:
            # Generate fake trajectories
            fake_trajectories = self.generator([word_conditions, noise])
            
            # Get discriminator predictions
            fake_pred = self.discriminator([fake_trajectories, word_conditions])
            
            # Generator loss (negative critic score)
            g_loss = -tf.reduce_mean(fake_pred)
        
        # Update generator
        g_gradients = tape.gradient(g_loss, self.generator.trainable_variables)
        self.g_optimizer.apply_gradients(zip(g_gradients, self.generator.trainable_variables))
        
        return g_loss
    
    def train(self, X_train, y_train, epochs: int = 100, batch_size: int = 32):
        """Train the GAN"""
        
        dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
        dataset = dataset.shuffle(1000).batch(batch_size, drop_remainder=True)
        
        history = {'d_loss': [], 'g_loss': [], 'gp': []}
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch + 1}/{epochs}")
            
            d_losses = []
            g_losses = []
            gp_values = []
            
            for batch_idx, (traj_batch, word_batch) in enumerate(dataset):
                # Train discriminator multiple times
                for _ in range(self.d_steps):
                    d_loss, gp = self.train_discriminator(traj_batch, word_batch)
                    d_losses.append(d_loss.numpy())
                    gp_values.append(gp.numpy())
                
                # Train generator once
                g_loss = self.train_generator(word_batch)
                g_losses.append(g_loss.numpy())
                
                if batch_idx % 10 == 0:
                    print(f"  Batch {batch_idx}: d_loss={np.mean(d_losses[-5:]):.4f}, "
                          f"g_loss={g_loss:.4f}, gp={np.mean(gp_values[-5:]):.4f}")
            
            # Record history
            history['d_loss'].append(np.mean(d_losses))
            history['g_loss'].append(np.mean(g_losses))
            history['gp'].append(np.mean(gp_values))
            
            print(f"  Epoch Summary - D Loss: {history['d_loss'][-1]:.4f}, "
                  f"G Loss: {history['g_loss'][-1]:.4f}, "
                  f"GP: {history['gp'][-1]:.4f}")
            
            # Generate samples for visualization
            if (epoch + 1) % 10 == 0:
                self.generate_samples(word_batch[:3], epoch)
        
        return history
    
    def generate_samples(self, word_conditions, epoch):
        """Generate sample trajectories for visualization"""
        
        noise = tf.random.normal([len(word_conditions), self.latent_dim])
        generated = self.generator([word_conditions, noise])
        
        print(f"\n  Sample trajectories at epoch {epoch + 1}:")
        for i in range(min(3, len(generated))):
            traj = generated[i].numpy()
            # Find end of stroke
            eos_idx = np.where(traj[:, 2] > 0.5)[0]
            if len(eos_idx) > 0:
                traj_len = eos_idx[0] + 1
            else:
                traj_len = len(traj)
            
            print(f"    Sample {i+1}: {traj_len} points")
    
    def generate_trajectory(self, word_encoded: List[int]) -> np.ndarray:
        """Generate trajectory for encoded word"""
        
        word_tensor = tf.constant([word_encoded])
        noise = tf.random.normal([1, self.latent_dim])
        
        trajectory = self.generator([word_tensor, noise]).numpy()[0]
        
        # Find end of stroke
        eos_idx = np.where(trajectory[:, 2] > 0.5)[0]
        if len(eos_idx) > 0:
            trajectory = trajectory[:eos_idx[0] + 1]
        
        return trajectory


class GANDataProcessor:
    """Process data for GAN training"""
    
    def __init__(self):
        self.char_to_idx = {}
        self.idx_to_char = {}
        self.vocab_size = 0
        
    def load_and_process(self, real_dir: str = "datasets/swipelogs",
                        synthetic_dir: str = "datasets/synthetic_swipelogs",
                        limit: int = None) -> Tuple:
        """Load and process trajectory data"""
        
        print("Loading trajectory data for GAN...")
        
        traces = []
        words = []
        
        # Process both real and synthetic data
        for data_dir in [Path(real_dir), Path(synthetic_dir)]:
            if not data_dir.exists():
                continue
                
            for log_file in list(data_dir.glob("*.log"))[:limit]:
                try:
                    df = pd.read_csv(log_file, sep='\s+', engine='python', 
                                    on_bad_lines='skip', header=0)
                    
                    if 'word' in df.columns and 'x_pos' in df.columns:
                        for word in df['word'].unique():
                            if pd.isna(word) or word == 'word':
                                continue
                            
                            word_df = df[df['word'] == word]
                            traj = word_df[['x_pos', 'y_pos']].values
                            
                            if len(traj) > 2 and len(traj) < 150:
                                # Normalize trajectory
                                traj = (traj - traj.mean(axis=0)) / (traj.std(axis=0) + 1e-6)
                                
                                # Add end-of-stroke marker
                                eos = np.zeros((len(traj), 1))
                                eos[-1] = 1
                                traj = np.concatenate([traj, eos], axis=1)
                                
                                traces.append(traj)
                                words.append(word.lower())
                
                except Exception:
                    continue
        
        print(f"Loaded {len(traces)} trajectories")
        
        # Build vocabulary
        chars = set('<PAD><START><END>')
        for word in words:
            chars.update(word)
        
        self.char_to_idx = {char: idx for idx, char in enumerate(sorted(chars))}
        self.idx_to_char = {idx: char for char, idx in self.char_to_idx.items()}
        self.vocab_size = len(self.char_to_idx)
        
        # Encode words
        encoded_words = []
        for word in words:
            encoded = [self.char_to_idx['<START>']]
            for char in word:
                if char in self.char_to_idx:
                    encoded.append(self.char_to_idx[char])
            encoded.append(self.char_to_idx['<END>'])
            encoded_words.append(encoded)
        
        # Pad sequences
        X = tf.keras.preprocessing.sequence.pad_sequences(
            traces, maxlen=150, dtype='float32', padding='post'
        )
        y = tf.keras.preprocessing.sequence.pad_sequences(
            encoded_words, padding='post', value=self.char_to_idx['<PAD>']
        )
        
        return X, y


def main():
    """Main training function"""
    
    print("="*60)
    print("OPTIMIZED GAN TRAINING")
    print("="*60)
    
    # Load data
    processor = GANDataProcessor()
    X, y = processor.load_and_process(limit=200)
    
    if len(X) == 0:
        print("❌ No data loaded")
        return
    
    print(f"\nDataset size: {len(X)} samples")
    print(f"Vocabulary size: {processor.vocab_size}")
    
    # Create GAN
    gan = TrajectoryGAN(
        vocab_size=processor.vocab_size,
        latent_dim=100,
        max_seq_length=150
    )
    
    # Train
    history = gan.train(X, y, epochs=50, batch_size=32)
    
    # Save models
    os.makedirs('models', exist_ok=True)
    gan.generator.save('models/gan_generator.h5')
    gan.discriminator.save('models/gan_discriminator.h5')
    
    # Save processor info
    with open('models/gan_processor_info.json', 'w') as f:
        json.dump({
            'vocab_size': processor.vocab_size,
            'char_to_idx': processor.char_to_idx
        }, f)
    
    print("\n" + "="*60)
    print("GAN TRAINING COMPLETE")
    print("="*60)
    print(f"Final D Loss: {history['d_loss'][-1]:.4f}")
    print(f"Final G Loss: {history['g_loss'][-1]:.4f}")
    print("Models saved to models/")
    
    # Test generation
    print("\nTesting generation...")
    test_words = ['hello', 'world', 'the']
    for word in test_words:
        encoded = [processor.char_to_idx.get(c, 0) for c in word]
        trajectory = gan.generate_trajectory(encoded)
        print(f"  '{word}': Generated {len(trajectory)} points")
    
    return history


if __name__ == "__main__":
    history = main()