#!/usr/bin/env python
"""
Train RNN model on generated synthetic dataset
Using simplified RNN architecture for gesture trajectory learning
"""

import numpy as np
import pickle
import json
from pathlib import Path
import random
from typing import List, Tuple, Dict

# Simple RNN implementation without TensorFlow dependency
class SimpleGestureRNN:
    """
    Simplified RNN for gesture trajectory generation
    Using numpy for maximum compatibility
    """
    
    def __init__(self, input_dim: int = 3, hidden_dim: int = 128, output_dim: int = 3):
        """
        Initialize RNN with random weights
        
        Args:
            input_dim: Input features (dx, dy, pen_state)
            hidden_dim: Hidden state dimensions
            output_dim: Output features
        """
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # Initialize weights with Xavier initialization
        self.Wxh = np.random.randn(input_dim, hidden_dim) * np.sqrt(2.0 / input_dim)
        self.Whh = np.random.randn(hidden_dim, hidden_dim) * np.sqrt(2.0 / hidden_dim)
        self.Why = np.random.randn(hidden_dim, output_dim) * np.sqrt(2.0 / hidden_dim)
        
        # Biases
        self.bh = np.zeros((1, hidden_dim))
        self.by = np.zeros((1, output_dim))
        
        # Training history
        self.loss_history = []
        
    def forward(self, inputs: np.ndarray, h_prev: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Forward pass through RNN
        
        Args:
            inputs: Sequence of input vectors (seq_len, input_dim)
            h_prev: Previous hidden state
            
        Returns:
            outputs: Sequence outputs
            h_states: Hidden states
        """
        seq_len = inputs.shape[0]
        
        if h_prev is None:
            h_prev = np.zeros((1, self.hidden_dim))
        
        outputs = []
        h_states = []
        
        for t in range(seq_len):
            # RNN cell computation
            x_t = inputs[t:t+1]
            h_t = np.tanh(np.dot(x_t, self.Wxh) + np.dot(h_prev, self.Whh) + self.bh)
            y_t = np.dot(h_t, self.Why) + self.by
            
            outputs.append(y_t)
            h_states.append(h_t)
            h_prev = h_t
        
        return np.vstack(outputs), np.vstack(h_states)
    
    def compute_loss(self, predicted: np.ndarray, target: np.ndarray) -> float:
        """Compute MSE loss"""
        return np.mean((predicted - target) ** 2)
    
    def train_step(self, inputs: np.ndarray, targets: np.ndarray, 
                   learning_rate: float = 0.001) -> float:
        """
        Single training step with backpropagation through time
        
        Args:
            inputs: Input sequence
            targets: Target sequence
            learning_rate: Learning rate
            
        Returns:
            loss: Training loss
        """
        # Forward pass
        outputs, h_states = self.forward(inputs)
        
        # Compute loss
        loss = self.compute_loss(outputs, targets)
        
        # Backward pass (simplified gradient descent)
        # For demonstration, using numerical gradients
        # In production, would use automatic differentiation
        
        # Gradient of output layer
        d_outputs = 2 * (outputs - targets) / len(outputs)
        
        # Simple weight update (placeholder for full BPTT)
        self.Why -= learning_rate * np.dot(h_states.T, d_outputs)
        self.by -= learning_rate * np.sum(d_outputs, axis=0, keepdims=True)
        
        self.loss_history.append(loss)
        return loss

def load_dataset(data_path: str = 'training_data/rnn_data.pkl') -> Dict:
    """Load the prepared RNN dataset"""
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    return data

def prepare_batch(strokes: List, batch_size: int = 32) -> Tuple[List, List]:
    """
    Prepare a batch of sequences for training
    
    Args:
        strokes: List of stroke sequences
        batch_size: Batch size
        
    Returns:
        batch_inputs: Batch of input sequences
        batch_targets: Batch of target sequences
    """
    # Random sampling
    indices = random.sample(range(len(strokes)), min(batch_size, len(strokes)))
    
    batch_inputs = []
    batch_targets = []
    
    for idx in indices:
        stroke = strokes[idx]
        if len(stroke) > 1:
            # Input is all but last point
            batch_inputs.append(stroke[:-1])
            # Target is all but first point
            batch_targets.append(stroke[1:])
    
    return batch_inputs, batch_targets

def train_model(model: SimpleGestureRNN, dataset: Dict, 
                num_epochs: int = 10, batch_size: int = 32) -> SimpleGestureRNN:
    """
    Train the RNN model
    
    Args:
        model: RNN model
        dataset: Training dataset
        num_epochs: Number of training epochs
        batch_size: Batch size
        
    Returns:
        Trained model
    """
    strokes = dataset['strokes']
    labels = dataset['labels']
    
    print(f"Training on {len(strokes)} sequences")
    print(f"Vocabulary size: {len(dataset['vocabulary'])}")
    
    # Split data
    n_train = int(0.8 * len(strokes))
    train_strokes = strokes[:n_train]
    val_strokes = strokes[n_train:]
    
    print(f"Train: {len(train_strokes)}, Val: {len(val_strokes)}")
    
    for epoch in range(num_epochs):
        # Training
        epoch_loss = []
        for _ in range(len(train_strokes) // batch_size):
            batch_inputs, batch_targets = prepare_batch(train_strokes, batch_size)
            
            # Train on each sequence in batch
            batch_losses = []
            for inp, tgt in zip(batch_inputs, batch_targets):
                loss = model.train_step(inp, tgt)
                batch_losses.append(loss)
            
            epoch_loss.extend(batch_losses)
        
        # Validation
        val_losses = []
        for _ in range(min(10, len(val_strokes) // batch_size)):
            batch_inputs, batch_targets = prepare_batch(val_strokes, batch_size)
            
            for inp, tgt in zip(batch_inputs, batch_targets):
                outputs, _ = model.forward(inp)
                loss = model.compute_loss(outputs, tgt)
                val_losses.append(loss)
        
        # Print progress
        avg_train_loss = np.mean(epoch_loss) if epoch_loss else 0
        avg_val_loss = np.mean(val_losses) if val_losses else 0
        
        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
    
    return model

def generate_trajectory(model: SimpleGestureRNN, word: str, 
                       vocabulary: List[str], seed_length: int = 5) -> np.ndarray:
    """
    Generate trajectory for a word using trained model
    
    Args:
        model: Trained RNN model
        word: Word to generate trajectory for
        vocabulary: Word vocabulary
        seed_length: Length of seed sequence
        
    Returns:
        Generated trajectory
    """
    # Create seed sequence (simplified)
    seed = np.random.randn(seed_length, 3) * 0.1
    
    # Generate trajectory
    trajectory, _ = model.forward(seed)
    
    # Continue generation (auto-regressive)
    for _ in range(len(word) * 10):  # Approximate length
        last_output = trajectory[-1:, :]
        next_output, _ = model.forward(last_output)
        trajectory = np.vstack([trajectory, next_output])
    
    return trajectory

def evaluate_model(model: SimpleGestureRNN, dataset: Dict) -> Dict:
    """
    Evaluate trained model
    
    Args:
        model: Trained model
        dataset: Test dataset
        
    Returns:
        Evaluation metrics
    """
    strokes = dataset['strokes']
    
    # Calculate metrics on test set
    test_losses = []
    trajectory_lengths = []
    
    for stroke in strokes[-100:]:  # Last 100 samples as test
        if len(stroke) > 1:
            inputs = stroke[:-1]
            targets = stroke[1:]
            
            outputs, _ = model.forward(inputs)
            loss = model.compute_loss(outputs, targets)
            test_losses.append(loss)
            trajectory_lengths.append(len(stroke))
    
    metrics = {
        'mean_test_loss': np.mean(test_losses),
        'std_test_loss': np.std(test_losses),
        'mean_trajectory_length': np.mean(trajectory_lengths),
        'total_parameters': model.Wxh.size + model.Whh.size + model.Why.size + model.bh.size + model.by.size
    }
    
    return metrics

def save_model(model: SimpleGestureRNN, path: str = 'models/rnn_gesture_model.pkl'):
    """Save trained model"""
    Path(path).parent.mkdir(exist_ok=True)
    
    model_data = {
        'Wxh': model.Wxh,
        'Whh': model.Whh,
        'Why': model.Why,
        'bh': model.bh,
        'by': model.by,
        'config': {
            'input_dim': model.input_dim,
            'hidden_dim': model.hidden_dim,
            'output_dim': model.output_dim
        },
        'loss_history': model.loss_history
    }
    
    with open(path, 'wb') as f:
        pickle.dump(model_data, f)
    
    print(f"Model saved to {path}")

def main():
    """Main training pipeline"""
    print("="*60)
    print("RNN GESTURE MODEL TRAINING")
    print("="*60)
    
    # Load dataset
    print("\nLoading dataset...")
    dataset = load_dataset()
    
    # Create model
    print("\nInitializing model...")
    model = SimpleGestureRNN(input_dim=3, hidden_dim=128, output_dim=3)
    print(f"Model parameters: {model.Wxh.size + model.Whh.size + model.Why.size + model.bh.size + model.by.size}")
    
    # Train model
    print("\nTraining model...")
    model = train_model(model, dataset, num_epochs=10, batch_size=32)
    
    # Evaluate model
    print("\nEvaluating model...")
    metrics = evaluate_model(model, dataset)
    
    print("\nEvaluation Metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")
    
    # Generate sample trajectories
    print("\nGenerating sample trajectories...")
    sample_words = ['hello', 'world', 'gesture']
    
    for word in sample_words:
        trajectory = generate_trajectory(model, word, dataset['vocabulary'])
        print(f"  Generated trajectory for '{word}': shape {trajectory.shape}")
    
    # Save model
    save_model(model)
    
    # Save metrics
    with open('models/rnn_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print("\n✅ Training complete!")
    
    return model, metrics

if __name__ == "__main__":
    model, metrics = main()