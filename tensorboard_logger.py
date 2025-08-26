#!/usr/bin/env python
"""
Tensorboard logging integration for all trajectory generators
Provides centralized logging for training metrics, visualizations, and model comparisons
"""

import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
import json

# Try to import plotting libraries
try:
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    print("⚠️ Matplotlib not available, trajectory visualization disabled")
    MATPLOTLIB_AVAILABLE = False

# Try to import full Tensorboard, fall back to simple version
try:
    from torch.utils.tensorboard import SummaryWriter
    import io
    try:
        from PIL import Image
        PIL_AVAILABLE = True
    except ImportError:
        PIL_AVAILABLE = False
    TENSORBOARD_AVAILABLE = True and PIL_AVAILABLE and MATPLOTLIB_AVAILABLE
except ImportError:
    TENSORBOARD_AVAILABLE = False

if not TENSORBOARD_AVAILABLE:
    print("⚠️ Full Tensorboard not available, using simple logging")

class TrajectoryTensorboardLogger:
    """Enhanced Tensorboard logger for trajectory generation models"""
    
    def __init__(self, 
                 log_dir: str = "tensorboard_logs",
                 experiment_name: str = None,
                 model_name: str = "trajectory_generator"):
        
        self.model_name = model_name
        
        # Create experiment-specific directory
        if experiment_name is None:
            experiment_name = f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        self.log_dir = Path(log_dir) / experiment_name
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize appropriate logger
        if TENSORBOARD_AVAILABLE:
            self.writer = SummaryWriter(str(self.log_dir))
            self.simple_logger = None
            print(f"🔥 Full Tensorboard logger initialized: tensorboard --logdir={log_dir}")
        else:
            from simple_tensorboard_logger import SimpleTensorboardLogger
            self.simple_logger = SimpleTensorboardLogger(log_dir, experiment_name, model_name)
            self.writer = None
            print(f"📊 Simple logger initialized (Tensorboard not available)")
        
        # Store experiment metadata
        self.experiment_metadata = {
            'experiment_name': experiment_name,
            'model_name': model_name,
            'start_time': datetime.now().isoformat(),
            'log_dir': str(self.log_dir)
        }
        
        # Save metadata
        with open(self.log_dir / 'experiment_metadata.json', 'w') as f:
            json.dump(self.experiment_metadata, f, indent=2)
    
    def log_scalar(self, tag: str, value: float, step: int):
        """Log scalar metric"""
        if self.writer:
            self.writer.add_scalar(tag, value, step)
        elif self.simple_logger:
            self.simple_logger.log_scalar(tag, value, step)
    
    def log_scalars(self, main_tag: str, tag_scalar_dict: Dict[str, float], step: int):
        """Log multiple scalars under one main tag"""
        if self.writer:
            self.writer.add_scalars(main_tag, tag_scalar_dict, step)
        elif self.simple_logger:
            self.simple_logger.log_scalars(main_tag, tag_scalar_dict, step)
    
    def log_training_metrics(self, 
                           epoch: int, 
                           train_loss: float,
                           val_loss: Optional[float] = None,
                           learning_rate: Optional[float] = None,
                           additional_metrics: Optional[Dict[str, float]] = None):
        """Log training metrics"""
        
        self.log_scalar('Loss/Train', train_loss, epoch)
        
        if val_loss is not None:
            self.log_scalar('Loss/Validation', val_loss, epoch)
            self.log_scalars('Loss/Comparison', {
                'train': train_loss,
                'validation': val_loss
            }, epoch)
        
        if learning_rate is not None:
            self.log_scalar('Optimizer/Learning_Rate', learning_rate, epoch)
        
        # Log additional metrics
        if additional_metrics:
            for metric_name, value in additional_metrics.items():
                self.log_scalar(f'Metrics/{metric_name}', value, epoch)
    
    def log_trajectory_visualization(self, 
                                   trajectories: List[List[Dict]], 
                                   words: List[str],
                                   step: int,
                                   tag: str = "Trajectories"):
        """
        Visualize and log trajectory data
        
        Args:
            trajectories: List of trajectory data (list of {x, y, t, p} dicts)
            words: Corresponding words for each trajectory
            step: Training step
            tag: Tensorboard tag
        """
        
        if not MATPLOTLIB_AVAILABLE:
            print("⚠️ Cannot log trajectory visualization - matplotlib not available")
            return
        
        # Use simple logger if available
        if self.simple_logger:
            self.simple_logger.log_trajectory_visualization(trajectories, words, step, tag)
            return
        
        # Create figure with subplots
        n_trajectories = min(len(trajectories), 9)  # Max 9 trajectories in 3x3 grid
        fig, axes = plt.subplots(3, 3, figsize=(12, 12))
        axes = axes.flatten()
        
        for i in range(n_trajectories):
            ax = axes[i]
            trajectory = trajectories[i]
            word = words[i] if i < len(words) else f"Traj_{i}"
            
            # Extract coordinates and pen states
            x_coords = [p['x'] for p in trajectory]
            y_coords = [p['y'] for p in trajectory]
            pen_states = [p.get('p', 0) for p in trajectory]
            
            # Plot trajectory with different colors for pen up/down
            pen_down_indices = [j for j, p in enumerate(pen_states) if p == 0]
            pen_up_indices = [j for j, p in enumerate(pen_states) if p == 1]
            
            if pen_down_indices:
                ax.plot([x_coords[j] for j in pen_down_indices], 
                       [y_coords[j] for j in pen_down_indices], 
                       'b-', linewidth=2, label='Pen Down', alpha=0.8)
            
            if pen_up_indices:
                ax.plot([x_coords[j] for j in pen_up_indices], 
                       [y_coords[j] for j in pen_up_indices], 
                       'r--', linewidth=1, label='Pen Up', alpha=0.6)
            
            # Mark start and end points
            if x_coords and y_coords:
                ax.plot(x_coords[0], y_coords[0], 'go', markersize=8, label='Start')
                ax.plot(x_coords[-1], y_coords[-1], 'ro', markersize=8, label='End')
            
            ax.set_title(f'Word: "{word}"')
            ax.set_xlabel('X coordinate')
            ax.set_ylabel('Y coordinate')
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)
            ax.set_aspect('equal', adjustable='box')
        
        # Hide unused subplots
        for i in range(n_trajectories, 9):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        
        if self.writer and TENSORBOARD_AVAILABLE:
            # Convert plot to image and log to tensorboard
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
            buf.seek(0)
            
            # Convert to PIL Image then to tensor
            image = Image.open(buf)
            image_tensor = torch.tensor(np.array(image)).permute(2, 0, 1)
            
            self.writer.add_image(tag, image_tensor, step)
            buf.close()
        elif self.simple_logger:
            # Use simple logger's trajectory visualization
            self.simple_logger.log_trajectory_visualization(trajectories, words, step, tag)
            # Don't save the plot again since simple_logger handles it
            plt.close(fig)
            return
        
        plt.close(fig)
    
    def log_model_parameters(self, model: torch.nn.Module, step: int):
        """Log model parameter histograms and gradients"""
        
        if self.writer:
            for name, param in model.named_parameters():
                if param.requires_grad:
                    # Log parameter values
                    self.writer.add_histogram(f'Parameters/{name}', param.data, step)
                    
                    # Log gradients if available
                    if param.grad is not None:
                        self.writer.add_histogram(f'Gradients/{name}', param.grad, step)
                        
                        # Log gradient norms
                        grad_norm = param.grad.norm().item()
                        self.writer.add_scalar(f'Gradient_Norms/{name}', grad_norm, step)
        elif self.simple_logger:
            self.simple_logger.log_model_parameters(model, step)
    
    def log_attention_weights(self, 
                            attention_weights: torch.Tensor,
                            input_sequence: List[str],
                            step: int,
                            tag: str = "Attention"):
        """
        Log attention weight visualizations
        
        Args:
            attention_weights: Tensor of shape [seq_len, seq_len] or [batch, heads, seq_len, seq_len]
            input_sequence: List of characters/tokens
            step: Training step
            tag: Tensorboard tag
        """
        
        # Handle different attention tensor shapes
        if attention_weights.dim() == 4:  # [batch, heads, seq, seq]
            attention_weights = attention_weights[0, 0]  # Take first batch, first head
        elif attention_weights.dim() == 3:  # [batch, seq, seq]
            attention_weights = attention_weights[0]  # Take first batch
        
        # Convert to numpy
        attention_np = attention_weights.detach().cpu().numpy()
        
        # Create attention heatmap
        fig, ax = plt.subplots(figsize=(10, 8))
        
        im = ax.imshow(attention_np, cmap='Blues', aspect='auto')
        
        # Set tick labels
        ax.set_xticks(range(len(input_sequence)))
        ax.set_yticks(range(len(input_sequence)))
        ax.set_xticklabels(input_sequence)
        ax.set_yticklabels(input_sequence)
        
        # Rotate labels for better readability
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        # Add colorbar
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        ax.set_title('Attention Weights')
        ax.set_xlabel('Input Position')
        ax.set_ylabel('Output Position')
        
        plt.tight_layout()
        
        # Convert to tensor and log
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        
        image = Image.open(buf)
        image_tensor = torch.tensor(np.array(image)).permute(2, 0, 1)
        
        self.writer.add_image(tag, image_tensor, step)
        
        plt.close(fig)
        buf.close()
    
    def log_loss_components(self, 
                           loss_dict: Dict[str, float], 
                           step: int,
                           prefix: str = "Loss_Components"):
        """Log individual loss components for complex losses"""
        
        for loss_name, loss_value in loss_dict.items():
            self.writer.add_scalar(f'{prefix}/{loss_name}', loss_value, step)
        
        # Also log as grouped scalars
        self.writer.add_scalars(prefix, loss_dict, step)
    
    def log_hyperparameters(self, 
                           hparams: Dict[str, Any], 
                           metrics: Dict[str, float]):
        """Log hyperparameters and final metrics"""
        
        # Convert non-serializable values to strings
        hparams_clean = {}
        for key, value in hparams.items():
            if isinstance(value, (int, float, str, bool)):
                hparams_clean[key] = value
            else:
                hparams_clean[key] = str(value)
        
        if self.writer:
            self.writer.add_hparams(hparams_clean, metrics)
        elif self.simple_logger:
            self.simple_logger.log_hyperparameters(hparams_clean, metrics)
    
    def log_model_architecture(self, model: torch.nn.Module, input_size: torch.Size):
        """Log model architecture graph"""
        
        try:
            # Create dummy input
            dummy_input = torch.randn(input_size)
            
            # Log model graph
            self.writer.add_graph(model, dummy_input)
            
            # Calculate and log model statistics
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            # Log as text
            model_info = f"""
            Model Architecture: {model.__class__.__name__}
            Total Parameters: {total_params:,}
            Trainable Parameters: {trainable_params:,}
            Model Size (MB): {total_params * 4 / (1024**2):.2f}
            """
            
            self.writer.add_text('Model/Architecture', model_info, 0)
            
        except Exception as e:
            print(f"Failed to log model architecture: {e}")
    
    def log_dataset_statistics(self, 
                             dataset_stats: Dict[str, Any], 
                             step: int = 0):
        """Log dataset statistics and sample distributions"""
        
        # Log basic statistics as scalars
        for stat_name, stat_value in dataset_stats.items():
            if isinstance(stat_value, (int, float)):
                self.writer.add_scalar(f'Dataset/{stat_name}', stat_value, step)
            elif isinstance(stat_value, str):
                self.writer.add_text(f'Dataset/{stat_name}', stat_value, step)
    
    def close(self):
        """Close the tensorboard writer"""
        
        # Update experiment metadata with end time
        self.experiment_metadata['end_time'] = datetime.now().isoformat()
        
        with open(self.log_dir / 'experiment_metadata.json', 'w') as f:
            json.dump(self.experiment_metadata, f, indent=2)
        
        if self.writer:
            self.writer.close()
            print(f"✅ Tensorboard logs saved to: {self.log_dir}")
        elif self.simple_logger:
            self.simple_logger.close()
            print(f"✅ Simple logs saved to: {self.log_dir}")

# Training integration mixins
class TensorboardTrainingMixin:
    """Mixin class to add Tensorboard logging to training classes"""
    
    def init_tensorboard(self, 
                        log_dir: str = "tensorboard_logs",
                        experiment_name: str = None,
                        model_name: str = None):
        """Initialize Tensorboard logger"""
        
        if model_name is None:
            model_name = self.__class__.__name__
        
        self.tb_logger = TrajectoryTensorboardLogger(
            log_dir=log_dir,
            experiment_name=experiment_name,
            model_name=model_name
        )
        
        return self.tb_logger
    
    def log_training_step(self, 
                         epoch: int,
                         train_loss: float,
                         val_loss: Optional[float] = None,
                         **kwargs):
        """Convenience method to log training step"""
        
        if hasattr(self, 'tb_logger'):
            self.tb_logger.log_training_metrics(
                epoch=epoch,
                train_loss=train_loss,
                val_loss=val_loss,
                **kwargs
            )
    
    def log_model_checkpoint(self, model: torch.nn.Module, step: int):
        """Log model parameters and gradients at checkpoint"""
        
        if hasattr(self, 'tb_logger'):
            self.tb_logger.log_model_parameters(model, step)
    
    def close_tensorboard(self):
        """Close Tensorboard logger"""
        
        if hasattr(self, 'tb_logger'):
            self.tb_logger.close()

# Quick usage example
if __name__ == "__main__":
    # Test the logger
    logger = TrajectoryTensorboardLogger(experiment_name="test_experiment")
    
    # Log some sample data
    logger.log_scalar('test/loss', 0.5, 0)
    
    # Create sample trajectory data
    sample_trajectory = [
        {'x': 100 + i*10, 'y': 200 + i*5, 't': i*16.67, 'p': 0 if i < 5 else 1}
        for i in range(10)
    ]
    
    logger.log_trajectory_visualization([sample_trajectory], ["test"], 0)
    
    logger.close()
    print("✅ Tensorboard test completed")