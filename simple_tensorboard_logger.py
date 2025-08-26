#!/usr/bin/env python
"""
Simple Tensorboard-like logger for environments where full Tensorboard isn't available
Logs to JSON files and creates simple visualizations
"""

import torch
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from datetime import datetime

# Try to import matplotlib
try:
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    print("⚠️ Matplotlib not available - visualization disabled")
    MATPLOTLIB_AVAILABLE = False

class SimpleTensorboardLogger:
    """Simplified Tensorboard logger that writes to JSON and creates plots"""
    
    def __init__(self, 
                 log_dir: str = "simple_tb_logs",
                 experiment_name: str = None,
                 model_name: str = "model"):
        
        self.model_name = model_name
        
        # Create experiment-specific directory
        if experiment_name is None:
            experiment_name = f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        self.log_dir = Path(log_dir) / experiment_name
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Storage for different types of logs
        self.scalars = {}
        self.images_dir = self.log_dir / "images"
        self.images_dir.mkdir(exist_ok=True)
        
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
        
        print(f"📊 Simple logger initialized: {self.log_dir}")
        print(f"📈 View logs: python -m http.server 8080 (in {log_dir})")
    
    def log_scalar(self, tag: str, value: float, step: int):
        """Log scalar metric"""
        if tag not in self.scalars:
            self.scalars[tag] = {'steps': [], 'values': []}
        
        self.scalars[tag]['steps'].append(step)
        self.scalars[tag]['values'].append(float(value))
        
        # Save to file periodically
        if step % 100 == 0:
            self._save_scalars()
    
    def log_scalars(self, main_tag: str, tag_scalar_dict: Dict[str, float], step: int):
        """Log multiple scalars under one main tag"""
        for tag, value in tag_scalar_dict.items():
            self.log_scalar(f"{main_tag}/{tag}", value, step)
    
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
        """
        
        if not MATPLOTLIB_AVAILABLE:
            print(f"⚠️ Cannot create trajectory visualization for {tag} - matplotlib not available")
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
        
        # Save plot
        plot_path = self.images_dir / f"{tag}_step_{step:06d}.png"
        plt.savefig(plot_path, dpi=100, bbox_inches='tight')
        plt.close(fig)
        
        print(f"📊 Saved trajectory visualization: {plot_path}")
    
    def log_model_parameters(self, model: torch.nn.Module, step: int):
        """Log model parameter statistics"""
        
        param_stats = {}
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                # Log parameter statistics
                param_data = param.data.cpu().numpy().flatten()
                param_stats[f'{name}/mean'] = float(np.mean(param_data))
                param_stats[f'{name}/std'] = float(np.std(param_data))
                param_stats[f'{name}/min'] = float(np.min(param_data))
                param_stats[f'{name}/max'] = float(np.max(param_data))
                
                # Log gradients if available
                if param.grad is not None:
                    grad_data = param.grad.cpu().numpy().flatten()
                    param_stats[f'{name}_grad/norm'] = float(np.linalg.norm(grad_data))
                    param_stats[f'{name}_grad/mean'] = float(np.mean(grad_data))
        
        # Log all parameter stats
        for stat_name, value in param_stats.items():
            self.log_scalar(f'Parameters/{stat_name}', value, step)
    
    def log_loss_components(self, 
                           loss_dict: Dict[str, float], 
                           step: int,
                           prefix: str = "Loss_Components"):
        """Log individual loss components"""
        
        for loss_name, loss_value in loss_dict.items():
            self.log_scalar(f'{prefix}/{loss_name}', loss_value, step)
    
    def log_hyperparameters(self, 
                           hparams: Dict[str, Any], 
                           metrics: Dict[str, float]):
        """Log hyperparameters and final metrics"""
        
        # Save hyperparameters and metrics to JSON
        hparams_file = self.log_dir / 'hyperparameters.json'
        
        data = {
            'hyperparameters': hparams,
            'final_metrics': metrics,
            'logged_at': datetime.now().isoformat()
        }
        
        with open(hparams_file, 'w') as f:
            json.dump(data, f, indent=2, default=str)
    
    def log_model_architecture(self, model: torch.nn.Module, input_size: torch.Size):
        """Log model architecture information"""
        
        try:
            # Calculate model statistics
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            model_info = {
                'model_class': model.__class__.__name__,
                'total_parameters': total_params,
                'trainable_parameters': trainable_params,
                'model_size_mb': total_params * 4 / (1024**2),
                'input_size': list(input_size),
                'architecture': str(model),
                'logged_at': datetime.now().isoformat()
            }
            
            # Save to file
            arch_file = self.log_dir / 'model_architecture.json'
            with open(arch_file, 'w') as f:
                json.dump(model_info, f, indent=2)
            
            print(f"📋 Model info saved: {arch_file}")
            
        except Exception as e:
            print(f"Failed to log model architecture: {e}")
    
    def log_dataset_statistics(self, 
                             dataset_stats: Dict[str, Any], 
                             step: int = 0):
        """Log dataset statistics"""
        
        # Save dataset statistics
        stats_file = self.log_dir / 'dataset_statistics.json'
        
        data = {
            'statistics': dataset_stats,
            'logged_at': datetime.now().isoformat()
        }
        
        with open(stats_file, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        
        # Also log numeric stats as scalars
        for stat_name, stat_value in dataset_stats.items():
            if isinstance(stat_value, (int, float)):
                self.log_scalar(f'Dataset/{stat_name}', stat_value, step)
    
    def _save_scalars(self):
        """Save scalar logs to JSON"""
        scalars_file = self.log_dir / 'scalars.json'
        with open(scalars_file, 'w') as f:
            json.dump(self.scalars, f, indent=2)
    
    def create_plots(self):
        """Create summary plots from logged data"""
        
        if not self.scalars or not MATPLOTLIB_AVAILABLE:
            if not MATPLOTLIB_AVAILABLE:
                print("⚠️ Cannot create plots - matplotlib not available")
            return
        
        # Create plots directory
        plots_dir = self.log_dir / "plots"
        plots_dir.mkdir(exist_ok=True)
        
        # Group related metrics
        metric_groups = {}
        for tag in self.scalars.keys():
            group = tag.split('/')[0] if '/' in tag else 'General'
            if group not in metric_groups:
                metric_groups[group] = []
            metric_groups[group].append(tag)
        
        # Create plots for each group
        for group_name, tags in metric_groups.items():
            plt.figure(figsize=(12, 8))
            
            for tag in tags:
                steps = self.scalars[tag]['steps']
                values = self.scalars[tag]['values']
                plt.plot(steps, values, label=tag, marker='o', markersize=2)
            
            plt.title(f'{group_name} Metrics Over Time')
            plt.xlabel('Step')
            plt.ylabel('Value')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            plot_path = plots_dir / f"{group_name.lower()}_metrics.png"
            plt.savefig(plot_path, dpi=100, bbox_inches='tight')
            plt.close()
            
            print(f"📊 Created plot: {plot_path}")
    
    def close(self):
        """Close the logger and create final outputs"""
        
        # Update experiment metadata with end time
        self.experiment_metadata['end_time'] = datetime.now().isoformat()
        
        with open(self.log_dir / 'experiment_metadata.json', 'w') as f:
            json.dump(self.experiment_metadata, f, indent=2)
        
        # Save final scalars
        self._save_scalars()
        
        # Create summary plots
        self.create_plots()
        
        # Create HTML summary
        self._create_html_summary()
        
        print(f"✅ Logger closed. Summary saved to: {self.log_dir}")
    
    def _create_html_summary(self):
        """Create an HTML summary of the experiment"""
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Experiment: {self.experiment_metadata['experiment_name']}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .metric {{ margin: 20px 0; }}
                .plot {{ margin: 20px 0; }}
                img {{ max-width: 100%; height: auto; }}
                pre {{ background: #f5f5f5; padding: 10px; }}
            </style>
        </head>
        <body>
            <h1>Experiment: {self.experiment_metadata['experiment_name']}</h1>
            <p><strong>Model:</strong> {self.model_name}</p>
            <p><strong>Started:</strong> {self.experiment_metadata['start_time']}</p>
            <p><strong>Ended:</strong> {self.experiment_metadata.get('end_time', 'In Progress')}</p>
            
            <h2>Training Plots</h2>
        """
        
        # Add plots
        plots_dir = self.log_dir / "plots"
        if plots_dir.exists():
            for plot_file in plots_dir.glob("*.png"):
                html_content += f'<div class="plot"><h3>{plot_file.stem.replace("_", " ").title()}</h3>'
                html_content += f'<img src="plots/{plot_file.name}" alt="{plot_file.stem}"></div>\n'
        
        # Add trajectory images
        if self.images_dir.exists():
            html_content += "<h2>Generated Trajectories</h2>\n"
            for img_file in sorted(self.images_dir.glob("*.png")):
                html_content += f'<div class="plot"><h3>{img_file.stem}</h3>'
                html_content += f'<img src="images/{img_file.name}" alt="{img_file.stem}"></div>\n'
        
        html_content += """
        </body>
        </html>
        """
        
        # Save HTML
        html_file = self.log_dir / 'summary.html'
        with open(html_file, 'w') as f:
            f.write(html_content)
        
        print(f"📄 HTML summary created: {html_file}")

# Quick usage example
if __name__ == "__main__":
    # Test the logger
    logger = SimpleTensorboardLogger(experiment_name="test_experiment")
    
    # Log some sample data
    for step in range(100):
        train_loss = 1.0 / (step + 1)
        val_loss = 1.2 / (step + 1)
        logger.log_training_metrics(step, train_loss, val_loss)
    
    # Create sample trajectory data
    sample_trajectory = [
        {'x': 100 + i*10, 'y': 200 + i*5, 't': i*16.67, 'p': 0 if i < 5 else 1}
        for i in range(10)
    ]
    
    logger.log_trajectory_visualization([sample_trajectory], ["test"], 0)
    
    # Test model parameters
    dummy_model = torch.nn.Linear(10, 5)
    logger.log_model_parameters(dummy_model, 0)
    
    logger.close()
    print("✅ Simple tensorboard test completed")