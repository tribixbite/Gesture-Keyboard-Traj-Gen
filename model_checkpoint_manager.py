#!/usr/bin/env python
"""
Unified model checkpoint management system
Handles saving, loading, and managing checkpoints for all model types
"""

import os
import json
import torch
import pickle
import shutil
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelCheckpoint:
    """Represents a single model checkpoint"""
    
    def __init__(self, 
                 model_type: str,
                 model_name: str, 
                 epoch: int,
                 checkpoint_path: Path,
                 metadata: Dict = None):
        
        self.model_type = model_type
        self.model_name = model_name
        self.epoch = epoch
        self.checkpoint_path = Path(checkpoint_path)
        self.metadata = metadata or {}
        self.created_at = datetime.now()
        
        # Extract metrics from metadata if available
        self.metrics = self.metadata.get('metrics', {})
        self.config = self.metadata.get('config', {})
        
    def get_info(self) -> Dict:
        """Get checkpoint information"""
        return {
            'model_type': self.model_type,
            'model_name': self.model_name,
            'epoch': self.epoch,
            'checkpoint_path': str(self.checkpoint_path),
            'created_at': self.created_at.isoformat(),
            'file_size_mb': self.get_file_size_mb(),
            'metrics': self.metrics,
            'config': self.config
        }
    
    def get_file_size_mb(self) -> float:
        """Get checkpoint file size in MB"""
        if self.checkpoint_path.exists():
            return self.checkpoint_path.stat().st_size / (1024 * 1024)
        return 0.0
    
    def is_valid(self) -> bool:
        """Check if checkpoint file exists and is readable"""
        return self.checkpoint_path.exists() and self.checkpoint_path.stat().st_size > 0

class CheckpointManager:
    """Unified checkpoint management for all model types"""
    
    def __init__(self, base_dir: str = "checkpoints"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True)
        
        # Checkpoint metadata
        self.metadata_file = self.base_dir / "checkpoint_metadata.json"
        self.checkpoints = {}  # model_key -> List[ModelCheckpoint]
        
        self._load_metadata()
        self._scan_existing_checkpoints()
    
    def _load_metadata(self):
        """Load checkpoint metadata from disk"""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r') as f:
                    metadata = json.load(f)
                    # Reconstruct checkpoints from metadata
                    for model_key, checkpoint_list in metadata.items():
                        self.checkpoints[model_key] = []
                        for cp_data in checkpoint_list:
                            checkpoint = ModelCheckpoint(
                                model_type=cp_data['model_type'],
                                model_name=cp_data['model_name'],
                                epoch=cp_data['epoch'],
                                checkpoint_path=cp_data['checkpoint_path'],
                                metadata=cp_data.get('metadata', {})
                            )
                            if 'created_at' in cp_data:
                                checkpoint.created_at = datetime.fromisoformat(cp_data['created_at'])
                            self.checkpoints[model_key].append(checkpoint)
                            
                logger.info(f"Loaded metadata for {len(self.checkpoints)} models")
            except Exception as e:
                logger.warning(f"Could not load metadata: {e}")
                self.checkpoints = {}
    
    def _save_metadata(self):
        """Save checkpoint metadata to disk"""
        metadata = {}
        for model_key, checkpoint_list in self.checkpoints.items():
            metadata[model_key] = [cp.get_info() for cp in checkpoint_list]
        
        try:
            with open(self.metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
        except Exception as e:
            logger.error(f"Could not save metadata: {e}")
    
    def _scan_existing_checkpoints(self):
        """Scan for existing checkpoint files not in metadata"""
        
        # Common checkpoint patterns
        patterns = [
            "*.pt",     # PyTorch
            "*.pth",    # PyTorch alternative
            "*.pkl",    # Pickle
            "*.ckpt",   # General checkpoint
            "*.h5",     # TensorFlow/Keras
            "*.json"    # Configuration files
        ]
        
        existing_files = set()
        for pattern in patterns:
            existing_files.update(self.base_dir.glob(pattern))
        
        # Check for files not in our metadata
        tracked_files = set()
        for checkpoint_list in self.checkpoints.values():
            for cp in checkpoint_list:
                tracked_files.add(cp.checkpoint_path)
        
        untracked = existing_files - tracked_files
        if untracked:
            logger.info(f"Found {len(untracked)} untracked checkpoint files")
            
            # Try to identify and register them
            for file_path in untracked:
                self._try_register_checkpoint(file_path)
    
    def _try_register_checkpoint(self, file_path: Path):
        """Try to register an untracked checkpoint file"""
        
        # Extract info from filename
        filename = file_path.stem
        
        # Common patterns
        if "wgan_gp_epoch_" in filename:
            epoch = int(filename.split('_')[-1])
            self._register_checkpoint(
                model_type="wgan_gp",
                model_name="trajectory_wgan",
                epoch=epoch,
                checkpoint_path=file_path,
                metadata={'discovered': True}
            )
        elif "attention_rnn" in filename:
            self._register_checkpoint(
                model_type="rnn_attention", 
                model_name="attention_rnn",
                epoch=0,  # Unknown epoch
                checkpoint_path=file_path,
                metadata={'discovered': True}
            )
        elif "pytorch_rnn" in filename:
            self._register_checkpoint(
                model_type="rnn_simple",
                model_name="pytorch_rnn", 
                epoch=0,
                checkpoint_path=file_path,
                metadata={'discovered': True}
            )
        else:
            logger.debug(f"Could not identify checkpoint: {filename}")
    
    def _register_checkpoint(self, model_type: str, model_name: str, 
                           epoch: int, checkpoint_path: Path, metadata: Dict = None):
        """Register a checkpoint internally"""
        
        model_key = f"{model_type}_{model_name}"
        checkpoint = ModelCheckpoint(
            model_type=model_type,
            model_name=model_name,
            epoch=epoch,
            checkpoint_path=checkpoint_path,
            metadata=metadata or {}
        )
        
        if model_key not in self.checkpoints:
            self.checkpoints[model_key] = []
        
        self.checkpoints[model_key].append(checkpoint)
        self.checkpoints[model_key].sort(key=lambda x: x.epoch)
    
    def save_checkpoint(self, 
                       model: torch.nn.Module,
                       model_type: str,
                       model_name: str,
                       epoch: int,
                       optimizer: Optional[torch.optim.Optimizer] = None,
                       scheduler: Optional[Any] = None,
                       metrics: Optional[Dict] = None,
                       config: Optional[Dict] = None,
                       extra_data: Optional[Dict] = None) -> ModelCheckpoint:
        """
        Save a model checkpoint
        
        Args:
            model: PyTorch model to save
            model_type: Type of model (e.g., 'rnn_attention', 'wgan_gp')
            model_name: Name/identifier for the model
            epoch: Training epoch number
            optimizer: Optimizer state to save
            scheduler: Learning rate scheduler state
            metrics: Training metrics (loss, accuracy, etc.)
            config: Model configuration
            extra_data: Additional data to save
            
        Returns:
            ModelCheckpoint object
        """
        
        # Create checkpoint filename
        timestamp = int(time.time())
        filename = f"{model_type}_{model_name}_epoch_{epoch}_{timestamp}.pt"
        checkpoint_path = self.base_dir / filename
        
        # Prepare checkpoint data
        checkpoint_data = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'model_type': model_type,
            'model_name': model_name,
            'timestamp': timestamp,
            'pytorch_version': torch.__version__
        }
        
        # Add optimizer state
        if optimizer is not None:
            checkpoint_data['optimizer_state_dict'] = optimizer.state_dict()
            checkpoint_data['optimizer_class'] = optimizer.__class__.__name__
        
        # Add scheduler state
        if scheduler is not None:
            checkpoint_data['scheduler_state_dict'] = scheduler.state_dict()
            checkpoint_data['scheduler_class'] = scheduler.__class__.__name__
        
        # Add metrics and config
        if metrics:
            checkpoint_data['metrics'] = metrics
        if config:
            checkpoint_data['config'] = config
        if extra_data:
            checkpoint_data.update(extra_data)
        
        # Save checkpoint
        try:
            torch.save(checkpoint_data, checkpoint_path)
            logger.info(f"Saved checkpoint: {checkpoint_path}")
            
            # Register checkpoint
            metadata = {
                'metrics': metrics or {},
                'config': config or {},
                'file_size_mb': checkpoint_path.stat().st_size / (1024 * 1024)
            }
            
            checkpoint = ModelCheckpoint(
                model_type=model_type,
                model_name=model_name,
                epoch=epoch,
                checkpoint_path=checkpoint_path,
                metadata=metadata
            )
            
            # Add to tracking
            model_key = f"{model_type}_{model_name}"
            if model_key not in self.checkpoints:
                self.checkpoints[model_key] = []
            
            self.checkpoints[model_key].append(checkpoint)
            self.checkpoints[model_key].sort(key=lambda x: x.epoch)
            
            # Save metadata
            self._save_metadata()
            
            # Clean up old checkpoints if needed
            self._cleanup_old_checkpoints(model_key)
            
            return checkpoint
            
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
            raise
    
    def load_checkpoint(self, 
                       model: torch.nn.Module,
                       model_type: str,
                       model_name: str,
                       epoch: Optional[int] = None,
                       optimizer: Optional[torch.optim.Optimizer] = None,
                       scheduler: Optional[Any] = None,
                       device: str = 'cpu') -> Optional[Dict]:
        """
        Load a model checkpoint
        
        Args:
            model: PyTorch model to load into
            model_type: Type of model
            model_name: Model name/identifier
            epoch: Specific epoch to load (None for latest)
            optimizer: Optimizer to load state into
            scheduler: Scheduler to load state into
            device: Device to load model on
            
        Returns:
            Checkpoint data dictionary or None if not found
        """
        
        checkpoint = self.get_checkpoint(model_type, model_name, epoch)
        if checkpoint is None:
            logger.warning(f"No checkpoint found for {model_type}_{model_name}")
            return None
        
        if not checkpoint.is_valid():
            logger.error(f"Checkpoint file is invalid: {checkpoint.checkpoint_path}")
            return None
        
        try:
            # Load checkpoint data
            checkpoint_data = torch.load(checkpoint.checkpoint_path, map_location=device)
            
            # Load model state
            model.load_state_dict(checkpoint_data['model_state_dict'])
            logger.info(f"Loaded model state from epoch {checkpoint.epoch}")
            
            # Load optimizer state
            if optimizer is not None and 'optimizer_state_dict' in checkpoint_data:
                optimizer.load_state_dict(checkpoint_data['optimizer_state_dict'])
                logger.info("Loaded optimizer state")
            
            # Load scheduler state
            if scheduler is not None and 'scheduler_state_dict' in checkpoint_data:
                scheduler.load_state_dict(checkpoint_data['scheduler_state_dict'])
                logger.info("Loaded scheduler state")
            
            return checkpoint_data
            
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            return None
    
    def get_checkpoint(self, 
                      model_type: str, 
                      model_name: str, 
                      epoch: Optional[int] = None) -> Optional[ModelCheckpoint]:
        """Get a specific checkpoint"""
        
        model_key = f"{model_type}_{model_name}"
        if model_key not in self.checkpoints:
            return None
        
        checkpoints = self.checkpoints[model_key]
        if not checkpoints:
            return None
        
        if epoch is None:
            # Return latest checkpoint
            return max(checkpoints, key=lambda x: x.epoch)
        else:
            # Find specific epoch
            for cp in checkpoints:
                if cp.epoch == epoch:
                    return cp
            return None
    
    def list_checkpoints(self, 
                        model_type: Optional[str] = None,
                        model_name: Optional[str] = None) -> List[ModelCheckpoint]:
        """List available checkpoints"""
        
        all_checkpoints = []
        for model_key, checkpoint_list in self.checkpoints.items():
            for cp in checkpoint_list:
                # Filter by type and name if specified
                if model_type and cp.model_type != model_type:
                    continue
                if model_name and cp.model_name != model_name:
                    continue
                all_checkpoints.append(cp)
        
        # Sort by creation time (newest first)
        all_checkpoints.sort(key=lambda x: x.created_at, reverse=True)
        return all_checkpoints
    
    def get_best_checkpoint(self, 
                           model_type: str,
                           model_name: str,
                           metric: str = 'val_loss',
                           mode: str = 'min') -> Optional[ModelCheckpoint]:
        """
        Get the best checkpoint based on a metric
        
        Args:
            model_type: Type of model
            model_name: Model name
            metric: Metric to optimize ('val_loss', 'accuracy', etc.)
            mode: 'min' for minimize, 'max' for maximize
        """
        
        model_key = f"{model_type}_{model_name}"
        if model_key not in self.checkpoints:
            return None
        
        checkpoints = self.checkpoints[model_key]
        valid_checkpoints = [cp for cp in checkpoints if metric in cp.metrics]
        
        if not valid_checkpoints:
            logger.warning(f"No checkpoints found with metric '{metric}'")
            return None
        
        if mode == 'min':
            return min(valid_checkpoints, key=lambda x: x.metrics[metric])
        else:
            return max(valid_checkpoints, key=lambda x: x.metrics[metric])
    
    def delete_checkpoint(self, checkpoint: ModelCheckpoint) -> bool:
        """Delete a checkpoint file and remove from tracking"""
        
        try:
            # Delete file
            if checkpoint.checkpoint_path.exists():
                checkpoint.checkpoint_path.unlink()
                logger.info(f"Deleted checkpoint file: {checkpoint.checkpoint_path}")
            
            # Remove from tracking
            model_key = f"{checkpoint.model_type}_{checkpoint.model_name}"
            if model_key in self.checkpoints:
                self.checkpoints[model_key] = [
                    cp for cp in self.checkpoints[model_key] 
                    if cp.checkpoint_path != checkpoint.checkpoint_path
                ]
            
            self._save_metadata()
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete checkpoint: {e}")
            return False
    
    def _cleanup_old_checkpoints(self, model_key: str, keep_last: int = 5):
        """Clean up old checkpoints, keeping only the most recent"""
        
        if model_key not in self.checkpoints:
            return
        
        checkpoints = self.checkpoints[model_key]
        if len(checkpoints) <= keep_last:
            return
        
        # Sort by epoch (newest first)
        checkpoints.sort(key=lambda x: x.epoch, reverse=True)
        
        # Delete old checkpoints
        to_delete = checkpoints[keep_last:]
        for cp in to_delete:
            logger.info(f"Cleaning up old checkpoint: epoch {cp.epoch}")
            self.delete_checkpoint(cp)
    
    def export_model(self, 
                    model_type: str,
                    model_name: str,
                    export_path: str,
                    epoch: Optional[int] = None,
                    format: str = 'pytorch') -> bool:
        """
        Export a model for deployment
        
        Args:
            model_type: Type of model
            model_name: Model name
            export_path: Path to export to
            epoch: Specific epoch (None for latest)
            format: Export format ('pytorch', 'torchscript', 'onnx')
        """
        
        checkpoint = self.get_checkpoint(model_type, model_name, epoch)
        if checkpoint is None:
            logger.error(f"No checkpoint found for export")
            return False
        
        export_path = Path(export_path)
        export_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            if format == 'pytorch':
                # Simple copy
                shutil.copy2(checkpoint.checkpoint_path, export_path)
                logger.info(f"Exported PyTorch model to {export_path}")
                return True
            else:
                logger.warning(f"Export format '{format}' not yet implemented")
                return False
                
        except Exception as e:
            logger.error(f"Failed to export model: {e}")
            return False
    
    def get_summary(self) -> Dict:
        """Get summary of all checkpoints"""
        
        total_checkpoints = sum(len(cps) for cps in self.checkpoints.values())
        total_size_mb = sum(
            cp.get_file_size_mb() 
            for checkpoint_list in self.checkpoints.values()
            for cp in checkpoint_list
        )
        
        model_types = set()
        for checkpoint_list in self.checkpoints.values():
            for cp in checkpoint_list:
                model_types.add(cp.model_type)
        
        return {
            'total_models': len(self.checkpoints),
            'total_checkpoints': total_checkpoints,
            'total_size_mb': round(total_size_mb, 2),
            'model_types': list(model_types),
            'base_directory': str(self.base_dir)
        }

def demo_checkpoint_manager():
    """Demonstrate checkpoint manager functionality"""
    
    print("="*60)
    print("CHECKPOINT MANAGER DEMO")
    print("="*60)
    
    # Create manager
    manager = CheckpointManager()
    
    # Show summary
    summary = manager.get_summary()
    print(f"\nCheckpoint Summary:")
    print(f"  Total models: {summary['total_models']}")
    print(f"  Total checkpoints: {summary['total_checkpoints']}")
    print(f"  Total size: {summary['total_size_mb']} MB")
    print(f"  Model types: {', '.join(summary['model_types'])}")
    
    # List all checkpoints
    print(f"\nAvailable Checkpoints:")
    checkpoints = manager.list_checkpoints()
    if checkpoints:
        for cp in checkpoints[:10]:  # Show first 10
            print(f"  {cp.model_type}/{cp.model_name} epoch {cp.epoch} "
                  f"({cp.get_file_size_mb():.1f} MB)")
        if len(checkpoints) > 10:
            print(f"  ... and {len(checkpoints) - 10} more")
    else:
        print("  No checkpoints found")
    
    # Test with a dummy model if none exist
    if not checkpoints:
        print(f"\nTesting checkpoint save with dummy model...")
        
        # Create a simple model
        import torch.nn as nn
        model = nn.Sequential(
            nn.Linear(10, 50),
            nn.ReLU(),
            nn.Linear(50, 1)
        )
        
        optimizer = torch.optim.Adam(model.parameters())
        
        # Save checkpoint
        checkpoint = manager.save_checkpoint(
            model=model,
            model_type="demo",
            model_name="test_model",
            epoch=1,
            optimizer=optimizer,
            metrics={'loss': 0.5, 'accuracy': 0.8},
            config={'hidden_size': 50}
        )
        
        print(f"  Saved demo checkpoint: {checkpoint.checkpoint_path.name}")
        
        # Test loading
        new_model = nn.Sequential(
            nn.Linear(10, 50),
            nn.ReLU(), 
            nn.Linear(50, 1)
        )
        
        data = manager.load_checkpoint(
            model=new_model,
            model_type="demo",
            model_name="test_model",
            epoch=1
        )
        
        if data:
            print(f"  Successfully loaded checkpoint from epoch {data['epoch']}")
        
        # Clean up demo checkpoint
        manager.delete_checkpoint(checkpoint)
        print(f"  Cleaned up demo checkpoint")
    
    print(f"\n✨ Checkpoint manager demo complete!")
    
    return manager

if __name__ == "__main__":
    manager = demo_checkpoint_manager()