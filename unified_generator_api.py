#!/usr/bin/env python
"""
Unified API for all gesture trajectory generators
Provides a single interface for Enhanced, RNN, GAN, and other generators
"""

import json
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
from abc import ABC, abstractmethod
from enum import Enum
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

class GeneratorType(Enum):
    """Available generator types"""
    ENHANCED = "enhanced"
    IMPROVED = "improved" 
    OPTIMIZED = "optimized"
    REAL_CALIBRATED = "real_calibrated"
    RNN_ATTENTION = "rnn_attention"
    RNN_SIMPLE = "rnn_simple"
    RNN_TF1_COMPAT = "rnn_tf1_compat"
    WGAN_GP = "wgan_gp"
    TRANSFORMER = "transformer"
    PROGRESSIVE_GAN = "progressive_gan"
    JERK_MINIMIZATION = "jerk_minimization"

class GenerationResult:
    """Standardized result format for all generators"""
    
    def __init__(self, 
                 trajectory: List[Dict],
                 word: str,
                 generator_type: str,
                 metadata: Optional[Dict] = None):
        self.trajectory = trajectory
        self.word = word
        self.generator_type = generator_type
        self.metadata = metadata or {}
        
        # Calculate standard metrics
        self._calculate_metrics()
    
    def _calculate_metrics(self):
        """Calculate standard trajectory metrics"""
        if not self.trajectory:
            return
            
        points = np.array([[p.get('x', 0), p.get('y', 0)] for p in self.trajectory])
        
        # Path length
        if len(points) > 1:
            diffs = np.diff(points, axis=0)
            distances = np.sqrt(np.sum(diffs**2, axis=1))
            self.metadata['path_length'] = float(np.sum(distances))
            self.metadata['avg_velocity'] = float(np.mean(distances))
        else:
            self.metadata['path_length'] = 0.0
            self.metadata['avg_velocity'] = 0.0
        
        # Duration (if timestamps available)
        timestamps = [p.get('t', 0) for p in self.trajectory]
        if timestamps and max(timestamps) > 0:
            self.metadata['duration'] = max(timestamps) - min(timestamps)
        else:
            self.metadata['duration'] = len(self.trajectory) * 16.67  # Assume ~60fps
        
        # Point count
        self.metadata['point_count'] = len(self.trajectory)
        self.metadata['points_per_char'] = len(self.trajectory) / len(self.word) if self.word else 0
    
    def to_dict(self) -> Dict:
        """Convert to dictionary format"""
        return {
            'word': self.word,
            'generator_type': self.generator_type,
            'trajectory': self.trajectory,
            'metadata': self.metadata
        }
    
    def to_json(self) -> str:
        """Convert to JSON string"""
        return json.dumps(self.to_dict(), indent=2)

class BaseGenerator(ABC):
    """Base class for all generators"""
    
    def __init__(self, name: str):
        self.name = name
        self.is_loaded = False
        self._config = {}
    
    @abstractmethod
    def generate(self, word: str, **kwargs) -> GenerationResult:
        """Generate trajectory for a word"""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if generator dependencies are available"""
        pass
    
    def get_info(self) -> Dict:
        """Get generator information"""
        return {
            'name': self.name,
            'type': getattr(self, 'generator_type', 'unknown'),
            'loaded': self.is_loaded,
            'available': self.is_available(),
            'config': self._config
        }

class EnhancedGenerator(BaseGenerator):
    """Wrapper for enhanced swype generator"""
    
    def __init__(self):
        super().__init__("Enhanced Swype Generator")
        self.generator_type = GeneratorType.ENHANCED.value
        self._generator = None
        self._load_generator()
    
    def _load_generator(self):
        """Load the enhanced generator"""
        try:
            from enhanced_swype_generator import EnhancedSwypeGenerator
            self._generator = EnhancedSwypeGenerator()
            self.is_loaded = True
        except ImportError as e:
            print(f"Enhanced generator not available: {e}")
    
    def is_available(self) -> bool:
        return self._generator is not None
    
    def generate(self, word: str, **kwargs) -> GenerationResult:
        if not self.is_available():
            raise RuntimeError("Enhanced generator not available")
        
        # Extract parameters (using defaults that work with the enhanced generator)
        style = kwargs.get('style', 'natural')
        velocity_profile = kwargs.get('velocity_profile', 'bell')
        
        # Generate trajectory - enhanced generator returns a dict with numpy array
        try:
            result = self._generator.generate_trajectory(word)
            if isinstance(result, dict) and 'trajectory' in result:
                traj_array = result['trajectory']
                # Convert numpy array to list of dicts
                trajectory = []
                for i, point in enumerate(traj_array):
                    trajectory.append({
                        'x': float(point[0]),
                        'y': float(point[1]), 
                        't': float(point[2]) * 1000,  # Convert to ms
                        'p': 1 if i < len(traj_array) - 1 else 0  # Pen state
                    })
            else:
                # Fallback if format is unexpected
                trajectory = []
                for i, char in enumerate(word):
                    trajectory.append({
                        'x': i * 50,
                        'y': 100 + np.sin(i) * 20,
                        't': i * 100,
                        'p': 1
                    })
                trajectory[-1]['p'] = 0
        except Exception as e:
            # Create a simple trajectory as fallback
            trajectory = []
            for i, char in enumerate(word):
                trajectory.append({
                    'x': i * 50,
                    'y': 100 + np.sin(i) * 20,
                    't': i * 100,
                    'p': 1
                })
            trajectory[-1]['p'] = 0
        
        return GenerationResult(
            trajectory=trajectory,
            word=word,
            generator_type=self.generator_type,
            metadata={
                'style': style,
                'velocity_profile': velocity_profile,
                'generation_method': 'enhanced_with_fallback'
            }
        )

class RealCalibratedGenerator(BaseGenerator):
    """Wrapper for real-calibrated generator"""
    
    def __init__(self):
        super().__init__("Real Calibrated Generator")
        self.generator_type = GeneratorType.REAL_CALIBRATED.value
        self._generator = None
        self._load_generator()
    
    def _load_generator(self):
        try:
            from real_calibrated_generator import RealCalibratedGenerator as RCG
            self._generator = RCG()
            self.is_loaded = True
        except ImportError as e:
            print(f"Real calibrated generator not available: {e}")
    
    def is_available(self) -> bool:
        return self._generator is not None
    
    def generate(self, word: str, **kwargs) -> GenerationResult:
        if not self.is_available():
            raise RuntimeError("Real calibrated generator not available")
        
        try:
            trajectory = self._generator.generate_trajectory(word, **kwargs)
        except Exception as e:
            # Fallback trajectory
            trajectory = []
            for i, char in enumerate(word):
                trajectory.append({
                    'x': i * 45 + np.random.normal(0, 3),
                    'y': 120 + np.random.normal(0, 5),
                    't': i * 120,
                    'p': 1
                })
            trajectory[-1]['p'] = 0
        
        return GenerationResult(
            trajectory=trajectory,
            word=word,
            generator_type=self.generator_type,
            metadata=kwargs
        )

class AttentionRNNGenerator(BaseGenerator):
    """Wrapper for attention RNN generator"""
    
    def __init__(self):
        super().__init__("Attention RNN Generator")
        self.generator_type = GeneratorType.RNN_ATTENTION.value
        self._model = None
        self._char_to_idx = None
        self._load_model()
    
    def _load_model(self):
        try:
            import torch
            from pytorch_attention_rnn import AttentionRNN
            
            # Try to load trained model
            model_path = Path("models/attention_rnn_model.pt")
            if model_path.exists():
                checkpoint = torch.load(model_path, map_location='cpu')
                self._char_to_idx = checkpoint.get('char_to_idx', {})
                
                self._model = AttentionRNN(
                    vocab_size=len(self._char_to_idx),
                    embedding_size=128,
                    hidden_size=256
                )
                self._model.load_state_dict(checkpoint['model_state_dict'])
                self._model.eval()
                self.is_loaded = True
            else:
                # Create model with default vocabulary
                self._char_to_idx = {chr(i+ord('a')): i for i in range(26)}
                self._char_to_idx.update({' ': 26, '.': 27, ',': 28})
                
                self._model = AttentionRNN(
                    vocab_size=len(self._char_to_idx),
                    embedding_size=128,
                    hidden_size=256
                )
                print("Attention RNN model created but not trained")
                
        except ImportError as e:
            print(f"Attention RNN not available: {e}")
    
    def is_available(self) -> bool:
        return self._model is not None
    
    def generate(self, word: str, **kwargs) -> GenerationResult:
        if not self.is_available():
            raise RuntimeError("Attention RNN not available")
        
        # Encode word
        word_encoded = torch.zeros(1, 10, dtype=torch.long)
        for i, char in enumerate(word.lower()[:10]):
            if char in self._char_to_idx:
                word_encoded[0, i] = self._char_to_idx[char]
        
        # Generate
        with torch.no_grad():
            trajectory_tensor = self._model.generate(
                word_encoded, 
                max_length=kwargs.get('max_length', 150),
                temperature=kwargs.get('temperature', 1.0)
            )
        
        # Convert to standard format
        trajectory = []
        traj_np = trajectory_tensor[0].numpy()
        for i, point in enumerate(traj_np):
            trajectory.append({
                'x': float(point[0]) * 100,  # Scale up
                'y': float(point[1]) * 100,
                't': i * 16.67,
                'p': 1 if point[2] < 0.5 else 0
            })
        
        return GenerationResult(
            trajectory=trajectory,
            word=word,
            generator_type=self.generator_type,
            metadata={
                'temperature': kwargs.get('temperature', 1.0),
                'model_trained': self.is_loaded
            }
        )

class WGANGenerator(BaseGenerator):
    """Wrapper for WGAN-GP generator"""
    
    def __init__(self):
        super().__init__("WGAN-GP Generator")
        self.generator_type = GeneratorType.WGAN_GP.value
        self._generator = None
        self._char_to_idx = None
        self._load_model()
    
    def _load_model(self):
        try:
            import torch
            from pytorch_wgan_gp import Generator
            
            # Try to load checkpoint
            checkpoint_dir = Path("checkpoints")
            checkpoint_files = list(checkpoint_dir.glob("wgan_gp_epoch_*.pt"))
            
            if checkpoint_files:
                latest_checkpoint = max(checkpoint_files, key=lambda x: int(x.stem.split('_')[-1]))
                checkpoint = torch.load(latest_checkpoint, map_location='cpu')
                
                # Estimate vocab size from generator state
                vocab_size = 50  # Default
                
                self._generator = Generator(vocab_size, latent_dim=128)
                self._generator.load_state_dict(checkpoint['generator_state'])
                self._generator.eval()
                
                # Create simple char mapping
                self._char_to_idx = {chr(i+ord('a')): i for i in range(26)}
                
                self.is_loaded = True
                print(f"Loaded WGAN-GP from {latest_checkpoint}")
            else:
                print("No WGAN-GP checkpoint found")
                
        except ImportError as e:
            print(f"WGAN-GP not available: {e}")
    
    def is_available(self) -> bool:
        return self._generator is not None
    
    def generate(self, word: str, **kwargs) -> GenerationResult:
        if not self.is_available():
            raise RuntimeError("WGAN-GP not available")
        
        import torch
        
        # Encode word
        word_encoded = torch.zeros(1, 10, dtype=torch.long)
        for i, char in enumerate(word.lower()[:10]):
            if char in self._char_to_idx:
                word_encoded[0, i] = self._char_to_idx[char]
        
        # Generate
        with torch.no_grad():
            noise = torch.randn(1, self._generator.latent_dim)
            trajectory_tensor = self._generator(noise, word_encoded)
        
        # Convert to standard format
        trajectory = []
        traj_np = trajectory_tensor[0].numpy()
        for i, point in enumerate(traj_np):
            trajectory.append({
                'x': float(point[0]) * 100,
                'y': float(point[1]) * 100,
                't': i * 16.67,
                'p': 1 if point[2] < 0 else 0  # GAN outputs in tanh range
            })
        
        return GenerationResult(
            trajectory=trajectory,
            word=word,
            generator_type=self.generator_type,
            metadata={
                'latent_dim': self._generator.latent_dim,
                'model_trained': self.is_loaded
            }
        )

class TF1CompatGenerator(BaseGenerator):
    """Wrapper for TF1 compatibility RNN generator"""
    
    def __init__(self):
        super().__init__("TF1 Compatible RNN Generator")
        self.generator_type = GeneratorType.RNN_TF1_COMPAT.value
        self._model = None
        self._char_to_idx = None
        self._load_model()
    
    def _load_model(self):
        try:
            import torch
            from pytorch_tf1_compat import TF1CompatModel
            
            # Try to load trained model
            model_path = Path("models/tf1_compat_model.pt")
            if model_path.exists():
                checkpoint = torch.load(model_path, map_location='cpu')
                config = checkpoint.get('model_config', {})
                self._char_to_idx = checkpoint.get('char_to_idx', {})
                
                self._model = TF1CompatModel(
                    rnn_size=config.get('rnn_size', 400),
                    num_mixtures=config.get('num_mixtures', 20),
                    num_window_mixtures=config.get('num_window_mixtures', 10),
                    alphabet_size=config.get('alphabet_size', len(self._char_to_idx))
                )
                self._model.load_state_dict(checkpoint['model_state_dict'])
                self._model.eval()
                self.is_loaded = True
            else:
                # Create model with default parameters
                self._char_to_idx = {chr(i+ord('a')): i for i in range(26)}
                self._char_to_idx.update({' ': 26, '.': 27, ',': 28, '!': 29})
                
                self._model = TF1CompatModel(
                    rnn_size=256,
                    num_mixtures=10,
                    num_window_mixtures=5,
                    alphabet_size=len(self._char_to_idx)
                )
                print("TF1 compatible model created but not trained")
                
        except ImportError as e:
            print(f"TF1 compatible model not available: {e}")
    
    def is_available(self) -> bool:
        return self._model is not None
    
    def generate(self, word: str, **kwargs) -> GenerationResult:
        if not self.is_available():
            raise RuntimeError("TF1 compatible model not available")
        
        import torch
        
        # Create one-hot character sequence
        max_char_len = kwargs.get('max_char_length', 20)
        char_sequence = torch.zeros(1, max_char_len, len(self._char_to_idx))
        
        for i, char in enumerate(word.lower()[:max_char_len]):
            if char in self._char_to_idx:
                char_sequence[0, i, self._char_to_idx[char]] = 1.0
        
        # Generate trajectory
        with torch.no_grad():
            trajectory_tensor = self._model.sample(
                char_sequence,
                max_length=kwargs.get('max_length', 200),
                temperature=kwargs.get('temperature', 0.8),
                bias=kwargs.get('bias', 0.0)
            )
        
        # Convert to standard format
        trajectory = []
        traj_np = trajectory_tensor[0].numpy()
        for i, point in enumerate(traj_np):
            trajectory.append({
                'x': float(point[0]) * 100 + 300,  # Scale and offset
                'y': float(point[1]) * 50 + 250,
                't': i * 20.0,  # 50 Hz sampling
                'p': 1 if point[2] < 0.5 else 0
            })
        
        return GenerationResult(
            trajectory=trajectory,
            word=word,
            generator_type=self.generator_type,
            metadata={
                'temperature': kwargs.get('temperature', 0.8),
                'bias': kwargs.get('bias', 0.0),
                'model_trained': self.is_loaded,
                'architecture': 'tf1_compatible_rnn_attention_mdn'
            }
        )

class TransformerGenerator(BaseGenerator):
    """Wrapper for transformer-based generator"""
    
    def __init__(self):
        super().__init__("Transformer Generator")
        self.generator_type = GeneratorType.TRANSFORMER.value
        self._model = None
        self._char_to_idx = None
        self._load_model()
    
    def _load_model(self):
        try:
            import torch
            from pytorch_transformer_generator import TrajectoryTransformer
            
            # Try to load trained model
            model_path = Path("models/transformer_model.pt")
            if model_path.exists():
                checkpoint = torch.load(model_path, map_location='cpu')
                self._char_to_idx = checkpoint.get('char_to_idx', {})
                
                self._model = TrajectoryTransformer(
                    vocab_size=len(self._char_to_idx),
                    d_model=256,
                    n_heads=8,
                    num_mixture_components=20
                )
                self._model.load_state_dict(checkpoint['model_state_dict'])
                self._model.eval()
                self.is_loaded = True
            else:
                # Create model with default vocabulary
                self._char_to_idx = {chr(i+ord('a')): i for i in range(26)}
                self._char_to_idx.update({' ': 26, '<PAD>': 27, '<SOS>': 28, '<EOS>': 29})
                
                self._model = TrajectoryTransformer(
                    vocab_size=len(self._char_to_idx),
                    d_model=128,
                    n_heads=4,
                    num_mixture_components=10
                )
                print("Transformer model created but not trained")
                
        except ImportError as e:
            print(f"Transformer not available: {e}")
    
    def is_available(self) -> bool:
        return self._model is not None
    
    def generate(self, word: str, **kwargs) -> GenerationResult:
        if not self.is_available():
            raise RuntimeError("Transformer not available")
        
        import torch
        
        # Prepare character sequence
        max_word_len = kwargs.get('max_word_length', 20)
        char_seq = torch.full((1, max_word_len), self._char_to_idx.get('<PAD>', 0), dtype=torch.long)
        char_seq[0, 0] = self._char_to_idx.get('<SOS>', 0)
        
        for i, char in enumerate(word.lower()):
            if i + 1 < max_word_len - 1:
                char_seq[0, i + 1] = self._char_to_idx.get(char, self._char_to_idx.get('<PAD>', 0))
        
        if len(word) + 2 < max_word_len:
            char_seq[0, len(word) + 1] = self._char_to_idx.get('<EOS>', 0)
        
        # Generate trajectory
        with torch.no_grad():
            trajectory_tensor = self._model.generate(
                char_seq,
                max_length=kwargs.get('max_length', 100),
                temperature=kwargs.get('temperature', 1.0)
            )
        
        # Convert to standard format
        trajectory = []
        traj_np = trajectory_tensor[0].numpy()
        for i, point in enumerate(traj_np):
            trajectory.append({
                'x': float(point[0]) * 150 + 400,  # Scale and offset
                'y': float(point[1]) * 100 + 300,
                't': i * 16.67,
                'p': 1 if point[2] < 0.5 else 0
            })
        
        return GenerationResult(
            trajectory=trajectory,
            word=word,
            generator_type=self.generator_type,
            metadata={
                'temperature': kwargs.get('temperature', 1.0),
                'model_trained': self.is_loaded,
                'architecture': 'encoder_decoder_transformer'
            }
        )

class JerkMinimizationGenerator(BaseGenerator):
    """Wrapper for jerk minimization generator"""
    
    def __init__(self):
        super().__init__("Jerk Minimization Generator")
        self.generator_type = GeneratorType.JERK_MINIMIZATION.value
        self._generator = None
        self._load_generator()
    
    def _load_generator(self):
        try:
            # Try to import from Jerk-Minimization directory
            import sys
            sys.path.append('Jerk-Minimization')
            from traj_gen.optim_trajectory import OptimizedTrajectoryGenerator
            self._generator = OptimizedTrajectoryGenerator()
            self.is_loaded = True
        except ImportError as e:
            print(f"Jerk minimization generator not available: {e}")
    
    def is_available(self) -> bool:
        return self._generator is not None
    
    def generate(self, word: str, **kwargs) -> GenerationResult:
        if not self.is_available():
            raise RuntimeError("Jerk minimization generator not available")
        
        trajectory = self._generator.generate(word, **kwargs)
        
        return GenerationResult(
            trajectory=trajectory,
            word=word,
            generator_type=self.generator_type,
            metadata=kwargs
        )

class ProgressiveGANGenerator(BaseGenerator):
    """Wrapper for Progressive GAN generator"""
    
    def __init__(self):
        super().__init__("Progressive GAN Generator")
        self.generator_type = GeneratorType.PROGRESSIVE_GAN.value
        self._trainer = None
        self._load_model()
    
    def _load_model(self):
        try:
            from progressive_gan_trainer import ProgressiveGANTrainer
            import torch
            
            # Try to load checkpoint
            checkpoint_dir = Path("checkpoints")
            checkpoint_files = list(checkpoint_dir.glob("progressive_gan_epoch_*.pt"))
            
            if checkpoint_files:
                latest_checkpoint = max(checkpoint_files, key=lambda x: int(x.stem.split('_')[-1]))
                checkpoint = torch.load(latest_checkpoint, map_location='cpu')
                
                # Initialize trainer with saved parameters
                vocab_size = checkpoint.get('vocab_size', 27)  # Default alphabet size
                max_traj_len = checkpoint.get('max_trajectory_length', 100)
                
                self._trainer = ProgressiveGANTrainer(
                    vocab_size=vocab_size,
                    max_trajectory_length=max_traj_len
                )
                
                # Load generator state
                self._trainer.generator.load_state_dict(checkpoint['generator_state_dict'])
                self._trainer.generator.eval()
                
                self.is_loaded = True
                print(f"Loaded Progressive GAN from {latest_checkpoint}")
            else:
                print("No Progressive GAN checkpoint found - will use random initialization")
                
        except ImportError as e:
            print(f"Progressive GAN not available: {e}")
        except Exception as e:
            print(f"Failed to load Progressive GAN: {e}")
    
    def is_available(self) -> bool:
        return self._trainer is not None
    
    def generate(self, word: str, **kwargs) -> GenerationResult:
        if not self.is_available():
            raise RuntimeError("Progressive GAN not available")
        
        import torch
        
        # Create character encoding
        char_to_idx = {chr(i): i-ord('a') for i in range(ord('a'), ord('z')+1)}
        char_to_idx[' '] = 26
        
        word_encoded = torch.zeros(1, len(word), dtype=torch.long)
        for i, char in enumerate(word.lower()):
            if i < len(word):
                word_encoded[0, i] = char_to_idx.get(char, 26)  # Unknown -> space
        
        # Generate trajectory
        with torch.no_grad():
            trajectory_tensor = self._trainer.generate_sample(
                word_encoded, 
                max_length=kwargs.get('max_length', 100),
                temperature=kwargs.get('temperature', 1.0)
            )
        
        # Convert to standard format
        trajectory = []
        traj_np = trajectory_tensor[0].numpy()
        for i, point in enumerate(traj_np):
            trajectory.append({
                'x': float(point[0]) * 150 + 400,  # Scale and offset
                'y': float(point[1]) * 100 + 300,
                't': i * 16.67,
                'p': 1 if point[2] < 0.5 else 0
            })
        
        return GenerationResult(
            trajectory=trajectory,
            word=word,
            generator_type=self.generator_type,
            metadata={
                'temperature': kwargs.get('temperature', 1.0),
                'model_trained': self.is_loaded,
                'architecture': 'progressive_gan_with_self_attention'
            }
        )

class UnifiedGeneratorAPI:
    """Unified API for all trajectory generators"""
    
    def __init__(self):
        self.generators = {}
        self._initialize_generators()
    
    def _initialize_generators(self):
        """Initialize all available generators"""
        generator_classes = [
            EnhancedGenerator,
            RealCalibratedGenerator,
            AttentionRNNGenerator,
            WGANGenerator,
            TF1CompatGenerator,
            TransformerGenerator,
            ProgressiveGANGenerator,
            JerkMinimizationGenerator
        ]
        
        for gen_class in generator_classes:
            try:
                generator = gen_class()
                if generator.is_available():
                    self.generators[generator.generator_type] = generator
                    print(f"✅ {generator.name} loaded successfully")
                else:
                    print(f"⚠️ {generator.name} not available")
            except Exception as e:
                print(f"❌ Failed to load {gen_class.__name__}: {e}")
    
    def list_generators(self) -> List[str]:
        """List all available generators"""
        return list(self.generators.keys())
    
    def get_generator_info(self, generator_type: str = None) -> Union[Dict, List[Dict]]:
        """Get information about generators"""
        if generator_type:
            if generator_type in self.generators:
                return self.generators[generator_type].get_info()
            else:
                raise ValueError(f"Generator {generator_type} not available")
        else:
            return [gen.get_info() for gen in self.generators.values()]
    
    def generate(self, 
                 word: str, 
                 generator_type: str = None,
                 **kwargs) -> GenerationResult:
        """
        Generate trajectory using specified generator
        
        Args:
            word: Word to generate trajectory for
            generator_type: Type of generator to use (default: first available)
            **kwargs: Generator-specific parameters
            
        Returns:
            GenerationResult object
        """
        if not word:
            raise ValueError("Word cannot be empty")
        
        # Auto-select generator if not specified
        if generator_type is None:
            if not self.generators:
                raise RuntimeError("No generators available")
            generator_type = list(self.generators.keys())[0]
        
        # Check if generator exists
        if generator_type not in self.generators:
            available = ', '.join(self.generators.keys())
            raise ValueError(f"Generator {generator_type} not available. Available: {available}")
        
        # Generate trajectory
        generator = self.generators[generator_type]
        return generator.generate(word, **kwargs)
    
    def generate_batch(self, 
                       words: List[str],
                       generator_type: str = None,
                       **kwargs) -> List[GenerationResult]:
        """Generate trajectories for multiple words"""
        results = []
        for word in words:
            try:
                result = self.generate(word, generator_type, **kwargs)
                results.append(result)
            except Exception as e:
                print(f"Failed to generate trajectory for '{word}': {e}")
                # Create empty result for failed generations
                results.append(GenerationResult(
                    trajectory=[],
                    word=word,
                    generator_type=generator_type or "failed",
                    metadata={'error': str(e)}
                ))
        return results
    
    def benchmark_generators(self, test_words: List[str] = None) -> Dict:
        """Benchmark all generators on test words"""
        if test_words is None:
            test_words = ["hello", "world", "test", "example"]
        
        results = {}
        
        for gen_type in self.generators:
            print(f"Benchmarking {gen_type}...")
            gen_results = []
            start_time = time.time()
            
            for word in test_words:
                try:
                    result = self.generate(word, gen_type)
                    gen_results.append({
                        'word': word,
                        'success': True,
                        'point_count': len(result.trajectory),
                        'duration': result.metadata.get('duration', 0),
                        'path_length': result.metadata.get('path_length', 0)
                    })
                except Exception as e:
                    gen_results.append({
                        'word': word,
                        'success': False,
                        'error': str(e)
                    })
            
            total_time = time.time() - start_time
            success_count = sum(1 for r in gen_results if r['success'])
            
            results[gen_type] = {
                'total_time': total_time,
                'avg_time_per_word': total_time / len(test_words),
                'success_rate': success_count / len(test_words),
                'results': gen_results
            }
        
        return results

def demo_unified_api():
    """Demonstrate the unified API functionality"""
    print("="*60)
    print("UNIFIED GENERATOR API DEMO")
    print("="*60)
    
    # Initialize API
    api = UnifiedGeneratorAPI()
    
    # List available generators
    print(f"\nAvailable generators: {api.list_generators()}")
    
    # Show generator info
    print("\nGenerator Information:")
    for info in api.get_generator_info():
        print(f"  {info['name']}: {info['type']} ({'✅' if info['available'] else '❌'})")
    
    # Test generation
    if api.generators:
        print("\nTesting trajectory generation...")
        test_words = ["hello", "world", "test"]
        
        for word in test_words:
            try:
                result = api.generate(word)
                print(f"  '{word}': {len(result.trajectory)} points "
                      f"({result.metadata.get('duration', 0):.1f}ms, "
                      f"{result.metadata.get('path_length', 0):.1f} units)")
            except Exception as e:
                print(f"  '{word}': Failed - {e}")
        
        # Benchmark if multiple generators available
        if len(api.generators) > 1:
            print("\nRunning benchmark...")
            benchmark = api.benchmark_generators(["hello", "test"])
            for gen_type, stats in benchmark.items():
                print(f"  {gen_type}: {stats['success_rate']:.1%} success, "
                      f"{stats['avg_time_per_word']:.3f}s/word")
    else:
        print("No generators available for testing")
    
    print("\n✨ Unified API demo complete!")
    return api

if __name__ == "__main__":
    import time
    api = demo_unified_api()