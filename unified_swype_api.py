#!/usr/bin/env python
"""
Unified API for all swype trajectory generators
Provides consistent interface across different generation methods
"""

import numpy as np
import pickle
from typing import Dict, List, Optional, Union
from enum import Enum
from dataclasses import dataclass

# Import all generators
from enhanced_swype_generator import EnhancedSwypeGenerator
from improved_swype_generator import ImprovedSwypeGenerator  
from optimized_swype_generator import OptimizedSwypeGenerator, StyleParameters

class GeneratorType(Enum):
    """Available generator types"""
    ENHANCED = "enhanced"
    IMPROVED = "improved"
    OPTIMIZED = "optimized"
    RNN = "rnn"
    JERK_MIN = "jerk_minimization"

@dataclass
class TrajectoryConfig:
    """Configuration for trajectory generation"""
    word: str
    generator_type: GeneratorType = GeneratorType.OPTIMIZED
    style: Union[str, StyleParameters] = "natural"
    sampling_rate: int = 100
    duration: Optional[float] = None
    user_speed: float = 1.0
    precision: float = 0.8
    adaptive_sampling: bool = False
    include_pressure: bool = True

class UnifiedSwypeAPI:
    """
    Unified interface for all trajectory generation methods
    """
    
    def __init__(self, keyboard_layout_path: str = 'holokeyboard.csv'):
        """
        Initialize all available generators
        
        Args:
            keyboard_layout_path: Path to keyboard layout CSV
        """
        self.keyboard_layout_path = keyboard_layout_path
        self.generators = {}
        
        # Initialize generators
        self._init_generators()
        
        # Load RNN model if available
        self.rnn_model = None
        self._load_rnn_model()
    
    def _init_generators(self):
        """Initialize all generator instances"""
        try:
            self.generators[GeneratorType.ENHANCED] = EnhancedSwypeGenerator(
                self.keyboard_layout_path
            )
        except Exception as e:
            print(f"Warning: Could not initialize Enhanced generator: {e}")
        
        try:
            self.generators[GeneratorType.IMPROVED] = ImprovedSwypeGenerator(
                self.keyboard_layout_path
            )
        except Exception as e:
            print(f"Warning: Could not initialize Improved generator: {e}")
        
        try:
            self.generators[GeneratorType.OPTIMIZED] = OptimizedSwypeGenerator(
                self.keyboard_layout_path
            )
        except Exception as e:
            print(f"Warning: Could not initialize Optimized generator: {e}")
    
    def _load_rnn_model(self):
        """Load pre-trained RNN model if available"""
        try:
            with open('models/rnn_gesture_model.pkl', 'rb') as f:
                model_data = pickle.load(f)
            
            # Reconstruct model
            from train_rnn_model import SimpleGestureRNN
            
            self.rnn_model = SimpleGestureRNN(
                model_data['config']['input_dim'],
                model_data['config']['hidden_dim'],
                model_data['config']['output_dim']
            )
            
            # Load weights
            self.rnn_model.Wxh = model_data['Wxh']
            self.rnn_model.Whh = model_data['Whh']
            self.rnn_model.Why = model_data['Why']
            self.rnn_model.bh = model_data['bh']
            self.rnn_model.by = model_data['by']
            
            print("RNN model loaded successfully")
        except Exception as e:
            print(f"Warning: Could not load RNN model: {e}")
    
    def generate(self, config: TrajectoryConfig) -> Dict:
        """
        Generate trajectory using specified configuration
        
        Args:
            config: Trajectory generation configuration
            
        Returns:
            Dictionary containing trajectory data
        """
        if config.generator_type == GeneratorType.RNN:
            return self._generate_rnn(config)
        elif config.generator_type == GeneratorType.JERK_MIN:
            return self._generate_jerk_min(config)
        else:
            return self._generate_standard(config)
    
    def _generate_standard(self, config: TrajectoryConfig) -> Dict:
        """Generate using standard generators"""
        generator = self.generators.get(config.generator_type)
        
        if not generator:
            raise ValueError(f"Generator {config.generator_type} not available")
        
        # Map parameters based on generator type
        if config.generator_type == GeneratorType.OPTIMIZED:
            # Use full feature set
            return generator.generate_trajectory(
                word=config.word,
                sampling_rate=config.sampling_rate,
                style=config.style,
                adaptive_sampling=config.adaptive_sampling
            )
        elif config.generator_type == GeneratorType.IMPROVED:
            # Map to improved generator parameters
            return generator.generate_trajectory(
                word=config.word,
                sampling_rate=config.sampling_rate,
                duration=config.duration,
                user_speed=config.user_speed,
                precision=config.precision,
                style=config.style if isinstance(config.style, str) else 'natural'
            )
        else:  # ENHANCED
            # Basic parameters only
            generator.velocity_profile = 'bell'  # Use best profile
            return generator.generate_trajectory(
                word=config.word,
                sampling_rate=config.sampling_rate,
                duration=config.duration,
                user_speed=config.user_speed,
                precision=config.precision
            )
    
    def _generate_rnn(self, config: TrajectoryConfig) -> Dict:
        """Generate using RNN model"""
        if not self.rnn_model:
            raise ValueError("RNN model not loaded")
        
        # Generate seed sequence
        seed_length = 5
        seed = np.random.randn(seed_length, 3) * 0.1
        
        # Generate trajectory
        trajectory, _ = self.rnn_model.forward(seed)
        
        # Continue generation based on word length
        target_length = len(config.word) * 15  # Approximate
        
        while len(trajectory) < target_length:
            last_output = trajectory[-1:, :]
            next_output, _ = self.rnn_model.forward(last_output)
            trajectory = np.vstack([trajectory, next_output])
        
        # Add time stamps
        duration = len(trajectory) / config.sampling_rate
        t_stamps = np.linspace(0, duration, len(trajectory))
        trajectory = np.column_stack([trajectory[:, :2], t_stamps])
        
        # Package results
        return {
            'word': config.word,
            'trajectory': trajectory,
            'duration': duration,
            'sampling_rate': config.sampling_rate,
            'generator': 'RNN',
            'metadata': {
                'seed_length': seed_length,
                'target_length': target_length
            }
        }
    
    def _generate_jerk_min(self, config: TrajectoryConfig) -> Dict:
        """Generate using jerk minimization (placeholder)"""
        # Would integrate with Jerk-Minimization module
        # For now, fall back to optimized generator
        config.generator_type = GeneratorType.OPTIMIZED
        result = self._generate_standard(config)
        result['generator'] = 'jerk_minimization_fallback'
        return result
    
    def batch_generate(self, configs: List[TrajectoryConfig]) -> List[Dict]:
        """
        Generate multiple trajectories
        
        Args:
            configs: List of trajectory configurations
            
        Returns:
            List of trajectory dictionaries
        """
        results = []
        for config in configs:
            try:
                result = self.generate(config)
                results.append(result)
            except Exception as e:
                print(f"Error generating trajectory for '{config.word}': {e}")
                results.append(None)
        
        return results
    
    def compare_generators(self, word: str, style: str = 'natural') -> Dict:
        """
        Compare outputs from all available generators
        
        Args:
            word: Word to generate
            style: Style to use
            
        Returns:
            Comparison results
        """
        comparison = {
            'word': word,
            'style': style,
            'generators': {}
        }
        
        for gen_type in GeneratorType:
            if gen_type == GeneratorType.RNN and not self.rnn_model:
                continue
            
            config = TrajectoryConfig(
                word=word,
                generator_type=gen_type,
                style=style
            )
            
            try:
                result = self.generate(config)
                
                # Extract metrics
                trajectory = result.get('trajectory', np.array([]))
                if len(trajectory) > 0:
                    path_length = np.sum(np.linalg.norm(
                        np.diff(trajectory[:, :2], axis=0), axis=1
                    ))
                else:
                    path_length = 0
                
                comparison['generators'][gen_type.value] = {
                    'success': True,
                    'num_points': len(trajectory),
                    'duration': result.get('duration', 0),
                    'path_length': path_length,
                    'mean_jerk': result.get('metadata', {}).get('mean_jerk', None)
                }
            except Exception as e:
                comparison['generators'][gen_type.value] = {
                    'success': False,
                    'error': str(e)
                }
        
        return comparison
    
    def get_available_generators(self) -> List[str]:
        """Get list of available generators"""
        available = []
        for gen_type, generator in self.generators.items():
            if generator is not None:
                available.append(gen_type.value)
        
        if self.rnn_model is not None:
            available.append(GeneratorType.RNN.value)
        
        return available
    
    def get_generator_info(self, generator_type: GeneratorType) -> Dict:
        """
        Get information about a specific generator
        
        Args:
            generator_type: Type of generator
            
        Returns:
            Generator information
        """
        info = {
            'type': generator_type.value,
            'available': False,
            'features': [],
            'parameters': []
        }
        
        if generator_type in self.generators and self.generators[generator_type]:
            info['available'] = True
            
            if generator_type == GeneratorType.OPTIMIZED:
                info['features'] = [
                    'Continuous style parameters',
                    'Curvature-dependent velocity',
                    'Pressure variation',
                    'Adaptive sampling',
                    'Savitzky-Golay derivatives'
                ]
                info['parameters'] = [
                    'style (str or StyleParameters)',
                    'adaptive_sampling (bool)',
                    'sampling_rate (int)'
                ]
            elif generator_type == GeneratorType.IMPROVED:
                info['features'] = [
                    'Physical constraints',
                    'Multiple smoothing passes',
                    'Catmull-Rom splines',
                    'Jerk minimization'
                ]
                info['parameters'] = [
                    'style (str)',
                    'user_speed (float)',
                    'precision (float)'
                ]
            elif generator_type == GeneratorType.ENHANCED:
                info['features'] = [
                    'Basic smoothing',
                    'Velocity profiles',
                    'Corner smoothing',
                    'Noise addition'
                ]
                info['parameters'] = [
                    'user_speed (float)',
                    'precision (float)',
                    'sampling_rate (int)'
                ]
        
        if generator_type == GeneratorType.RNN and self.rnn_model:
            info['available'] = True
            info['features'] = [
                'Neural network-based',
                'Learned from data',
                'Auto-regressive generation'
            ]
            info['parameters'] = [
                'word (str)',
                'sampling_rate (int)'
            ]
        
        return info


# Example usage and testing
def demo():
    """Demonstrate unified API usage"""
    print("="*60)
    print("UNIFIED SWYPE API DEMONSTRATION")
    print("="*60)
    
    # Initialize API
    api = UnifiedSwypeAPI()
    
    # Show available generators
    print("\nAvailable generators:")
    for gen in api.get_available_generators():
        print(f"  - {gen}")
    
    # Test different generators
    test_word = "hello"
    
    print(f"\nGenerating '{test_word}' with different methods:")
    
    configs = [
        TrajectoryConfig(test_word, GeneratorType.ENHANCED),
        TrajectoryConfig(test_word, GeneratorType.IMPROVED),
        TrajectoryConfig(test_word, GeneratorType.OPTIMIZED),
    ]
    
    if 'rnn' in api.get_available_generators():
        configs.append(TrajectoryConfig(test_word, GeneratorType.RNN))
    
    results = api.batch_generate(configs)
    
    for config, result in zip(configs, results):
        if result:
            print(f"\n{config.generator_type.value}:")
            print(f"  Points: {len(result.get('trajectory', []))}")
            print(f"  Duration: {result.get('duration', 0):.2f}s")
            
            if 'metadata' in result and 'mean_jerk' in result['metadata']:
                print(f"  Mean jerk: {result['metadata']['mean_jerk']:,.0f}")
    
    # Compare generators
    print("\nGenerator comparison:")
    comparison = api.compare_generators("gesture", "natural")
    
    for gen_name, metrics in comparison['generators'].items():
        if metrics['success']:
            print(f"\n{gen_name}:")
            print(f"  Path length: {metrics['path_length']:.1f}")
            print(f"  Duration: {metrics['duration']:.2f}s")
            if metrics['mean_jerk']:
                print(f"  Mean jerk: {metrics['mean_jerk']:,.0f}")
    
    print("\n✅ API demonstration complete!")

if __name__ == "__main__":
    demo()