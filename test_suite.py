#!/usr/bin/env python
"""
Comprehensive test suite for all trajectory generation components
Covers generators, models, APIs, and utility functions
"""

import unittest
import torch
import numpy as np
import json
import tempfile
import os
import sys
from pathlib import Path
from typing import List, Dict, Any
import time
import warnings

# Suppress warnings for cleaner test output
warnings.filterwarnings('ignore')

class TestTrajectoryGenerators(unittest.TestCase):
    """Test all trajectory generators"""
    
    def setUp(self):
        """Set up test environment"""
        self.test_words = ["hello", "world", "test", "python", "ai"]
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up test environment"""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_enhanced_generator(self):
        """Test enhanced swype generator"""
        try:
            from enhanced_swype_generator import EnhancedSwypeGenerator
            
            generator = EnhancedSwypeGenerator()
            
            # Test single generation
            result = generator.generate_trajectory("hello")
            self.assertIsInstance(result, dict)
            self.assertIn('trajectory', result)
            self.assertIn('word', result)
            
            # Test trajectory properties
            trajectory = result['trajectory']
            self.assertGreater(len(trajectory), 0)
            self.assertEqual(trajectory.shape[1], 3)  # x, y, t
            
            print("✅ Enhanced generator test passed")
            
        except ImportError:
            self.skipTest("Enhanced generator not available")
    
    def test_real_calibrated_generator(self):
        """Test real calibrated generator"""
        try:
            from real_calibrated_generator import RealCalibratedGenerator
            
            generator = RealCalibratedGenerator()
            
            for word in self.test_words[:3]:
                trajectory = generator.generate_trajectory(word)
                self.assertIsInstance(trajectory, list)
                self.assertGreater(len(trajectory), 0)
                
                # Check trajectory format
                first_point = trajectory[0]
                self.assertIn('x', first_point)
                self.assertIn('y', first_point)
                self.assertIn('t', first_point)
                self.assertIn('p', first_point)
            
            print("✅ Real calibrated generator test passed")
            
        except ImportError:
            self.skipTest("Real calibrated generator not available")
    
    def test_pytorch_models(self):
        """Test PyTorch model components"""
        try:
            import torch
            
            # Test attention RNN
            from pytorch_attention_rnn import AttentionRNN
            model = AttentionRNN(vocab_size=30, hidden_size=64)
            
            # Test forward pass
            batch_size, seq_len, word_len = 2, 20, 5
            trajectories = torch.randn(batch_size, seq_len, 3)
            words = torch.randint(0, 30, (batch_size, word_len))
            
            with torch.no_grad():
                output = model(trajectories, words)
            
            self.assertEqual(output.shape[:2], (batch_size, seq_len))
            
            print("✅ PyTorch attention RNN test passed")
            
        except ImportError:
            self.skipTest("PyTorch models not available")
    
    def test_transformer_model(self):
        """Test transformer model"""
        try:
            import torch
            from pytorch_transformer_generator import TrajectoryTransformer
            
            model = TrajectoryTransformer(
                vocab_size=30,
                d_model=64,
                n_heads=4,
                n_encoder_layers=2,
                n_decoder_layers=2,
                num_mixture_components=5
            )
            
            # Test forward pass
            batch_size, char_len, traj_len = 2, 10, 15
            char_seq = torch.randint(0, 30, (batch_size, char_len))
            trajectory = torch.randn(batch_size, traj_len, 3)
            
            with torch.no_grad():
                output = model(char_seq, trajectory)
            
            expected_output_size = 6 * 5 + 1  # 6 * num_components + 1
            self.assertEqual(output.shape, (batch_size, traj_len, expected_output_size))
            
            # Test generation
            generated = model.generate(char_seq[:1], max_length=10)
            self.assertEqual(generated.shape[0], 1)
            self.assertEqual(generated.shape[2], 3)
            self.assertLessEqual(generated.shape[1], 10)
            
            print("✅ Transformer model test passed")
            
        except ImportError:
            self.skipTest("Transformer model not available")
    
    def test_wgan_model(self):
        """Test WGAN-GP model"""
        try:
            import torch
            from pytorch_wgan_gp import Generator, Discriminator
            
            vocab_size, latent_dim, max_len = 30, 64, 50
            generator = Generator(vocab_size, latent_dim, max_len)
            discriminator = Discriminator(vocab_size, max_len)
            
            # Test generator
            batch_size = 2
            noise = torch.randn(batch_size, latent_dim)
            word = torch.randint(0, vocab_size, (batch_size, 5))
            
            with torch.no_grad():
                fake_traj = generator(noise, word)
                self.assertEqual(fake_traj.shape, (batch_size, max_len, 3))
                
                # Test discriminator
                validity = discriminator(fake_traj, word)
                self.assertEqual(validity.shape, (batch_size, 1))
            
            print("✅ WGAN-GP model test passed")
            
        except ImportError:
            self.skipTest("WGAN-GP model not available")

class TestUnifiedAPI(unittest.TestCase):
    """Test unified generator API"""
    
    def setUp(self):
        self.test_words = ["hello", "test", "api"]
    
    def test_api_initialization(self):
        """Test API initialization"""
        try:
            from unified_generator_api import UnifiedGeneratorAPI
            
            api = UnifiedGeneratorAPI()
            generators = api.list_generators()
            
            self.assertIsInstance(generators, list)
            self.assertGreater(len(generators), 0)
            
            # Test generator info
            info = api.get_generator_info()
            self.assertIsInstance(info, list)
            self.assertEqual(len(info), len(generators))
            
            print(f"✅ API initialized with {len(generators)} generators")
            
        except ImportError:
            self.skipTest("Unified API not available")
    
    def test_generation(self):
        """Test trajectory generation through API"""
        try:
            from unified_generator_api import UnifiedGeneratorAPI
            
            api = UnifiedGeneratorAPI()
            if not api.generators:
                self.skipTest("No generators available")
            
            # Test single generation
            result = api.generate("hello")
            self.assertIsNotNone(result)
            self.assertEqual(result.word, "hello")
            self.assertIsInstance(result.trajectory, list)
            self.assertGreater(len(result.trajectory), 0)
            
            # Test batch generation
            results = api.generate_batch(self.test_words)
            self.assertEqual(len(results), len(self.test_words))
            
            for i, result in enumerate(results):
                self.assertEqual(result.word, self.test_words[i])
                if result.trajectory:  # Some may fail
                    self.assertIsInstance(result.trajectory, list)
            
            print("✅ API generation test passed")
            
        except ImportError:
            self.skipTest("Unified API not available")

class TestCheckpointManager(unittest.TestCase):
    """Test model checkpoint management"""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_checkpoint_operations(self):
        """Test checkpoint save/load operations"""
        try:
            import torch
            from model_checkpoint_manager import CheckpointManager
            
            # Create checkpoint manager
            manager = CheckpointManager(self.temp_dir)
            
            # Create dummy model
            model = torch.nn.Sequential(
                torch.nn.Linear(10, 20),
                torch.nn.ReLU(),
                torch.nn.Linear(20, 1)
            )
            
            optimizer = torch.optim.Adam(model.parameters())
            
            # Test save checkpoint
            checkpoint = manager.save_checkpoint(
                model=model,
                model_type="test",
                model_name="dummy",
                epoch=1,
                optimizer=optimizer,
                metrics={'loss': 0.5},
                config={'hidden_size': 20}
            )
            
            self.assertIsNotNone(checkpoint)
            self.assertTrue(checkpoint.is_valid())
            
            # Test load checkpoint
            new_model = torch.nn.Sequential(
                torch.nn.Linear(10, 20),
                torch.nn.ReLU(),
                torch.nn.Linear(20, 1)
            )
            
            data = manager.load_checkpoint(
                model=new_model,
                model_type="test",
                model_name="dummy",
                epoch=1
            )
            
            self.assertIsNotNone(data)
            self.assertEqual(data['epoch'], 1)
            
            # Test checkpoint listing
            checkpoints = manager.list_checkpoints()
            self.assertGreater(len(checkpoints), 0)
            
            print("✅ Checkpoint manager test passed")
            
        except ImportError:
            self.skipTest("Checkpoint manager not available")

class TestDatasetClasses(unittest.TestCase):
    """Test dataset classes and data loading"""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        
        # Create mock swipelog data
        self.mock_data_dir = Path(self.temp_dir) / "mock_swipelogs"
        self.mock_data_dir.mkdir()
        
        # Create a mock log file
        mock_log = self.mock_data_dir / "mock_001.log"
        with open(mock_log, 'w') as f:
            f.write("word x_pos y_pos timestamp\n")
            f.write("hello 100 200 0.0\n")
            f.write("hello 120 210 0.1\n")
            f.write("hello 140 220 0.2\n")
            f.write("world 200 300 0.3\n")
            f.write("world 220 310 0.4\n")
    
    def tearDown(self):
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_pytorch_dataset(self):
        """Test PyTorch dataset classes"""
        try:
            import torch
            from pytorch_rnn_trainer import SwipelogDataset
            
            # Create dummy data
            traces = [np.random.randn(10, 2) for _ in range(5)]
            labels = ["hello", "world", "test", "data", "set"]
            char_to_idx = {char: i for i, char in enumerate("abcdefghijklmnopqrstuvwxyz")}
            
            dataset = SwipelogDataset(traces, labels, char_to_idx)
            
            self.assertEqual(len(dataset), 5)
            
            # Test data loading
            trajectory, word = dataset[0]
            self.assertIsInstance(trajectory, torch.Tensor)
            self.assertIsInstance(word, torch.Tensor)
            self.assertEqual(trajectory.shape[1], 3)  # x, y, pen_state
            
            print("✅ PyTorch dataset test passed")
            
        except ImportError:
            self.skipTest("PyTorch dataset not available")
    
    def test_transformer_dataset(self):
        """Test transformer dataset"""
        try:
            from pytorch_transformer_generator import TransformerDataset
            
            dataset = TransformerDataset(str(self.mock_data_dir))
            
            if len(dataset) > 0:
                trajectory, char_seq = dataset[0]
                self.assertIsInstance(trajectory, torch.Tensor)
                self.assertIsInstance(char_seq, torch.Tensor)
                self.assertEqual(trajectory.shape[1], 3)
                self.assertGreater(char_seq.shape[0], 0)
            
            print("✅ Transformer dataset test passed")
            
        except ImportError:
            self.skipTest("Transformer dataset not available")

class TestUtilities(unittest.TestCase):
    """Test utility functions and helpers"""
    
    def test_data_loading(self):
        """Test data loading functions"""
        try:
            from pytorch_rnn_trainer import load_data
            
            traces, labels, char_to_idx, idx_to_char = load_data()
            
            self.assertIsInstance(traces, list)
            self.assertIsInstance(labels, list)
            self.assertIsInstance(char_to_idx, dict)
            self.assertIsInstance(idx_to_char, dict)
            
            if traces and labels:
                self.assertEqual(len(traces), len(labels))
                self.assertGreater(len(char_to_idx), 0)
            
            print(f"✅ Data loading test passed ({len(traces)} traces)")
            
        except ImportError:
            self.skipTest("Data loading functions not available")
    
    def test_trajectory_metrics(self):
        """Test trajectory quality metrics"""
        # Create synthetic trajectory
        trajectory = []
        for i in range(20):
            trajectory.append({
                'x': i * 10 + np.random.normal(0, 2),
                'y': 100 + np.sin(i * 0.5) * 20 + np.random.normal(0, 1),
                't': i * 50,
                'p': 1 if i < 19 else 0
            })
        
        # Test metrics calculation (from GenerationResult)
        from unified_generator_api import GenerationResult
        result = GenerationResult(trajectory, "test", "test")
        
        self.assertGreater(result.metadata['path_length'], 0)
        self.assertGreater(result.metadata['duration'], 0)
        self.assertEqual(result.metadata['point_count'], len(trajectory))
        
        print("✅ Trajectory metrics test passed")

class TestPerformance(unittest.TestCase):
    """Performance and stress tests"""
    
    def test_generation_speed(self):
        """Test generation speed across different generators"""
        try:
            from unified_generator_api import UnifiedGeneratorAPI
            
            api = UnifiedGeneratorAPI()
            if not api.generators:
                self.skipTest("No generators available")
            
            test_words = ["hello", "world", "test"] * 10  # 30 words
            
            for gen_type in api.generators:
                start_time = time.time()
                
                success_count = 0
                for word in test_words[:5]:  # Test with 5 words
                    try:
                        result = api.generate(word, gen_type)
                        if result.trajectory:
                            success_count += 1
                    except Exception:
                        pass
                
                elapsed_time = time.time() - start_time
                
                if success_count > 0:
                    avg_time = elapsed_time / success_count
                    print(f"✅ {gen_type}: {avg_time:.3f}s/word ({success_count}/{len(test_words[:5])} success)")
                else:
                    print(f"⚠️ {gen_type}: No successful generations")
        
        except ImportError:
            self.skipTest("Performance test requires unified API")
    
    def test_memory_usage(self):
        """Test memory usage of models"""
        try:
            import torch
            import psutil
            import os
            
            process = psutil.Process(os.getpid())
            initial_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            # Load multiple models
            from pytorch_attention_rnn import AttentionRNN
            from pytorch_transformer_generator import TrajectoryTransformer
            
            models = []
            models.append(AttentionRNN(vocab_size=50))
            models.append(TrajectoryTransformer(vocab_size=50, d_model=128))
            
            final_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_increase = final_memory - initial_memory
            
            self.assertLess(memory_increase, 500)  # Should use less than 500MB
            
            print(f"✅ Memory test passed (used {memory_increase:.1f} MB)")
            
        except ImportError:
            self.skipTest("Memory test requires psutil and models")

def run_comprehensive_tests():
    """Run all tests and generate report"""
    
    print("="*70)
    print("COMPREHENSIVE TEST SUITE")
    print("="*70)
    
    # Collect all test classes
    test_classes = [
        TestTrajectoryGenerators,
        TestUnifiedAPI, 
        TestCheckpointManager,
        TestDatasetClasses,
        TestUtilities,
        TestPerformance
    ]
    
    results = {}
    total_tests = 0
    total_passed = 0
    total_failed = 0
    total_skipped = 0
    
    for test_class in test_classes:
        print(f"\n📋 Running {test_class.__name__}...")
        
        suite = unittest.TestLoader().loadTestsFromTestCase(test_class)
        runner = unittest.TextTestRunner(verbosity=0, stream=open(os.devnull, 'w'))
        result = runner.run(suite)
        
        class_results = {
            'tests_run': result.testsRun,
            'failures': len(result.failures),
            'errors': len(result.errors),
            'skipped': len(result.skipped),
            'success': result.testsRun - len(result.failures) - len(result.errors) - len(result.skipped)
        }
        
        results[test_class.__name__] = class_results
        
        total_tests += class_results['tests_run']
        total_passed += class_results['success']
        total_failed += class_results['failures'] + class_results['errors']
        total_skipped += class_results['skipped']
        
        print(f"   {class_results['success']}/{class_results['tests_run']} passed, {class_results['skipped']} skipped")
    
    # Summary report
    print(f"\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    print(f"Total Tests: {total_tests}")
    print(f"✅ Passed: {total_passed}")
    print(f"❌ Failed: {total_failed}")
    print(f"⏭️ Skipped: {total_skipped}")
    print(f"Success Rate: {total_passed/total_tests*100:.1f}%")
    
    # Detailed results
    print(f"\nDetailed Results:")
    for class_name, result in results.items():
        status = "✅" if result['failures'] == 0 and result['errors'] == 0 else "❌"
        print(f"  {status} {class_name}: {result['success']}/{result['tests_run']} passed")
    
    if total_failed == 0:
        print(f"\n🎉 All tests passed successfully!")
    else:
        print(f"\n⚠️ {total_failed} tests failed - check individual test output")
    
    return total_failed == 0

if __name__ == "__main__":
    # Add current directory to Python path
    sys.path.append('.')
    
    success = run_comprehensive_tests()
    sys.exit(0 if success else 1)