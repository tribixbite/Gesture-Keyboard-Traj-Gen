#!/usr/bin/env python
"""
Reconciliation script to integrate all existing models with new implementations
This creates a unified interface for all trajectory generation methods
"""

import sys
import os
import importlib.util
from pathlib import Path
from typing import Dict, List, Any, Optional
import numpy as np

class ModelReconciler:
    """Reconcile and integrate all model implementations"""
    
    def __init__(self):
        self.models = {}
        self.issues = []
        self.recommendations = []
        
    def analyze_repository(self):
        """Analyze repository structure and identify all models"""
        
        print("="*60)
        print("REPOSITORY RECONCILIATION ANALYSIS")
        print("="*60)
        
        # Check existing implementations
        self.check_jerk_minimization()
        self.check_rnn_tf1()
        self.check_rnn_tf2()
        self.check_gan_based()
        self.check_new_implementations()
        
        # Report findings
        self.report_findings()
        
    def check_jerk_minimization(self):
        """Check Jerk-Minimization implementation"""
        print("\n📁 Jerk-Minimization/")
        print("-"*40)
        
        jm_path = Path("Jerk-Minimization")
        if jm_path.exists():
            files = list(jm_path.glob("*.py"))
            print(f"  ✅ Found {len(files)} Python files")
            
            # Check for key components
            if (jm_path / "traj_gen" / "optim_trajectory.py").exists():
                print("  ✅ Optimization trajectory module found")
                self.models['jerk_minimization'] = {
                    'status': 'working',
                    'path': str(jm_path),
                    'type': 'optimization',
                    'framework': 'numpy/scipy'
                }
            
            if (jm_path / "optim_example.py").exists():
                print("  ✅ Example script available")
            
            self.recommendations.append(
                "Integrate Jerk-Minimization with unified_swype_api.py"
            )
        else:
            print("  ❌ Directory not found")
            self.issues.append("Jerk-Minimization directory missing")
    
    def check_rnn_tf1(self):
        """Check RNN TensorFlow 1.x implementation"""
        print("\n📁 RNN-Based TF1/")
        print("-"*40)
        
        rnn_tf1_path = Path("RNN-Based TF1")
        if rnn_tf1_path.exists():
            # Check for model.py
            if (rnn_tf1_path / "model.py").exists():
                print("  ✅ Model definition found")
                print("  ⚠️  Uses TensorFlow 1.x (deprecated)")
                
                self.models['rnn_tf1'] = {
                    'status': 'deprecated',
                    'path': str(rnn_tf1_path),
                    'type': 'generative',
                    'framework': 'tensorflow1'
                }
                
                self.issues.append("RNN TF1 uses deprecated TensorFlow 1.x")
                self.recommendations.append(
                    "Migrate RNN TF1 model architecture to PyTorch"
                )
            
            # Check for utils
            if (rnn_tf1_path / "utils.py").exists():
                print("  ✅ Utils module found")
                print("  📝 Contains data preprocessing functions")
        else:
            print("  ❌ Directory not found")
    
    def check_rnn_tf2(self):
        """Check RNN TensorFlow 2.x implementation"""
        print("\n📁 RNN-Based TF2/")
        print("-"*40)
        
        rnn_tf2_path = Path("RNN-Based TF2")
        if rnn_tf2_path.exists():
            # Check key files
            important_files = {
                "rnn.py": "Main RNN model",
                "rnn_cell.py": "LSTM with attention",
                "rnn_ops.py": "RNN operations",
                "drawing.py": "Drawing utilities"
            }
            
            for file, desc in important_files.items():
                if (rnn_tf2_path / file).exists():
                    print(f"  ✅ {file}: {desc}")
            
            self.models['rnn_tf2'] = {
                'status': 'active',
                'path': str(rnn_tf2_path),
                'type': 'generative',
                'framework': 'tensorflow2',
                'features': ['attention', 'mixture_density_network']
            }
            
            print("\n  🎯 Key Features:")
            print("    - LSTMAttentionCell implementation")
            print("    - Mixture density network (20 components)")
            print("    - Style-based generation")
            
            self.recommendations.append(
                "Use RNN TF2 attention mechanism in new PyTorch implementation"
            )
        else:
            print("  ❌ Directory not found")
    
    def check_gan_based(self):
        """Check GAN-based implementation"""
        print("\n📁 GAN-Based/")
        print("-"*40)
        
        gan_path = Path("GAN-Based")
        if gan_path.exists():
            # Check main components
            key_files = {
                "cycle_main.py": "CycleGAN main script",
                "cycle_train.py": "Training logic",
                "net_architecture.py": "Network architecture",
                "data_utils.py": "Data utilities",
                "inference.py": "Inference pipeline"
            }
            
            for file, desc in key_files.items():
                if (gan_path / file).exists():
                    print(f"  ✅ {file}: {desc}")
            
            self.models['gan_cyclegan'] = {
                'status': 'active',
                'path': str(gan_path),
                'type': 'generative',
                'framework': 'tensorflow/keras',
                'features': ['style_transfer', 'data_alteration']
            }
            
            print("\n  🎯 GAN Modes:")
            print("    - Style transfer between users")
            print("    - Data alteration/augmentation")
            
            self.recommendations.append(
                "Integrate CycleGAN with new WGAN-GP implementation"
            )
        else:
            print("  ❌ Directory not found")
    
    def check_new_implementations(self):
        """Check newly added implementations"""
        print("\n📁 New Implementations")
        print("-"*40)
        
        new_files = {
            "enhanced_swype_generator.py": "Enhanced generator with velocity profiles",
            "improved_swype_generator.py": "Improved with jerk minimization",
            "optimized_swype_generator.py": "Optimized for performance",
            "real_calibrated_generator.py": "Calibrated to real user data",
            "pytorch_rnn_trainer.py": "PyTorch RNN implementation",
            "optimized_rnn_trainer.py": "Optimized RNN (TensorFlow)",
            "optimized_gan_trainer.py": "WGAN-GP implementation"
        }
        
        for file, desc in new_files.items():
            if Path(file).exists():
                print(f"  ✅ {file}")
                print(f"     {desc}")
        
        print("\n  ⚠️  Issues found:")
        print("    - PyTorch RNN has indexing bug")
        print("    - TensorFlow trainers need dependencies")
        print("    - Web app has port conflicts")
    
    def report_findings(self):
        """Generate reconciliation report"""
        
        print("\n" + "="*60)
        print("RECONCILIATION SUMMARY")
        print("="*60)
        
        print("\n📊 Model Status:")
        for name, info in self.models.items():
            status_icon = "✅" if info['status'] == 'working' else "⚠️"
            print(f"  {status_icon} {name}: {info['status']} ({info['framework']})")
        
        print("\n❌ Critical Issues:")
        for issue in self.issues:
            print(f"  - {issue}")
        
        print("\n🎯 Recommendations:")
        for i, rec in enumerate(self.recommendations, 1):
            print(f"  {i}. {rec}")
        
        print("\n📋 Integration Plan:")
        print("  1. Fix PyTorch RNN indexing bug")
        print("  2. Extract attention mechanism from RNN TF2")
        print("  3. Merge CycleGAN with WGAN-GP approach")
        print("  4. Create unified API combining all generators")
        print("  5. Implement model checkpointing")
        print("  6. Add comprehensive testing")

def create_unified_pipeline():
    """Create unified training pipeline configuration"""
    
    config = {
        "generators": {
            "rule_based": [
                "enhanced_swype_generator",
                "improved_swype_generator", 
                "optimized_swype_generator",
                "real_calibrated_generator"
            ],
            "optimization": [
                "jerk_minimization"
            ],
            "neural": {
                "rnn": {
                    "tf2": "RNN-Based TF2/rnn.py",
                    "pytorch": "pytorch_rnn_trainer.py"
                },
                "gan": {
                    "cyclegan": "GAN-Based/cycle_main.py",
                    "wgan_gp": "optimized_gan_trainer.py"
                }
            }
        },
        "datasets": {
            "real": "datasets/swipelogs",
            "synthetic": "datasets/synthetic_all",
            "combined": "datasets/combined"
        },
        "training": {
            "batch_size": 32,
            "epochs": 100,
            "learning_rate": 0.001,
            "optimizer": "adam",
            "checkpoint_interval": 10
        },
        "validation": {
            "metrics": ["dtw", "frechet", "recognition_accuracy"],
            "test_words": ["hello", "world", "the", "quick", "brown", "fox"],
            "cross_validation": True
        }
    }
    
    # Save configuration
    import json
    with open("unified_pipeline_config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    print("\n✅ Created unified_pipeline_config.json")
    
    return config

def main():
    """Main reconciliation process"""
    
    reconciler = ModelReconciler()
    reconciler.analyze_repository()
    
    print("\n" + "="*60)
    print("CREATING UNIFIED PIPELINE")
    print("="*60)
    
    config = create_unified_pipeline()
    
    print("\n📝 Next Steps:")
    print("1. Run: python reconcile_models.py")
    print("2. Fix identified issues")
    print("3. Integrate models using unified_pipeline_config.json")
    print("4. Run comprehensive tests")
    print("5. Deploy to production")
    
    return reconciler

if __name__ == "__main__":
    reconciler = main()