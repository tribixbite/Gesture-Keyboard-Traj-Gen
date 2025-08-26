#!/usr/bin/env python
"""
Simple recognition decoder to test if synthetic traces are recognizable
Uses template matching and DTW for word recognition
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
import pickle
from pathlib import Path
from dataclasses import dataclass
import json

@dataclass
class RecognitionResult:
    """Recognition result for a trace"""
    word: str
    confidence: float
    top_5: List[Tuple[str, float]]
    success: bool
    
class SimpleDecoder:
    """
    Simple gesture decoder using template matching
    """
    
    def __init__(self, keyboard_layout_path: str = 'holokeyboard.csv'):
        """Initialize decoder with keyboard layout"""
        import pandas as pd
        self.keyboard_df = pd.read_csv(keyboard_layout_path, index_col='keys')
        self.templates = {}
        self.vocabulary = []
        
    def build_templates(self, words: List[str], generator=None):
        """
        Build template traces for vocabulary
        
        Args:
            words: List of vocabulary words
            generator: Trace generator to create templates
        """
        print(f"Building templates for {len(words)} words...")
        
        if generator is None:
            # Use our optimized generator
            from optimized_swype_generator import OptimizedSwypeGenerator
            generator = OptimizedSwypeGenerator()
        
        for word in words:
            try:
                # Generate template trace
                result = generator.generate_trajectory(word, style='precise')
                self.templates[word] = result['trajectory'][:, :2]  # Use only x,y
                self.vocabulary.append(word)
            except Exception as e:
                print(f"Error generating template for '{word}': {e}")
        
        print(f"✅ Built {len(self.templates)} templates")
    
    def dtw_distance(self, trace1: np.ndarray, trace2: np.ndarray) -> float:
        """
        Calculate DTW distance between two traces
        """
        # Ensure 2D coordinates
        t1 = trace1[:, :2] if trace1.shape[1] > 2 else trace1
        t2 = trace2[:, :2] if trace2.shape[1] > 2 else trace2
        
        n, m = len(t1), len(t2)
        
        # DTW matrix
        dtw = np.full((n + 1, m + 1), np.inf)
        dtw[0, 0] = 0
        
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                cost = np.linalg.norm(t1[i-1] - t2[j-1])
                dtw[i, j] = cost + min(
                    dtw[i-1, j],     # Insertion
                    dtw[i, j-1],     # Deletion  
                    dtw[i-1, j-1]    # Match
                )
        
        return dtw[n, m]
    
    def decode(self, trace: np.ndarray, top_k: int = 5) -> RecognitionResult:
        """
        Decode a trace to find most likely word
        
        Args:
            trace: Input trajectory to decode
            top_k: Number of top candidates to return
            
        Returns:
            RecognitionResult with decoded word and confidence
        """
        if len(self.templates) == 0:
            raise ValueError("No templates loaded. Call build_templates first.")
        
        # Ensure trace is 2D
        if trace.shape[1] > 2:
            trace = trace[:, :2]
        
        # Calculate distances to all templates
        distances = []
        for word, template in self.templates.items():
            dist = self.dtw_distance(trace, template)
            distances.append((word, dist))
        
        # Sort by distance (lower is better)
        distances.sort(key=lambda x: x[1])
        
        # Get top candidates
        top_candidates = distances[:top_k]
        
        # Calculate confidence (inverse distance normalized)
        best_word, best_dist = distances[0]
        
        # Simple confidence calculation
        if best_dist < 100:
            confidence = 0.9
        elif best_dist < 500:
            confidence = 0.7
        elif best_dist < 1000:
            confidence = 0.5
        else:
            confidence = 0.3
        
        # Success if confident enough
        success = confidence > 0.5
        
        return RecognitionResult(
            word=best_word,
            confidence=confidence,
            top_5=[(w, 1.0/(1.0 + d/100)) for w, d in top_candidates],
            success=success
        )
    
    def test_accuracy(self, test_traces: List[Dict], verbose: bool = True) -> Dict:
        """
        Test decoder accuracy on a set of traces
        
        Args:
            test_traces: List of traces with ground truth
            verbose: Print detailed results
            
        Returns:
            Accuracy metrics
        """
        if not test_traces:
            return {'accuracy': 0, 'top_5_accuracy': 0}
        
        correct = 0
        top_5_correct = 0
        results = []
        
        for i, test in enumerate(test_traces):
            trace = test['trace']
            true_word = test['word']
            
            # Skip if word not in vocabulary
            if true_word not in self.vocabulary:
                continue
            
            # Decode
            result = self.decode(trace)
            results.append(result)
            
            # Check accuracy
            if result.word == true_word:
                correct += 1
                
            # Check top-5 accuracy
            if true_word in [w for w, _ in result.top_5]:
                top_5_correct += 1
            
            if verbose and i < 10:  # Show first 10
                status = "✅" if result.word == true_word else "❌"
                print(f"{status} True: '{true_word}' → Decoded: '{result.word}' (conf: {result.confidence:.2f})")
        
        n = len(results)
        accuracy = correct / n if n > 0 else 0
        top_5_accuracy = top_5_correct / n if n > 0 else 0
        
        return {
            'accuracy': accuracy,
            'top_5_accuracy': top_5_accuracy,
            'total_tested': n,
            'correct': correct,
            'top_5_correct': top_5_correct
        }


class NeuralDecoder:
    """
    Neural network-based decoder (simplified)
    """
    
    def __init__(self, model_path: Optional[str] = None):
        """Initialize neural decoder"""
        self.model = None
        if model_path and Path(model_path).exists():
            self.load_model(model_path)
    
    def load_model(self, model_path: str):
        """Load pre-trained model"""
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
    
    def decode(self, trace: np.ndarray) -> RecognitionResult:
        """Decode using neural network"""
        # Simplified - would use actual neural network
        return RecognitionResult(
            word="neural_decode",
            confidence=0.8,
            top_5=[("neural", 0.8), ("decode", 0.7)],
            success=True
        )


def test_recognition_system():
    """
    Test complete recognition system
    """
    print("="*60)
    print("RECOGNITION SYSTEM TEST")
    print("="*60)
    
    # Initialize decoder
    decoder = SimpleDecoder()
    
    # Build vocabulary
    vocabulary = [
        'the', 'and', 'you', 'that', 'was', 'for', 'are', 'with',
        'his', 'they', 'hello', 'world', 'computer', 'keyboard',
        'gesture', 'typing', 'swipe', 'trace', 'phone', 'test'
    ]
    
    decoder.build_templates(vocabulary)
    
    # Generate test traces
    print("\nGenerating test traces...")
    from optimized_swype_generator import OptimizedSwypeGenerator
    
    generator = OptimizedSwypeGenerator()
    
    test_traces = []
    for word in vocabulary[:10]:  # Test first 10 words
        # Generate with different styles to test robustness
        for style in ['natural', 'fast', 'sloppy']:
            try:
                result = generator.generate_trajectory(word, style=style)
                test_traces.append({
                    'word': word,
                    'trace': result['trajectory'],
                    'style': style
                })
            except Exception as e:
                print(f"Error generating '{word}' ({style}): {e}")
    
    print(f"Generated {len(test_traces)} test traces")
    
    # Test accuracy
    print("\nTesting recognition accuracy...")
    print("-"*60)
    
    metrics = decoder.test_accuracy(test_traces, verbose=True)
    
    # Print summary
    print("\n" + "="*60)
    print("RECOGNITION RESULTS")
    print("="*60)
    
    print(f"Accuracy: {metrics['accuracy']*100:.1f}% ({metrics['correct']}/{metrics['total_tested']})")
    print(f"Top-5 Accuracy: {metrics['top_5_accuracy']*100:.1f}% ({metrics['top_5_correct']}/{metrics['total_tested']})")
    
    if metrics['accuracy'] > 0.85:
        print("\n✅ EXCELLENT: Traces are highly recognizable")
    elif metrics['accuracy'] > 0.70:
        print("\n✅ GOOD: Traces are recognizable")
    elif metrics['accuracy'] > 0.50:
        print("\n⚠️  FAIR: Traces need improvement")
    else:
        print("\n❌ POOR: Traces are not recognizable")
    
    return metrics


def test_with_real_data():
    """
    Test with real gesture data if available
    """
    print("\n" + "="*60)
    print("TESTING WITH REAL DATA")
    print("="*60)
    
    # Check for parsed dollar dataset
    dollar_file = Path("real_datasets/parsed/dollar_parsed.pkl")
    
    if dollar_file.exists():
        with open(dollar_file, 'rb') as f:
            dollar_data = pickle.load(f)
        
        print(f"Loaded {dollar_data['num_gestures']} real gestures")
        
        # Create simple mapping
        gesture_map = {
            'circle': 'o',
            'x': 'x',
            'check': 'v',
            'triangle': 't',
            'rectangle': 'r',
            'v': 'v'
        }
        
        # Build decoder with mapped vocabulary
        decoder = SimpleDecoder()
        decoder.build_templates(list(gesture_map.values()))
        
        # Test on real gestures
        test_traces = []
        for gesture in dollar_data['gestures'][:20]:
            # Extract base gesture type
            base_type = gesture['word'].rstrip('0123456789')
            
            if base_type in gesture_map:
                test_traces.append({
                    'word': gesture_map[base_type],
                    'trace': gesture['trace']
                })
        
        if test_traces:
            print(f"\nTesting on {len(test_traces)} real gesture traces...")
            metrics = decoder.test_accuracy(test_traces, verbose=False)
            
            print(f"Real Data Accuracy: {metrics['accuracy']*100:.1f}%")
            print(f"Real Data Top-5: {metrics['top_5_accuracy']*100:.1f}%")
    else:
        print("No real gesture data available")


def main():
    """Main test function"""
    
    # Test with synthetic traces
    synthetic_metrics = test_recognition_system()
    
    # Save metrics
    with open('recognition_metrics.json', 'w') as f:
        json.dump(synthetic_metrics, f, indent=2)
    
    # Test with real data if available
    test_with_real_data()
    
    print("\n✅ Recognition decoder ready!")
    print("Metrics saved to recognition_metrics.json")
    
    return synthetic_metrics

if __name__ == "__main__":
    metrics = main()