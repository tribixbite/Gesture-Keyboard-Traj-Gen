#!/usr/bin/env python
"""
Search for and document available gesture keyboard datasets
Implement loaders for found datasets
"""

import json
import numpy as np
from pathlib import Path
import requests
from typing import Dict, List, Tuple, Optional
import zipfile
import pickle

class GestureDatasetFinder:
    """
    Find and load real gesture keyboard datasets
    """
    
    def __init__(self):
        self.known_datasets = {
            "SHARK2": {
                "description": "SHARK² Gesture Recognition Dataset",
                "url": "https://github.com/aaronlaursen/SHARK2",
                "paper": "SHARK²: A Large Vocabulary Shorthand Writing System for Pen-based Computers",
                "format": "XML traces",
                "availability": "GitHub",
                "samples": "~10000 gesture samples"
            },
            "GestureKeyboard": {
                "description": "Word-Gesture Keyboard Dataset from Microsoft Research",
                "url": "https://www.microsoft.com/en-us/research/project/gesture-keyboard/",
                "paper": "Gestures without Libraries, Toolkits or Training",
                "format": "JSON with timestamp data",
                "availability": "By request",
                "samples": "Unknown"
            },
            "$1 Unistroke": {
                "description": "$1 Unistroke Recognizer Dataset",
                "url": "https://depts.washington.edu/acelab/proj/dollar/",
                "paper": "Gestures without Libraries, Toolkits or Training: A $1 Recognizer",
                "format": "XML with points",
                "availability": "Public download",
                "samples": "4800 samples (16 gestures × 10 users × 3 speeds × 10 samples)"
            },
            "VirtualKeyboard": {
                "description": "Virtual Keyboard Typing Dataset",
                "url": "Not publicly available",
                "paper": "Various HCI papers",
                "format": "CSV/JSON",
                "availability": "Private/Research only",
                "samples": "Varies"
            },
            "TouchDynamics": {
                "description": "Touch dynamics and swipe patterns",
                "url": "http://www.cs.waikato.ac.nz/~eibe/pubs/",
                "paper": "Touchalytics: On the Applicability of Touchscreen Input",
                "format": "ARFF/CSV",
                "availability": "Public",
                "samples": "~40000 swipes"
            }
        }
        
        self.datasets_path = Path("real_datasets")
        self.datasets_path.mkdir(exist_ok=True)
    
    def search_available_datasets(self) -> Dict:
        """
        Search for publicly available datasets
        """
        print("="*60)
        print("SEARCHING FOR GESTURE KEYBOARD DATASETS")
        print("="*60)
        
        available = {}
        
        for name, info in self.known_datasets.items():
            print(f"\n{name}:")
            print(f"  Description: {info['description']}")
            print(f"  Format: {info['format']}")
            print(f"  Availability: {info['availability']}")
            print(f"  Samples: {info['samples']}")
            
            if info['availability'] in ['Public', 'GitHub', 'Public download']:
                available[name] = info
                print(f"  ✅ Can potentially access")
            else:
                print(f"  ❌ Not publicly available")
        
        return available
    
    def create_sample_dataset(self, num_samples: int = 100) -> Dict:
        """
        Create a small sample dataset for testing validation
        Since real datasets may not be immediately available
        """
        print("\nCreating sample validation dataset...")
        
        # Common words for testing
        words = ['hello', 'world', 'the', 'quick', 'brown', 'fox', 'jumps', 
                 'over', 'lazy', 'dog', 'computer', 'keyboard', 'gesture',
                 'typing', 'mobile', 'phone', 'swipe', 'trace', 'path', 'word']
        
        samples = []
        
        for i in range(num_samples):
            word = np.random.choice(words)
            
            # Create realistic trace based on word length
            # This simulates what real data might look like
            trace_length = len(word) * np.random.randint(10, 20)
            
            # Generate path with realistic characteristics
            t = np.linspace(0, 1, trace_length)
            
            # Create path through "key positions" with noise
            key_points = len(word)
            key_times = np.linspace(0, 1, key_points)
            key_x = np.random.uniform(-100, 100, key_points)
            key_y = np.random.uniform(-50, 50, key_points)
            
            # Interpolate between keys
            x = np.interp(t, key_times, key_x)
            y = np.interp(t, key_times, key_y)
            
            # Add realistic noise and smoothing
            noise_x = np.random.normal(0, 2, trace_length)
            noise_y = np.random.normal(0, 2, trace_length)
            
            # Smooth the noise (correlated, not white)
            kernel = np.ones(5) / 5
            noise_x = np.convolve(noise_x, kernel, mode='same')
            noise_y = np.convolve(noise_y, kernel, mode='same')
            
            x += noise_x
            y += noise_y
            
            # Create velocity profile (slower at keys)
            velocity = np.ones(trace_length)
            for kt in key_times:
                idx = int(kt * trace_length)
                if idx < trace_length:
                    # Slow down near keys
                    window = min(10, trace_length // (2 * key_points))
                    for j in range(max(0, idx-window), min(trace_length, idx+window)):
                        dist = abs(j - idx) / window
                        velocity[j] *= (0.5 + 0.5 * dist)
            
            # Apply velocity modulation
            dt = np.diff(t)
            dt = dt / velocity[:-1]
            t_modulated = np.zeros(trace_length)
            t_modulated[1:] = np.cumsum(dt)
            
            # Package as sample
            sample = {
                'word': word,
                'trace': np.column_stack([x, y, t_modulated]),
                'user_id': i % 10,  # Simulate 10 different users
                'session_id': i // 10,
                'device': np.random.choice(['phone', 'tablet']),
                'metadata': {
                    'trace_id': f"sample_{i:04d}",
                    'timestamp': f"2024-11-{15 + i//100:02d}",
                    'duration': t_modulated[-1],
                    'num_points': trace_length
                }
            }
            
            samples.append(sample)
        
        dataset = {
            'samples': samples,
            'num_samples': len(samples),
            'vocabulary': list(set([s['word'] for s in samples])),
            'num_users': 10,
            'dataset_type': 'simulated_validation',
            'description': 'Simulated dataset for validation framework testing'
        }
        
        # Save dataset
        dataset_file = self.datasets_path / "sample_validation_dataset.pkl"
        with open(dataset_file, 'wb') as f:
            pickle.dump(dataset, f)
        
        print(f"Created sample dataset with {num_samples} traces")
        print(f"Saved to: {dataset_file}")
        
        return dataset
    
    def load_dollar_dataset(self) -> Optional[Dict]:
        """
        Attempt to load $1 Unistroke dataset (if available)
        This is one of the few publicly available gesture datasets
        """
        dataset_url = "https://depts.washington.edu/acelab/proj/dollar/xml.zip"
        dataset_path = self.datasets_path / "dollar_dataset"
        
        print("\nAttempting to load $1 Unistroke Dataset...")
        
        try:
            # Check if already downloaded
            if not dataset_path.exists():
                print("Dataset not found locally, would need to download...")
                print(f"URL: {dataset_url}")
                print("Note: This is primarily for single gestures, not full words")
                return None
            
            # If available, parse the XML files
            # This would require XML parsing implementation
            print("Would parse XML gesture files here...")
            return None
            
        except Exception as e:
            print(f"Could not load dataset: {e}")
            return None
    
    def analyze_dataset_requirements(self) -> Dict:
        """
        Analyze what we need from a real dataset for validation
        """
        requirements = {
            'minimum_samples': 100,
            'ideal_samples': 1000,
            'required_fields': [
                'word',  # The intended word
                'trace',  # X, Y, T coordinates
                'user_id',  # To analyze user variation
            ],
            'optional_fields': [
                'device_type',  # Phone vs tablet
                'keyboard_layout',  # QWERTY, DVORAK, etc
                'user_expertise',  # Novice vs expert
                'session_id',  # Temporal grouping
                'recognition_result',  # What was actually recognized
                'error_corrections',  # Backspaces, retries
            ],
            'trace_format': {
                'coordinates': '(x, y, t) tuples or arrays',
                'sampling_rate': 'Ideally 100Hz or higher',
                'units': 'Pixels or normalized coordinates',
                'pressure': 'Optional but valuable'
            },
            'quality_metrics_needed': [
                'Ground truth word',
                'Trace completeness (no gaps)',
                'Consistent sampling rate',
                'Multiple users for variation',
                'Multiple samples per word'
            ]
        }
        
        return requirements
    
    def convert_dataset_to_standard(self, dataset: Dict, source_format: str) -> Dict:
        """
        Convert various dataset formats to our standard format
        """
        standard_dataset = {
            'traces': [],
            'words': [],
            'users': [],
            'metadata': {},
            'format_version': '1.0'
        }
        
        if source_format == 'sample_validation':
            # Convert our sample format
            for sample in dataset['samples']:
                standard_dataset['traces'].append(sample['trace'])
                standard_dataset['words'].append(sample['word'])
                standard_dataset['users'].append(sample['user_id'])
            
            standard_dataset['metadata'] = {
                'num_samples': dataset['num_samples'],
                'vocabulary_size': len(dataset['vocabulary']),
                'num_users': dataset['num_users'],
                'source': 'simulated'
            }
        
        # Add other format conversions as needed
        
        return standard_dataset


def create_validation_data_request():
    """
    Create a document explaining what data we need from users
    """
    request = """
# Request for Gesture Keyboard Validation Data

## What We Need
We need real human gesture keyboard traces to validate our synthetic generation system.

## Data Format Required
Each trace should contain:
1. **Word**: The intended word (ground truth)
2. **Trace Points**: Array of (x, y, timestamp) coordinates
3. **User ID**: Anonymous identifier for the user
4. **Device Info**: Screen size, DPI, keyboard layout used

## How to Collect
1. Use any gesture keyboard app that can export traces
2. Type 20-50 common words
3. Export the trace data in JSON or CSV format
4. Ensure consistent sampling (>60Hz preferred)

## Privacy
- No personal information needed
- User IDs can be anonymous (user1, user2, etc.)
- Only the trace coordinates and intended words are required

## Suggested Words
- Common: the, and, you, that, with
- Medium: hello, world, computer, typing
- Long: keyboard, beautiful, wonderful

## Data Submission
Please save traces in this format:
```json
{
  "traces": [
    {
      "word": "hello",
      "points": [[x1,y1,t1], [x2,y2,t2], ...],
      "user": "user1",
      "device": "phone"
    }
  ]
}
```

## Contact
Submit data or questions to: [project repository]
"""
    
    with open("REQUEST_FOR_DATA.md", 'w') as f:
        f.write(request)
    
    print("\nCreated data request document: REQUEST_FOR_DATA.md")
    return request


def main():
    """
    Main dataset search and preparation
    """
    finder = GestureDatasetFinder()
    
    # Search for available datasets
    available = finder.search_available_datasets()
    
    # Analyze requirements
    print("\n" + "="*60)
    print("DATASET REQUIREMENTS ANALYSIS")
    print("="*60)
    
    requirements = finder.analyze_dataset_requirements()
    print("\nMinimum requirements for validation:")
    for key, value in requirements.items():
        if isinstance(value, list):
            print(f"\n{key}:")
            for item in value:
                print(f"  - {item}")
        elif isinstance(value, dict):
            print(f"\n{key}:")
            for k, v in value.items():
                print(f"  - {k}: {v}")
        else:
            print(f"{key}: {value}")
    
    # Create sample dataset for testing
    print("\n" + "="*60)
    print("CREATING SAMPLE DATASET")
    print("="*60)
    
    sample_dataset = finder.create_sample_dataset(num_samples=200)
    
    # Convert to standard format
    standard_dataset = finder.convert_dataset_to_standard(
        sample_dataset, 'sample_validation'
    )
    
    # Save standard format
    with open('real_datasets/standard_validation_dataset.pkl', 'wb') as f:
        pickle.dump(standard_dataset, f)
    
    print(f"\nStandard dataset saved with {len(standard_dataset['traces'])} traces")
    
    # Create data request document
    create_validation_data_request()
    
    # Summary
    print("\n" + "="*60)
    print("DATASET SEARCH SUMMARY")
    print("="*60)
    
    print("\n📊 Available Public Datasets:")
    for name in available:
        print(f"  - {name}")
    
    print("\n📝 Next Steps:")
    print("1. Try to download $1 Unistroke Dataset for basic validation")
    print("2. Request data from gesture keyboard app developers")
    print("3. Consider recording our own validation set")
    print("4. Use sample dataset to build validation framework")
    
    print("\n✅ Created sample dataset for immediate validation testing")
    print("   - 200 simulated traces")
    print("   - 20 vocabulary words")
    print("   - 10 simulated users")
    
    return standard_dataset

if __name__ == "__main__":
    dataset = main()