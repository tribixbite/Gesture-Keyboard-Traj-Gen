#!/usr/bin/env python
"""
Generate synthetic swipe traces for words in dictionary but not in real dataset
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
import time
import random
from typing import Dict, List, Set
from collections import defaultdict
from real_calibrated_generator import RealCalibratedGenerator

class MissingWordsGenerator:
    """Generate traces for words missing from real dataset"""
    
    def __init__(self):
        self.generator = RealCalibratedGenerator()
        self.real_words = set()
        self.dict_words = []
        
    def load_real_words(self, swipelogs_dir: str = "datasets/swipelogs") -> Set[str]:
        """Extract unique words from real swipelogs"""
        print("Loading words from real dataset...")
        
        log_files = Path(swipelogs_dir).glob("*.log")
        words = set()
        
        for log_file in list(log_files)[:100]:  # Process first 100 files
            try:
                df = pd.read_csv(log_file, sep='\s+', engine='python', on_bad_lines='skip')
                if 'word' in df.columns:
                    words.update(df['word'].unique())
            except Exception:
                continue
        
        self.real_words = {w.lower() for w in words if isinstance(w, str)}
        print(f"Found {len(self.real_words)} unique words in real dataset")
        return self.real_words
    
    def load_dictionary(self, dict_file: str = "datasets/en.txt") -> List[str]:
        """Load dictionary words with frequencies"""
        print("Loading dictionary...")
        
        words_freq = []
        with open(dict_file, 'r') as f:
            for line in f:
                if line.startswith('#') or not line.strip():
                    continue
                parts = line.strip().split('\t')
                if len(parts) == 2:
                    word, freq = parts
                    words_freq.append((word.lower(), int(freq)))
        
        # Sort by frequency (highest first)
        words_freq.sort(key=lambda x: x[1], reverse=True)
        self.dict_words = [w for w, f in words_freq]
        
        print(f"Loaded {len(self.dict_words)} words from dictionary")
        return self.dict_words
    
    def find_missing_words(self, max_words: int = None) -> List[str]:
        """Find high-frequency words not in real dataset"""
        
        # Get words in dictionary but not in real dataset
        missing = []
        for word in self.dict_words:
            if word not in self.real_words:
                # Filter to reasonable length words (2-10 chars)
                if 2 <= len(word) <= 10 and word.isalpha():
                    missing.append(word)
                    if max_words and len(missing) >= max_words:
                        break
        
        print(f"Found {len(missing)} missing words to generate")
        return missing
    
    def generate_synthetic_traces(self, words: List[str], samples_per_word: int = 5):
        """Generate synthetic traces in swipelog format"""
        
        print(f"\nGenerating synthetic traces for {len(words)} words...")
        print(f"Samples per word: {samples_per_word}")
        
        all_sessions = {}
        session_id = 0
        
        # Define user profiles for variation
        user_profiles = [
            {'age': 25, 'gender': 'Male', 'hand': 'Right', 'finger': 'Thumb', 'skill': 'expert'},
            {'age': 32, 'gender': 'Female', 'hand': 'Right', 'finger': 'Index', 'skill': 'intermediate'},
            {'age': 19, 'gender': 'Female', 'hand': 'Both', 'finger': 'Thumb', 'skill': 'novice'},
            {'age': 45, 'gender': 'Male', 'hand': 'Left', 'finger': 'Thumb', 'skill': 'intermediate'},
            {'age': 28, 'gender': 'Other', 'hand': 'Right', 'finger': 'Thumb', 'skill': 'expert'},
        ]
        
        styles = ['natural', 'fast', 'precise', 'slow']
        
        generated_count = 0
        
        for word_idx, word in enumerate(words):
            if word_idx % 100 == 0:
                print(f"  Processing word {word_idx}/{len(words)}...")
            
            for sample in range(samples_per_word):
                # Select random profile and style
                profile = random.choice(user_profiles)
                style = random.choice(styles)
                
                try:
                    # Generate trajectory
                    result = self.generator.generate_trajectory(word, style=style)
                    
                    # Create session ID
                    session_key = f"synth_{session_id:08d}"
                    
                    # Create metadata (JSON format)
                    metadata = {
                        'gender': profile['gender'],
                        'age': profile['age'],
                        'nationality': 'SYNTH',
                        'familiarity': 'Everyday' if profile['skill'] == 'expert' else 'Sometimes',
                        'englishLevel': 'Native',
                        'dominantHand': profile['hand'],
                        'swipeHand': profile['hand'],
                        'swipeFinger': profile['finger'],
                        'screenWidth': 360,
                        'screenHeight': 760,
                        'devicePixelRatio': 3,
                        'maxTouchPoints': 5,
                        'platform': 'Synthetic',
                        'vendor': 'Synthetic Generator',
                        'language': 'en-US',
                        'userAgent': 'Synthetic/1.0',
                        'timestamp': int(time.time() * 1000)
                    }
                    
                    # Create trace log (LOG format)
                    trace_log = []
                    
                    # Add header
                    trace_log.append("sentence timestamp keyb_width keyb_height event x_pos y_pos x_radius y_radius angle word is_err")
                    
                    # Generate touch events from trajectory
                    trajectory = result['trajectory']
                    base_timestamp = int(time.time() * 1000)
                    
                    # Sentence is word repeated 3 times (common pattern in dataset)
                    sentence = f"{word}_{word}_{word}"
                    
                    for i, point in enumerate(trajectory):
                        x, y, t = point
                        
                        # Convert to mobile screen coordinates
                        x_screen = x * 1.2 + 180  # Center on 360px width
                        y_screen = -y * 1.2 + 400  # Center on mobile keyboard area
                        
                        # Determine event type
                        if i == 0:
                            event = "touchstart"
                        elif i == len(trajectory) - 1:
                            event = "touchend"
                        else:
                            event = "touchmove"
                        
                        # Add some realistic touch radius variation
                        radius = 0.5 + random.random() * 0.3
                        
                        # Format: sentence timestamp keyb_width keyb_height event x_pos y_pos x_radius y_radius angle word is_err
                        log_line = f"{sentence} {base_timestamp + int(t)} 360 215 {event} {x_screen:.1f} {y_screen:.1f} {radius:.6f} {radius:.6f} 0 {word} 0"
                        trace_log.append(log_line)
                    
                    all_sessions[session_key] = {
                        'metadata': metadata,
                        'log': trace_log
                    }
                    
                    session_id += 1
                    generated_count += 1
                    
                except Exception as e:
                    print(f"    Error generating '{word}': {e}")
                    continue
        
        print(f"\n✅ Generated {generated_count} synthetic traces")
        return all_sessions
    
    def save_dataset(self, sessions: Dict, output_dir: str = "datasets/synthetic_swipelogs"):
        """Save synthetic dataset in swipelog format"""
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        print(f"\nSaving synthetic dataset to {output_dir}")
        
        for session_id, data in sessions.items():
            # Save metadata as JSON
            json_file = output_path / f"{session_id}.json"
            with open(json_file, 'w') as f:
                json.dump(data['metadata'], f)
            
            # Save trace log
            log_file = output_path / f"{session_id}.log"
            with open(log_file, 'w') as f:
                f.write('\n'.join(data['log']))
        
        print(f"✅ Saved {len(sessions)} sessions")
        
        # Create summary
        summary = {
            'total_sessions': len(sessions),
            'timestamp': time.time(),
            'generator': 'RealCalibratedGenerator',
            'words': len(set(s['log'][1].split()[9] for s in sessions.values() if len(s['log']) > 1))
        }
        
        with open(output_path / 'summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        return output_path


def main():
    """Generate missing words dataset"""
    
    print("="*60)
    print("GENERATING MISSING WORDS DATASET")
    print("="*60)
    
    generator = MissingWordsGenerator()
    
    # Load real words
    real_words = generator.load_real_words()
    
    # Load dictionary
    dict_words = generator.load_dictionary()
    
    # Find missing high-frequency words
    missing_words = generator.find_missing_words()  # Get ALL missing words
    
    print(f"\nTop 20 missing words to generate:")
    for word in missing_words[:20]:
        print(f"  - {word}")
    
    # Ask user for batch size
    batch_size = min(1000, len(missing_words))  # Process in batches to avoid memory issues
    print(f"\nGenerating traces for first {batch_size} words (out of {len(missing_words)} total)")
    
    # Generate synthetic traces
    sessions = generator.generate_synthetic_traces(
        words=missing_words[:batch_size],
        samples_per_word=2  # 2 samples per word to manage size
    )
    
    # Save dataset
    output_dir = generator.save_dataset(sessions)
    
    print("\n" + "="*60)
    print("DATASET GENERATION COMPLETE")
    print("="*60)
    print(f"Output directory: {output_dir}")
    print(f"Total sessions: {len(sessions)}")
    print(f"Ready for training!")
    
    return sessions


if __name__ == "__main__":
    sessions = main()