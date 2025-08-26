#!/usr/bin/env python
"""
Generate synthetic traces for ALL missing words from dictionary
Processes in batches to manage memory
"""

import json
import time
from pathlib import Path
from generate_missing_words_dataset import MissingWordsGenerator

def generate_all_missing_words(batch_size: int = 500, samples_per_word: int = 2):
    """Generate traces for all missing words in batches"""
    
    print("="*60)
    print("GENERATING ALL MISSING WORDS DATASET")
    print("="*60)
    
    generator = MissingWordsGenerator()
    
    # Load real words and dictionary
    real_words = generator.load_real_words()
    dict_words = generator.load_dictionary()
    
    # Find ALL missing words
    all_missing = generator.find_missing_words()
    total_words = len(all_missing)
    
    print(f"\nTotal missing words to generate: {total_words}")
    print(f"Batch size: {batch_size} words")
    print(f"Samples per word: {samples_per_word}")
    print(f"Expected total traces: {total_words * samples_per_word}")
    
    # Process in batches
    all_sessions = {}
    session_counter = 0
    
    for batch_start in range(0, total_words, batch_size):
        batch_end = min(batch_start + batch_size, total_words)
        batch_words = all_missing[batch_start:batch_end]
        
        print(f"\n{'='*40}")
        print(f"Processing batch {batch_start//batch_size + 1}/{(total_words + batch_size - 1)//batch_size}")
        print(f"Words {batch_start + 1} to {batch_end} of {total_words}")
        print(f"{'='*40}")
        
        # Generate traces for this batch
        batch_sessions = generator.generate_synthetic_traces(
            words=batch_words,
            samples_per_word=samples_per_word
        )
        
        # Renumber session IDs to be continuous
        for old_key in list(batch_sessions.keys()):
            new_key = f"synth_{session_counter:08d}"
            all_sessions[new_key] = batch_sessions[old_key]
            session_counter += 1
        
        print(f"✅ Batch complete. Total sessions so far: {len(all_sessions)}")
        
        # Save periodically to avoid data loss
        if (batch_start // batch_size + 1) % 5 == 0:
            print("Saving checkpoint...")
            save_sessions(all_sessions, f"datasets/synthetic_all/checkpoint_{batch_start}")
    
    # Save final dataset
    print("\n" + "="*60)
    print("SAVING FINAL DATASET")
    print("="*60)
    
    output_dir = Path("datasets/synthetic_all")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Saving {len(all_sessions)} sessions to {output_dir}")
    
    # Save in chunks to avoid memory issues
    chunk_size = 1000
    for i in range(0, len(all_sessions), chunk_size):
        chunk_keys = list(all_sessions.keys())[i:i + chunk_size]
        for session_id in chunk_keys:
            data = all_sessions[session_id]
            
            # Save metadata as JSON
            json_file = output_dir / f"{session_id}.json"
            with open(json_file, 'w') as f:
                json.dump(data['metadata'], f)
            
            # Save trace log
            log_file = output_dir / f"{session_id}.log"
            with open(log_file, 'w') as f:
                f.write('\n'.join(data['log']))
        
        print(f"  Saved chunk {i//chunk_size + 1}/{(len(all_sessions) + chunk_size - 1)//chunk_size}")
    
    # Create summary
    summary = {
        'total_sessions': len(all_sessions),
        'total_words': total_words,
        'samples_per_word': samples_per_word,
        'timestamp': time.time(),
        'generator': 'RealCalibratedGenerator',
        'missing_words_count': total_words,
        'real_words_count': len(real_words),
        'dictionary_size': len(dict_words)
    }
    
    with open(output_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("\n" + "="*60)
    print("COMPLETE DATASET GENERATION FINISHED")
    print("="*60)
    print(f"Output directory: {output_dir}")
    print(f"Total sessions: {len(all_sessions)}")
    print(f"Total unique words: {total_words}")
    print(f"Dataset ready for training!")
    
    return all_sessions

def save_sessions(sessions, checkpoint_name):
    """Save sessions to checkpoint"""
    checkpoint_dir = Path(checkpoint_name)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    with open(checkpoint_dir / 'checkpoint.json', 'w') as f:
        json.dump({'num_sessions': len(sessions)}, f)
    
    print(f"  Checkpoint saved: {len(sessions)} sessions")

if __name__ == "__main__":
    # Generate with smaller batches to avoid memory issues
    sessions = generate_all_missing_words(
        batch_size=500,  # Process 500 words at a time
        samples_per_word=1  # 1 sample per word for now (can increase later)
    )