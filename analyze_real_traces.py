#!/usr/bin/env python
"""
Analyze real swipe trace data and compare with synthetic generation
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
import glob
from collections import Counter, defaultdict

class RealTraceAnalyzer:
    """Analyze real swipe traces from dataset"""
    
    def __init__(self, dataset_path: str = "datasets/swipelogs"):
        self.dataset_path = Path(dataset_path)
        self.metadata = {}
        self.traces = {}
        self.statistics = defaultdict(list)
        
    def load_dataset(self, limit: int = None):
        """Load trace data from log files"""
        print(f"Loading traces from {self.dataset_path}")
        
        json_files = sorted(self.dataset_path.glob("*.json"))[:limit]
        log_files = sorted(self.dataset_path.glob("*.log"))[:limit]
        
        print(f"Found {len(json_files)} trace sessions")
        
        loaded = 0
        for json_file in json_files:
            session_id = json_file.stem
            log_file = self.dataset_path / f"{session_id}.log"
            
            if not log_file.exists():
                continue
                
            # Load metadata
            with open(json_file) as f:
                self.metadata[session_id] = json.load(f)
            
            # Load traces - handle variable whitespace
            try:
                traces_df = pd.read_csv(log_file, sep='\s+', engine='python')
                self.traces[session_id] = traces_df
            except Exception as e:
                print(f"  Warning: Could not load {log_file.name}: {e}")
                continue
            loaded += 1
            
            if loaded % 100 == 0:
                print(f"  Loaded {loaded} sessions...")
        
        print(f"✅ Loaded {loaded} trace sessions")
        return loaded
    
    def extract_word_traces(self) -> Dict:
        """Extract individual word traces from sessions"""
        word_traces = defaultdict(list)
        
        for session_id, df in self.traces.items():
            # Group by word
            grouped = df.groupby('word')
            
            for word, group in grouped:
                # Skip errors
                if group['is_err'].sum() > len(group) * 0.3:  # Skip if >30% errors
                    continue
                
                # Extract trace points
                trace_points = []
                for _, row in group.iterrows():
                    if row['event'] in ['touchmove', 'touchstart', 'touchend']:
                        trace_points.append([
                            row['x_pos'],
                            row['y_pos'],
                            row['timestamp']
                        ])
                
                if len(trace_points) > 5:  # Need meaningful trace
                    word_traces[word].append({
                        'session': session_id,
                        'trace': np.array(trace_points),
                        'metadata': self.metadata[session_id]
                    })
        
        return dict(word_traces)
    
    def analyze_trace_characteristics(self, word_traces: Dict) -> Dict:
        """Analyze characteristics of real traces"""
        print("\n" + "="*60)
        print("REAL TRACE CHARACTERISTICS")
        print("="*60)
        
        stats = {
            'num_words': len(word_traces),
            'total_traces': sum(len(v) for v in word_traces.values()),
            'word_stats': {},
            'global_stats': {
                'path_lengths': [],
                'durations': [],
                'num_points': [],
                'velocities': [],
                'accelerations': []
            }
        }
        
        # Analyze each word
        for word, traces in word_traces.items():
            if not traces:
                continue
                
            word_stat = {
                'count': len(traces),
                'avg_points': 0,
                'avg_duration': 0,
                'avg_path_length': 0,
                'avg_velocity': 0
            }
            
            points_list = []
            durations = []
            path_lengths = []
            velocities = []
            
            for trace_data in traces:
                trace = trace_data['trace']
                
                # Basic metrics
                points_list.append(len(trace))
                
                # Duration (ms to seconds)
                duration = (trace[-1, 2] - trace[0, 2]) / 1000.0
                durations.append(duration)
                
                # Path length
                path_len = 0
                for i in range(1, len(trace)):
                    dx = trace[i, 0] - trace[i-1, 0]
                    dy = trace[i, 1] - trace[i-1, 1]
                    path_len += np.sqrt(dx*dx + dy*dy)
                path_lengths.append(path_len)
                
                # Average velocity
                if duration > 0:
                    avg_vel = path_len / duration
                    velocities.append(avg_vel)
                
                # Add to global stats
                stats['global_stats']['path_lengths'].append(path_len)
                stats['global_stats']['durations'].append(duration)
                stats['global_stats']['num_points'].append(len(trace))
                if duration > 0:
                    stats['global_stats']['velocities'].append(avg_vel)
            
            # Calculate word averages
            if points_list:
                word_stat['avg_points'] = np.mean(points_list)
                word_stat['avg_duration'] = np.mean(durations)
                word_stat['avg_path_length'] = np.mean(path_lengths)
                word_stat['avg_velocity'] = np.mean(velocities) if velocities else 0
            
            stats['word_stats'][word] = word_stat
        
        # Calculate global statistics
        for key in ['path_lengths', 'durations', 'num_points', 'velocities']:
            if stats['global_stats'][key]:
                values = stats['global_stats'][key]
                stats['global_stats'][f'{key}_mean'] = np.mean(values)
                stats['global_stats'][f'{key}_std'] = np.std(values)
                stats['global_stats'][f'{key}_min'] = np.min(values)
                stats['global_stats'][f'{key}_max'] = np.max(values)
        
        return stats
    
    def print_analysis_summary(self, stats: Dict):
        """Print analysis summary"""
        print(f"\nTotal unique words: {stats['num_words']}")
        print(f"Total trace samples: {stats['total_traces']}")
        
        # Top words by frequency
        print("\nTop 10 most frequent words:")
        sorted_words = sorted(stats['word_stats'].items(), 
                            key=lambda x: x[1]['count'], 
                            reverse=True)[:10]
        
        for word, word_stat in sorted_words:
            print(f"  '{word}': {word_stat['count']} samples")
        
        # Global statistics
        print("\n" + "-"*40)
        print("GLOBAL STATISTICS")
        print("-"*40)
        
        gs = stats['global_stats']
        if 'path_lengths_mean' in gs:
            print(f"Path Length: {gs['path_lengths_mean']:.1f} ± {gs['path_lengths_std']:.1f} pixels")
            print(f"  Range: [{gs['path_lengths_min']:.1f}, {gs['path_lengths_max']:.1f}]")
        
        if 'durations_mean' in gs:
            print(f"Duration: {gs['durations_mean']:.2f} ± {gs['durations_std']:.2f} seconds")
            print(f"  Range: [{gs['durations_min']:.2f}, {gs['durations_max']:.2f}]")
        
        if 'num_points_mean' in gs:
            print(f"Points per trace: {gs['num_points_mean']:.1f} ± {gs['num_points_std']:.1f}")
            print(f"  Range: [{gs['num_points_min']:.0f}, {gs['num_points_max']:.0f}]")
        
        if 'velocities_mean' in gs:
            print(f"Velocity: {gs['velocities_mean']:.1f} ± {gs['velocities_std']:.1f} pixels/sec")
            print(f"  Range: [{gs['velocities_min']:.1f}, {gs['velocities_max']:.1f}]")
    
    def extract_user_demographics(self) -> Dict:
        """Analyze user demographics from metadata"""
        demographics = {
            'ages': [],
            'genders': Counter(),
            'nationalities': Counter(),
            'dominant_hands': Counter(),
            'swipe_hands': Counter(),
            'swipe_fingers': Counter(),
            'familiarity': Counter(),
            'english_levels': Counter(),
            'devices': []
        }
        
        for session_id, meta in self.metadata.items():
            demographics['ages'].append(meta.get('age', 0))
            demographics['genders'][meta.get('gender', 'Unknown')] += 1
            demographics['nationalities'][meta.get('nationality', 'Unknown')] += 1
            demographics['dominant_hands'][meta.get('dominantHand', 'Unknown')] += 1
            demographics['swipe_hands'][meta.get('swipeHand', 'Unknown')] += 1
            demographics['swipe_fingers'][meta.get('swipeFinger', 'Unknown')] += 1
            demographics['familiarity'][meta.get('familiarity', 'Unknown')] += 1
            demographics['english_levels'][meta.get('englishLevel', 'Unknown')] += 1
            
            demographics['devices'].append({
                'width': meta.get('screenWidth', 0),
                'height': meta.get('screenHeight', 0)
            })
        
        return demographics
    
    def save_processed_traces(self, word_traces: Dict, output_file: str):
        """Save processed traces for training"""
        print(f"\nSaving processed traces to {output_file}")
        
        # Convert to serializable format
        processed = {}
        for word, traces in word_traces.items():
            processed[word] = []
            for trace_data in traces:
                processed[word].append({
                    'session': trace_data['session'],
                    'trace': trace_data['trace'].tolist(),
                    'age': trace_data['metadata'].get('age'),
                    'gender': trace_data['metadata'].get('gender'),
                    'hand': trace_data['metadata'].get('swipeHand')
                })
        
        with open(output_file, 'w') as f:
            json.dump(processed, f, indent=2)
        
        print(f"✅ Saved {len(processed)} words with traces")


def compare_with_synthetic():
    """Compare real traces with our synthetic generation"""
    print("\n" + "="*60)
    print("REAL VS SYNTHETIC COMPARISON")
    print("="*60)
    
    from optimized_swype_generator import OptimizedSwypeGenerator
    
    # Load a few real traces for comparison
    analyzer = RealTraceAnalyzer()
    analyzer.load_dataset(limit=50)
    word_traces = analyzer.extract_word_traces()
    
    # Find common test words
    test_words = ['the', 'and', 'you', 'that', 'this']
    
    generator = OptimizedSwypeGenerator()
    
    comparison_results = []
    
    for word in test_words:
        if word not in word_traces:
            continue
        
        print(f"\nComparing word: '{word}'")
        
        # Get real traces for this word
        real_traces = word_traces[word]
        if not real_traces:
            continue
        
        # Generate synthetic trace
        synthetic = generator.generate_trajectory(word, style='natural')
        synth_trace = synthetic['trajectory']
        
        # Compare characteristics
        for real_data in real_traces[:3]:  # Compare with first 3 real traces
            real_trace = real_data['trace']
            
            # Duration comparison
            real_duration = (real_trace[-1, 2] - real_trace[0, 2]) / 1000.0
            synth_duration = synthetic['duration']
            
            # Path length comparison
            real_path = 0
            for i in range(1, len(real_trace)):
                dx = real_trace[i, 0] - real_trace[i-1, 0]
                dy = real_trace[i, 1] - real_trace[i-1, 1]
                real_path += np.sqrt(dx*dx + dy*dy)
            
            synth_path = 0
            for i in range(1, len(synth_trace)):
                dx = synth_trace[i, 0] - synth_trace[i-1, 0]
                dy = synth_trace[i, 1] - synth_trace[i-1, 1]
                synth_path += np.sqrt(dx*dx + dy*dy)
            
            print(f"  Real: {len(real_trace)} points, {real_duration:.2f}s, {real_path:.1f} path")
            print(f"  Synth: {len(synth_trace)} points, {synth_duration:.2f}s, {synth_path:.1f} path")
            
            comparison_results.append({
                'word': word,
                'real_points': len(real_trace),
                'synth_points': len(synth_trace),
                'real_duration': real_duration,
                'synth_duration': synth_duration,
                'real_path': real_path,
                'synth_path': synth_path
            })
    
    if comparison_results:
        # Calculate overall statistics
        print("\n" + "-"*40)
        print("OVERALL COMPARISON")
        print("-"*40)
        
        df = pd.DataFrame(comparison_results)
        
        print(f"Points ratio (synth/real): {df['synth_points'].mean() / df['real_points'].mean():.2f}")
        print(f"Duration ratio (synth/real): {df['synth_duration'].mean() / df['real_duration'].mean():.2f}")
        print(f"Path ratio (synth/real): {df['synth_path'].mean() / df['real_path'].mean():.2f}")
        
        if df['synth_duration'].mean() / df['real_duration'].mean() > 1.5:
            print("\n⚠️  Synthetic traces are too slow - need speed adjustment")
        elif df['synth_path'].mean() / df['real_path'].mean() < 0.5:
            print("\n⚠️  Synthetic paths are too short - need scaling adjustment")
        else:
            print("\n✅ Synthetic traces match real trace characteristics well!")


def main():
    """Main analysis function"""
    
    # Analyze real traces
    analyzer = RealTraceAnalyzer()
    
    # Load dataset (limit for quick testing)
    loaded = analyzer.load_dataset(limit=100)
    
    if loaded == 0:
        print("❌ No trace data found")
        return
    
    # Extract word traces
    word_traces = analyzer.extract_word_traces()
    
    # Analyze characteristics
    stats = analyzer.analyze_trace_characteristics(word_traces)
    analyzer.print_analysis_summary(stats)
    
    # Analyze demographics
    print("\n" + "="*60)
    print("USER DEMOGRAPHICS")
    print("="*60)
    
    demographics = analyzer.extract_user_demographics()
    
    print(f"Age range: {min(demographics['ages'])} - {max(demographics['ages'])}")
    print(f"Mean age: {np.mean(demographics['ages']):.1f}")
    
    print(f"\nGender distribution:")
    for gender, count in demographics['genders'].most_common():
        print(f"  {gender}: {count}")
    
    print(f"\nSwipe finger preference:")
    for finger, count in demographics['swipe_fingers'].most_common():
        print(f"  {finger}: {count}")
    
    print(f"\nFamiliarity levels:")
    for level, count in demographics['familiarity'].most_common():
        print(f"  {level}: {count}")
    
    # Save processed traces
    analyzer.save_processed_traces(word_traces, "real_traces_processed.json")
    
    # Compare with synthetic
    compare_with_synthetic()
    
    print("\n✅ Analysis complete!")
    

if __name__ == "__main__":
    main()