#!/usr/bin/env python3
"""
Generate 100 diverse swype traces for testing and analysis
"""

import json
import csv
import random
import numpy as np
from datetime import datetime
from pathlib import Path
from enhanced_swype_generator import EnhancedSwypeGenerator

# Common words list for realistic typing scenarios
COMMON_WORDS = [
    # High frequency words
    "the", "and", "for", "are", "but", "not", "you", "all", "can", "her",
    "was", "one", "our", "out", "his", "has", "had", "how", "man", "say",
    
    # Medium frequency words  
    "time", "make", "good", "very", "look", "come", "work", "want", "give", "most",
    "find", "tell", "call", "hand", "keep", "high", "life", "point", "place", "over",
    
    # Longer words
    "people", "think", "great", "little", "world", "after", "should", "never", "these", "other",
    "about", "there", "their", "would", "could", "first", "water", "which", "where", "right",
    
    # Complex words
    "through", "together", "important", "children", "different", "country", "problem", "against", "become", "between",
    "government", "understand", "everyone", "remember", "business", "program", "question", "interest", "possible", "education",
    
    # Technical words
    "computer", "software", "internet", "database", "network", "security", "develop", "system", "process", "service"
]

def generate_diverse_parameters():
    """Generate diverse but realistic parameter combinations"""
    # Map styles to precision and speed combinations
    style = random.choice(['precise', 'fast', 'natural', 'sloppy'])
    
    if style == 'precise':
        user_speed = np.random.uniform(0.5, 0.8)
        precision = np.random.uniform(0.8, 0.95)
    elif style == 'fast':
        user_speed = np.random.uniform(1.5, 2.0)
        precision = np.random.uniform(0.5, 0.7)
    elif style == 'sloppy':
        user_speed = np.random.uniform(1.2, 1.8)
        precision = np.random.uniform(0.4, 0.6)
    else:  # natural
        user_speed = np.random.uniform(0.8, 1.2)
        precision = np.random.uniform(0.6, 0.8)
    
    return {
        'user_speed': user_speed,
        'precision': precision,
        'velocity_profile': random.choice(['bell', 'uniform', 'realistic']),
        'style_label': style  # Store style as metadata label
    }

def analyze_trace(trace_data):
    """Analyze a single trace for quality metrics"""
    # Handle the actual format from enhanced_swype_generator
    if 'trajectory' not in trace_data:
        return None
        
    trajectory = trace_data['trajectory']
    if len(trajectory) < 2:
        return None
    
    # trajectory is numpy array with columns [x, y, t]
    # Calculate path statistics
    path_length = 0
    velocities = []
    accelerations = []
    jerks = []
    
    for i in range(1, len(trajectory)):
        dt = trajectory[i, 2] - trajectory[i-1, 2] if trajectory[i, 2] != trajectory[i-1, 2] else 0.001
        dx = trajectory[i, 0] - trajectory[i-1, 0] 
        dy = trajectory[i, 1] - trajectory[i-1, 1]
        
        dist = np.sqrt(dx**2 + dy**2)
        path_length += dist
        
        if dt > 0:
            vel = dist / dt
            velocities.append(vel)
            
            if i > 1 and len(velocities) > 1:
                acc = (velocities[-1] - velocities[-2]) / dt
                accelerations.append(acc)
                
                if i > 2 and len(accelerations) > 1:
                    jerk = (accelerations[-1] - accelerations[-2]) / dt
                    jerks.append(abs(jerk))
    
    # Calculate curvature
    curvatures = []
    for i in range(2, len(trajectory) - 2):
        p1 = np.array([trajectory[i-1, 0], trajectory[i-1, 1]])
        p2 = np.array([trajectory[i, 0], trajectory[i, 1]])
        p3 = np.array([trajectory[i+1, 0], trajectory[i+1, 1]])
        
        v1 = p2 - p1
        v2 = p3 - p2
        
        if np.linalg.norm(v1) > 0 and np.linalg.norm(v2) > 0:
            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            cos_angle = np.clip(cos_angle, -1, 1)
            angle = np.arccos(cos_angle)
            curvatures.append(angle)
    
    return {
        'word': trace_data['word'],
        'word_length': len(trace_data['word']),
        'num_points': len(trajectory),
        'duration': trace_data['duration'],
        'path_length': path_length,
        'path_per_char': path_length / len(trace_data['word']),
        'mean_velocity': np.mean(velocities) if velocities else 0,
        'max_velocity': np.max(velocities) if velocities else 0,
        'mean_acceleration': np.mean(np.abs(accelerations)) if accelerations else 0,
        'mean_jerk': np.mean(jerks) if jerks else 0,
        'max_jerk': np.max(jerks) if jerks else 0,
        'mean_curvature': np.mean(curvatures) if curvatures else 0,
        'style': trace_data.get('style', 'unknown'),
        'velocity_profile': trace_data.get('velocity_profile', 'unknown')
    }

def main():
    """Generate 100 diverse traces and analyze them"""
    
    print("Initializing Enhanced Swype Generator...")
    generator = EnhancedSwypeGenerator()
    
    # Create output directory
    output_dir = Path("generated_traces")
    output_dir.mkdir(exist_ok=True)
    
    # Storage for all traces and analyses
    all_traces = []
    all_analyses = []
    generation_log = []
    
    print(f"\nGenerating 100 diverse traces...")
    print("-" * 60)
    
    # Generate traces
    for i in range(100):
        # Select a word
        word = random.choice(COMMON_WORDS)
        
        # Generate parameters
        params = generate_diverse_parameters()
        
        try:
            # Set generator configuration for this trace
            generator.velocity_profile = params['velocity_profile']
            
            # Generate trace with supported parameters
            trace = generator.generate_trajectory(
                word=word,
                user_speed=params['user_speed'],
                precision=params['precision']
            )
            
            # Add generation metadata directly to trace dict
            trace['style'] = params['style_label']
            trace['user_speed'] = params['user_speed']
            trace['precision'] = params['precision']
            trace['velocity_profile'] = params['velocity_profile']
            trace['trace_id'] = i + 1
            
            # Convert numpy arrays to lists for JSON serialization
            trace_for_save = {
                'word': trace['word'],
                'trajectory': trace['trajectory'].tolist(),
                'velocity': trace['velocity'].tolist() if 'velocity' in trace else [],
                'acceleration': trace['acceleration'].tolist() if 'acceleration' in trace else [],
                'jerk': trace['jerk'].tolist() if 'jerk' in trace else [],
                'duration': trace['duration'],
                'sampling_rate': trace['sampling_rate'],
                'key_positions': trace['key_positions'].tolist(),
                'style': trace['style'],
                'velocity_profile': trace['velocity_profile'],
                'trace_id': trace['trace_id']
            }
            
            # Analyze trace
            analysis = analyze_trace(trace)
            
            if analysis:
                all_traces.append(trace_for_save)
                all_analyses.append(analysis)
                
                # Log generation
                log_entry = {
                    'id': i + 1,
                    'word': word,
                    'success': True,
                    'style': params['style_label'],
                    'quality_score': 1.0 - (analysis['mean_jerk'] / 1000000)  # Simple quality metric
                }
                generation_log.append(log_entry)
                
                if (i + 1) % 10 == 0:
                    print(f"Generated {i + 1}/100 traces...")
            else:
                print(f"Warning: Failed to analyze trace for '{word}'")
                
        except Exception as e:
            print(f"Error generating trace for '{word}': {e}")
            generation_log.append({
                'id': i + 1,
                'word': word,
                'success': False,
                'error': str(e)
            })
    
    print(f"\nSuccessfully generated {len(all_traces)}/100 traces")
    
    # Save all traces to file
    traces_file = output_dir / "all_traces.json"
    with open(traces_file, 'w') as f:
        json.dump(all_traces, f, indent=2)
    print(f"Saved traces to {traces_file}")
    
    # Save analyses to CSV
    if all_analyses:
        analysis_file = output_dir / "trace_analysis.csv"
        with open(analysis_file, 'w', newline='') as f:
            fieldnames = all_analyses[0].keys()
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_analyses)
        print(f"Saved analysis to {analysis_file}")
    
    # Generate quality review report
    print("\n" + "="*60)
    print("QUALITY REVIEW REPORT")
    print("="*60)
    
    if all_analyses:
        # Overall statistics
        print("\n1. GENERATION STATISTICS")
        print(f"   - Total traces generated: {len(all_traces)}")
        print(f"   - Success rate: {len(all_traces)/100:.1%}")
        print(f"   - Unique words used: {len(set(a['word'] for a in all_analyses))}")
        
        # Word complexity distribution
        word_lengths = [a['word_length'] for a in all_analyses]
        print(f"\n2. WORD COMPLEXITY")
        print(f"   - Average word length: {np.mean(word_lengths):.1f} characters")
        print(f"   - Shortest word: {min(word_lengths)} characters")
        print(f"   - Longest word: {max(word_lengths)} characters")
        
        # Path characteristics
        path_lengths = [a['path_length'] for a in all_analyses]
        path_per_chars = [a['path_per_char'] for a in all_analyses]
        print(f"\n3. PATH CHARACTERISTICS")
        print(f"   - Average path length: {np.mean(path_lengths):.1f} units")
        print(f"   - Path length per character: {np.mean(path_per_chars):.1f} units")
        print(f"   - Path length variation (std): {np.std(path_lengths):.1f}")
        
        # Velocity analysis
        mean_vels = [a['mean_velocity'] for a in all_analyses]
        max_vels = [a['max_velocity'] for a in all_analyses]
        print(f"\n4. VELOCITY PROFILES")
        print(f"   - Average velocity: {np.mean(mean_vels):.1f} units/s")
        print(f"   - Maximum velocity reached: {np.max(max_vels):.1f} units/s")
        print(f"   - Velocity range: {np.min(mean_vels):.1f} - {np.max(mean_vels):.1f} units/s")
        
        # Smoothness metrics
        mean_jerks = [a['mean_jerk'] for a in all_analyses]
        max_jerks = [a['max_jerk'] for a in all_analyses]
        print(f"\n5. SMOOTHNESS ANALYSIS")
        print(f"   - Average jerk: {np.mean(mean_jerks):.0f} units/s³")
        print(f"   - Maximum jerk: {np.max(max_jerks):.0f} units/s³")
        print(f"   - Smoothest trace jerk: {np.min(mean_jerks):.0f} units/s³")
        print(f"   - Roughest trace jerk: {np.max(mean_jerks):.0f} units/s³")
        
        # Style distribution
        styles = [a['style'] for a in all_analyses]
        style_counts = {s: styles.count(s) for s in set(styles)}
        print(f"\n6. STYLE DISTRIBUTION")
        for style, count in sorted(style_counts.items()):
            print(f"   - {style}: {count} traces ({count/len(styles):.1%})")
        
        # Velocity profile distribution
        profiles = [a['velocity_profile'] for a in all_analyses]
        profile_counts = {p: profiles.count(p) for p in set(profiles)}
        print(f"\n7. VELOCITY PROFILE DISTRIBUTION")
        for profile, count in sorted(profile_counts.items()):
            print(f"   - {profile}: {count} traces ({count/len(profiles):.1%})")
        
        # Quality assessment
        quality_scores = []
        for a in all_analyses:
            # Simple quality score based on smoothness and consistency
            jerk_score = max(0, 1 - (a['mean_jerk'] / 1000000))
            path_score = max(0, 1 - abs(a['path_per_char'] - 400) / 400)  # Ideal ~400 units/char
            quality = (jerk_score + path_score) / 2
            quality_scores.append(quality)
        
        print(f"\n8. QUALITY ASSESSMENT")
        print(f"   - Average quality score: {np.mean(quality_scores):.2f}/1.00")
        print(f"   - Best quality score: {np.max(quality_scores):.2f}/1.00")
        print(f"   - Worst quality score: {np.min(quality_scores):.2f}/1.00")
        print(f"   - High quality traces (>0.7): {sum(q > 0.7 for q in quality_scores)}")
        print(f"   - Medium quality traces (0.4-0.7): {sum(0.4 <= q <= 0.7 for q in quality_scores)}")
        print(f"   - Low quality traces (<0.4): {sum(q < 0.4 for q in quality_scores)}")
        
        # Identify best and worst traces
        best_idx = np.argmax(quality_scores)
        worst_idx = np.argmin(quality_scores)
        
        print(f"\n9. NOTABLE TRACES")
        print(f"   Best trace: #{all_analyses[best_idx]['word']} (score: {quality_scores[best_idx]:.3f})")
        print(f"   - Style: {all_analyses[best_idx]['style']}")
        print(f"   - Mean jerk: {all_analyses[best_idx]['mean_jerk']:.0f}")
        print(f"   - Path/char: {all_analyses[best_idx]['path_per_char']:.1f}")
        
        print(f"\n   Worst trace: #{all_analyses[worst_idx]['word']} (score: {quality_scores[worst_idx]:.3f})")
        print(f"   - Style: {all_analyses[worst_idx]['style']}")
        print(f"   - Mean jerk: {all_analyses[worst_idx]['mean_jerk']:.0f}")
        print(f"   - Path/char: {all_analyses[worst_idx]['path_per_char']:.1f}")
        
        # Recommendations
        print(f"\n10. RECOMMENDATIONS")
        if np.mean(quality_scores) > 0.6:
            print("   ✓ Generation quality is GOOD overall")
            print("   - Traces show realistic smoothness and path characteristics")
        else:
            print("   ⚠ Generation quality needs improvement")
            print("   - Consider tuning noise and smoothing parameters")
        
        if np.std(quality_scores) > 0.2:
            print("   ✓ Good diversity in generated traces")
        else:
            print("   ⚠ Low diversity - consider wider parameter ranges")
        
        # Style-specific analysis
        print("\n11. STYLE-SPECIFIC PERFORMANCE")
        for style in set(styles):
            style_analyses = [a for a in all_analyses if a['style'] == style]
            if style_analyses:
                style_jerks = [a['mean_jerk'] for a in style_analyses]
                print(f"   {style}:")
                print(f"     - Count: {len(style_analyses)}")
                print(f"     - Avg jerk: {np.mean(style_jerks):.0f} units/s³")
    
    # Save report
    report_file = output_dir / "quality_report.txt"
    with open(report_file, 'w') as f:
        f.write(f"TRACE GENERATION QUALITY REPORT\n")
        f.write(f"Generated: {datetime.now().isoformat()}\n")
        f.write(f"Total traces: {len(all_traces)}\n")
        f.write(f"Success rate: {len(all_traces)/100:.1%}\n\n")
        
        if all_analyses:
            f.write(f"Key Metrics:\n")
            f.write(f"- Avg quality score: {np.mean(quality_scores):.3f}\n")
            f.write(f"- Avg jerk: {np.mean(mean_jerks):.0f} units/s³\n")
            f.write(f"- Avg path/char: {np.mean(path_per_chars):.1f} units\n")
    
    print(f"\nReport saved to {report_file}")
    print("\n" + "="*60)
    print("GENERATION COMPLETE")
    print("="*60)
    
    return all_traces, all_analyses

if __name__ == "__main__":
    traces, analyses = main()