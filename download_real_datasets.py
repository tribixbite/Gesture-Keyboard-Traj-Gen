#!/usr/bin/env python
"""
Download and parse real gesture datasets for validation
Focus on accessible public datasets
"""

import os
import json
import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
import pickle
import zipfile
import requests
from typing import Dict, List, Optional
import subprocess

class RealDatasetDownloader:
    """
    Download and parse real gesture datasets
    """
    
    def __init__(self):
        self.datasets_dir = Path("real_datasets")
        self.datasets_dir.mkdir(exist_ok=True)
        self.parsed_dir = self.datasets_dir / "parsed"
        self.parsed_dir.mkdir(exist_ok=True)
    
    def download_shark2_dataset(self) -> bool:
        """
        Attempt to download SHARK2 dataset from GitHub
        """
        print("="*60)
        print("DOWNLOADING SHARK2 DATASET")
        print("="*60)
        
        shark2_url = "https://github.com/aaronlaursen/SHARK2"
        shark2_dir = self.datasets_dir / "SHARK2"
        
        if shark2_dir.exists():
            print(f"SHARK2 already exists at {shark2_dir}")
            return True
        
        try:
            # Clone the repository
            print(f"Cloning SHARK2 repository...")
            result = subprocess.run(
                ["git", "clone", shark2_url, str(shark2_dir)],
                capture_output=True,
                text=True,
                timeout=60
            )
            
            if result.returncode == 0:
                print("✅ SHARK2 dataset downloaded successfully")
                return True
            else:
                print(f"❌ Failed to clone: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            print("❌ Download timeout")
            return False
        except Exception as e:
            print(f"❌ Error downloading SHARK2: {e}")
            return False
    
    def parse_shark2_traces(self) -> Optional[Dict]:
        """
        Parse SHARK2 gesture traces
        """
        shark2_dir = self.datasets_dir / "SHARK2"
        
        if not shark2_dir.exists():
            print("SHARK2 dataset not found")
            return None
        
        print("\nParsing SHARK2 traces...")
        
        # Look for trace files
        trace_files = []
        for ext in ['*.xml', '*.json', '*.txt']:
            trace_files.extend(shark2_dir.rglob(ext))
        
        if not trace_files:
            print("No trace files found in SHARK2 directory")
            # Look for any data files
            all_files = list(shark2_dir.rglob("*"))
            print(f"Found {len(all_files)} files total")
            
            # Sample first few files to understand structure
            for f in all_files[:10]:
                if f.is_file():
                    print(f"  - {f.relative_to(shark2_dir)}")
            return None
        
        parsed_traces = []
        
        for trace_file in trace_files[:100]:  # Parse first 100 files
            try:
                if trace_file.suffix == '.xml':
                    trace = self._parse_xml_trace(trace_file)
                elif trace_file.suffix == '.json':
                    trace = self._parse_json_trace(trace_file)
                else:
                    continue
                
                if trace:
                    parsed_traces.append(trace)
                    
            except Exception as e:
                print(f"Error parsing {trace_file.name}: {e}")
        
        if parsed_traces:
            print(f"✅ Parsed {len(parsed_traces)} traces")
            
            dataset = {
                'traces': parsed_traces,
                'source': 'SHARK2',
                'num_traces': len(parsed_traces)
            }
            
            # Save parsed dataset
            output_file = self.parsed_dir / "shark2_parsed.pkl"
            with open(output_file, 'wb') as f:
                pickle.dump(dataset, f)
            
            print(f"Saved to {output_file}")
            return dataset
        else:
            print("No traces successfully parsed")
            return None
    
    def _parse_xml_trace(self, xml_file: Path) -> Optional[Dict]:
        """Parse XML gesture trace"""
        try:
            tree = ET.parse(xml_file)
            root = tree.getroot()
            
            # Look for point data
            points = []
            for point in root.findall('.//point'):
                x = float(point.get('x', 0))
                y = float(point.get('y', 0))
                t = float(point.get('t', 0))
                points.append([x, y, t])
            
            if points:
                return {
                    'word': xml_file.stem,
                    'trace': np.array(points),
                    'source_file': str(xml_file.name)
                }
        except:
            return None
        
        return None
    
    def _parse_json_trace(self, json_file: Path) -> Optional[Dict]:
        """Parse JSON gesture trace"""
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            # Try different JSON structures
            if 'points' in data:
                points = data['points']
            elif 'trace' in data:
                points = data['trace']
            elif isinstance(data, list):
                points = data
            else:
                return None
            
            if points:
                return {
                    'word': data.get('word', json_file.stem),
                    'trace': np.array(points),
                    'source_file': str(json_file.name)
                }
        except:
            return None
        
        return None
    
    def download_dollar_dataset(self) -> bool:
        """
        Download $1 Unistroke Recognizer dataset
        """
        print("\n" + "="*60)
        print("DOWNLOADING $1 UNISTROKE DATASET")
        print("="*60)
        
        dataset_url = "https://depts.washington.edu/acelab/proj/dollar/xml.zip"
        output_file = self.datasets_dir / "dollar_xml.zip"
        extract_dir = self.datasets_dir / "dollar_dataset"
        
        if extract_dir.exists():
            print(f"$1 dataset already exists at {extract_dir}")
            return True
        
        try:
            print(f"Downloading from {dataset_url}...")
            
            # Download the zip file
            response = requests.get(dataset_url, timeout=30)
            
            if response.status_code == 200:
                with open(output_file, 'wb') as f:
                    f.write(response.content)
                print(f"✅ Downloaded {len(response.content)/1024:.1f} KB")
                
                # Extract zip file
                print("Extracting...")
                with zipfile.ZipFile(output_file, 'r') as zip_ref:
                    zip_ref.extractall(extract_dir)
                
                print(f"✅ Extracted to {extract_dir}")
                
                # Remove zip file
                output_file.unlink()
                
                return True
            else:
                print(f"❌ Failed to download: HTTP {response.status_code}")
                return False
                
        except requests.Timeout:
            print("❌ Download timeout")
            return False
        except Exception as e:
            print(f"❌ Error downloading: {e}")
            return False
    
    def parse_dollar_dataset(self) -> Optional[Dict]:
        """
        Parse $1 dataset XML files
        """
        dollar_dir = self.datasets_dir / "dollar_dataset"
        
        if not dollar_dir.exists():
            print("$1 dataset not found")
            return None
        
        print("\nParsing $1 Unistroke traces...")
        
        xml_files = list(dollar_dir.rglob("*.xml"))
        print(f"Found {len(xml_files)} XML files")
        
        parsed_gestures = []
        
        for xml_file in xml_files[:200]:  # Parse first 200
            try:
                tree = ET.parse(xml_file)
                root = tree.getroot()
                
                # Get gesture name
                gesture_name = root.get('Name', xml_file.stem)
                
                # Parse points
                points = []
                for point in root.findall('.//Point'):
                    x = float(point.get('X', 0))
                    y = float(point.get('Y', 0))
                    t = float(point.get('T', len(points) * 0.01))  # Estimate time
                    points.append([x, y, t])
                
                if points:
                    parsed_gestures.append({
                        'word': gesture_name,
                        'trace': np.array(points),
                        'source': 'dollar',
                        'file': xml_file.name
                    })
                    
            except Exception as e:
                print(f"Error parsing {xml_file.name}: {e}")
        
        if parsed_gestures:
            print(f"✅ Parsed {len(parsed_gestures)} gestures")
            
            # Group by gesture type
            gesture_types = {}
            for g in parsed_gestures:
                name = g['word'].split('-')[0] if '-' in g['word'] else g['word']
                if name not in gesture_types:
                    gesture_types[name] = []
                gesture_types[name].append(g)
            
            print(f"Gesture types: {list(gesture_types.keys())}")
            
            dataset = {
                'gestures': parsed_gestures,
                'types': gesture_types,
                'source': '$1_unistroke',
                'num_gestures': len(parsed_gestures),
                'num_types': len(gesture_types)
            }
            
            # Save parsed dataset
            output_file = self.parsed_dir / "dollar_parsed.pkl"
            with open(output_file, 'wb') as f:
                pickle.dump(dataset, f)
            
            print(f"Saved to {output_file}")
            return dataset
        else:
            print("No gestures parsed")
            return None
    
    def create_word_dataset_from_gestures(self, gesture_dataset: Dict) -> Dict:
        """
        Convert single gestures to word-like sequences
        """
        print("\nConverting gestures to word dataset...")
        
        # Map gestures to letters (simplified)
        gesture_to_letter = {
            'arrow': 'a',
            'caret': 'c',
            'circle': 'o',
            'check': 'v',
            'delete': 'd',
            'left_square_bracket': 'l',
            'right_square_bracket': 'r',
            'rectangle': 'n',
            'star': 's',
            'triangle': 't',
            'v': 'v',
            'x': 'x',
            'pigtail': 'p',
            'left_curly_brace': 'b',
            'right_curly_brace': 'b',
            'question_mark': 'q'
        }
        
        # Create simple word traces by combining gestures
        word_traces = []
        
        # Generate some simple "words" from available gestures
        if 'types' in gesture_dataset:
            available = list(gesture_dataset['types'].keys())
            
            # Create single letter "words"
            for gesture_type, samples in gesture_dataset['types'].items():
                if gesture_type in gesture_to_letter:
                    letter = gesture_to_letter[gesture_type]
                    for sample in samples[:5]:  # Use first 5 samples
                        word_traces.append({
                            'word': letter,
                            'trace': sample['trace'],
                            'source': 'dollar_converted'
                        })
        
        print(f"Created {len(word_traces)} word traces")
        
        return {
            'traces': [w['trace'] for w in word_traces],
            'words': [w['word'] for w in word_traces],
            'source': 'dollar_converted',
            'num_samples': len(word_traces)
        }


def main():
    """
    Main function to download and process real datasets
    """
    downloader = RealDatasetDownloader()
    
    success_count = 0
    
    # Try to download SHARK2
    if downloader.download_shark2_dataset():
        shark2_data = downloader.parse_shark2_traces()
        if shark2_data:
            success_count += 1
    
    # Try to download $1 dataset
    if downloader.download_dollar_dataset():
        dollar_data = downloader.parse_dollar_dataset()
        if dollar_data:
            # Convert to word dataset
            word_dataset = downloader.create_word_dataset_from_gestures(dollar_data)
            
            # Save as validation dataset
            output_file = downloader.parsed_dir / "dollar_word_dataset.pkl"
            with open(output_file, 'wb') as f:
                pickle.dump(word_dataset, f)
            
            print(f"Created word dataset: {output_file}")
            success_count += 1
    
    # Summary
    print("\n" + "="*60)
    print("DATASET DOWNLOAD SUMMARY")
    print("="*60)
    
    if success_count > 0:
        print(f"✅ Successfully processed {success_count} dataset(s)")
        print("\nNext steps:")
        print("1. Run validation with real data")
        print("2. Compare DTW distances")
        print("3. Tune generators based on results")
    else:
        print("❌ No datasets successfully downloaded")
        print("\nFallback options:")
        print("1. Create simple collection tool")
        print("2. Use simulated data for now")
        print("3. Contact researchers for access")
    
    return success_count > 0

if __name__ == "__main__":
    success = main()