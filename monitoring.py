#!/usr/bin/env python
"""
Monitoring and logging system for gesture generation
"""

import json
import time
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
import threading
from collections import deque, Counter

@dataclass
class MetricEvent:
    """Single metric event"""
    timestamp: float
    event_type: str
    value: Any
    metadata: Optional[Dict] = None

class MetricsCollector:
    """
    Collects and aggregates metrics
    """
    
    def __init__(self, window_size: int = 1000):
        self.events = deque(maxlen=window_size)
        self.counters = Counter()
        self.timings = {}
        self.lock = threading.Lock()
        
        # Configure logging
        self.logger = logging.getLogger('metrics')
        handler = logging.FileHandler('metrics.log')
        handler.setFormatter(
            logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        )
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)
    
    def record_event(self, event_type: str, value: Any = 1, metadata: Optional[Dict] = None):
        """Record a metric event"""
        with self.lock:
            event = MetricEvent(
                timestamp=time.time(),
                event_type=event_type,
                value=value,
                metadata=metadata
            )
            self.events.append(event)
            self.counters[event_type] += 1
            
            # Log significant events
            if event_type in ['error', 'warning', 'generation_failed']:
                self.logger.warning(f"{event_type}: {value} - {metadata}")
    
    def start_timing(self, operation: str) -> str:
        """Start timing an operation"""
        timer_id = f"{operation}_{time.time()}"
        self.timings[timer_id] = time.time()
        return timer_id
    
    def end_timing(self, timer_id: str) -> float:
        """End timing and record duration"""
        if timer_id not in self.timings:
            return 0.0
        
        duration = time.time() - self.timings[timer_id]
        del self.timings[timer_id]
        
        # Extract operation name
        operation = timer_id.rsplit('_', 1)[0]
        self.record_event(f"{operation}_duration", duration)
        
        return duration
    
    def get_summary(self, last_minutes: int = 5) -> Dict:
        """Get metrics summary for the last N minutes"""
        with self.lock:
            cutoff_time = time.time() - (last_minutes * 60)
            recent_events = [e for e in self.events if e.timestamp > cutoff_time]
            
            if not recent_events:
                return {
                    'period_minutes': last_minutes,
                    'total_events': 0,
                    'event_types': {},
                    'average_durations': {}
                }
            
            # Group by event type
            event_groups = {}
            durations = {}
            
            for event in recent_events:
                if event.event_type not in event_groups:
                    event_groups[event.event_type] = []
                event_groups[event.event_type].append(event.value)
                
                # Track durations
                if event.event_type.endswith('_duration'):
                    op_name = event.event_type.replace('_duration', '')
                    if op_name not in durations:
                        durations[op_name] = []
                    durations[op_name].append(event.value)
            
            # Calculate averages
            avg_durations = {
                op: sum(times) / len(times)
                for op, times in durations.items()
            }
            
            return {
                'period_minutes': last_minutes,
                'total_events': len(recent_events),
                'event_types': {k: len(v) for k, v in event_groups.items()},
                'average_durations': avg_durations,
                'error_count': sum(1 for e in recent_events if 'error' in e.event_type),
                'success_rate': self._calculate_success_rate(recent_events)
            }
    
    def _calculate_success_rate(self, events: List[MetricEvent]) -> float:
        """Calculate success rate from events"""
        successes = sum(1 for e in events if e.event_type == 'generation_success')
        failures = sum(1 for e in events if e.event_type == 'generation_failed')
        total = successes + failures
        
        return (successes / total * 100) if total > 0 else 100.0
    
    def export_metrics(self, filepath: str):
        """Export metrics to JSON file"""
        with self.lock:
            data = {
                'exported_at': datetime.now().isoformat(),
                'total_events': len(self.events),
                'counters': dict(self.counters),
                'recent_events': [asdict(e) for e in list(self.events)[-100:]],
                'summary': self.get_summary()
            }
            
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
            
            self.logger.info(f"Exported metrics to {filepath}")


class PerformanceMonitor:
    """
    Monitor system performance
    """
    
    def __init__(self):
        self.metrics = MetricsCollector()
        self.thresholds = {
            'generation_time': 1.0,  # seconds
            'batch_size': 100,
            'memory_usage': 1024 * 1024 * 500,  # 500MB
            'error_rate': 0.05  # 5%
        }
    
    def monitor_generation(self, func):
        """Decorator to monitor trajectory generation"""
        def wrapper(*args, **kwargs):
            timer_id = self.metrics.start_timing('generation')
            
            try:
                result = func(*args, **kwargs)
                duration = self.metrics.end_timing(timer_id)
                
                # Check performance threshold
                if duration > self.thresholds['generation_time']:
                    self.metrics.record_event(
                        'slow_generation',
                        duration,
                        {'word': kwargs.get('word', 'unknown')}
                    )
                
                self.metrics.record_event('generation_success')
                return result
                
            except Exception as e:
                self.metrics.end_timing(timer_id)
                self.metrics.record_event(
                    'generation_failed',
                    str(e),
                    {'word': kwargs.get('word', 'unknown')}
                )
                raise
        
        return wrapper
    
    def monitor_batch(self, func):
        """Decorator to monitor batch operations"""
        def wrapper(*args, **kwargs):
            timer_id = self.metrics.start_timing('batch_processing')
            batch_size = len(kwargs.get('words', []))
            
            try:
                # Check batch size
                if batch_size > self.thresholds['batch_size']:
                    self.metrics.record_event(
                        'large_batch',
                        batch_size
                    )
                
                result = func(*args, **kwargs)
                duration = self.metrics.end_timing(timer_id)
                
                # Calculate throughput
                throughput = batch_size / duration if duration > 0 else 0
                self.metrics.record_event('batch_throughput', throughput)
                
                return result
                
            except Exception as e:
                self.metrics.end_timing(timer_id)
                self.metrics.record_event('batch_failed', str(e))
                raise
        
        return wrapper
    
    def check_health(self) -> Dict:
        """Check system health"""
        summary = self.metrics.get_summary(last_minutes=5)
        
        health_status = 'healthy'
        issues = []
        
        # Check error rate
        error_rate = (100 - summary['success_rate']) / 100
        if error_rate > self.thresholds['error_rate']:
            health_status = 'degraded'
            issues.append(f"High error rate: {error_rate:.1%}")
        
        # Check average generation time
        avg_gen_time = summary['average_durations'].get('generation', 0)
        if avg_gen_time > self.thresholds['generation_time']:
            health_status = 'degraded'
            issues.append(f"Slow generation: {avg_gen_time:.2f}s avg")
        
        # Check for recent errors
        if summary['error_count'] > 10:
            health_status = 'unhealthy'
            issues.append(f"Multiple errors: {summary['error_count']}")
        
        return {
            'status': health_status,
            'issues': issues,
            'metrics_summary': summary,
            'checked_at': datetime.now().isoformat()
        }
    
    def start_background_monitoring(self, interval: int = 60):
        """Start background health monitoring"""
        def monitor_loop():
            while True:
                time.sleep(interval)
                health = self.check_health()
                
                if health['status'] != 'healthy':
                    self.metrics.logger.warning(f"Health check: {health}")
                
                # Export metrics periodically
                if time.time() % 3600 < interval:  # Every hour
                    timestamp = datetime.now().strftime("%Y%m%d_%H")
                    self.metrics.export_metrics(f"metrics_{timestamp}.json")
        
        thread = threading.Thread(target=monitor_loop, daemon=True)
        thread.start()


# Global monitor instance
monitor = PerformanceMonitor()


def example_usage():
    """Example of using the monitoring system"""
    
    # Start background monitoring
    monitor.start_background_monitoring()
    
    # Simulate some operations
    @monitor.monitor_generation
    def generate_trace(word: str):
        time.sleep(0.1)  # Simulate work
        return f"trace_for_{word}"
    
    @monitor.monitor_batch
    def process_batch(words: List[str]):
        results = []
        for word in words:
            results.append(generate_trace(word=word))
        return results
    
    # Run some operations
    print("Running monitored operations...")
    
    for i in range(10):
        try:
            result = generate_trace(word=f"test_{i}")
            print(f"Generated: {result}")
        except Exception as e:
            print(f"Error: {e}")
    
    # Process batch
    batch_result = process_batch(words=['hello', 'world', 'test'])
    print(f"Batch processed: {len(batch_result)} items")
    
    # Check health
    health = monitor.check_health()
    print("\nHealth Check:")
    print(json.dumps(health, indent=2))
    
    # Export metrics
    monitor.metrics.export_metrics("metrics_example.json")
    print("\nMetrics exported to metrics_example.json")


if __name__ == "__main__":
    example_usage()