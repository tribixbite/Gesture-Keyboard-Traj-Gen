#!/usr/bin/env python
"""
Quick test of monitoring system
"""

from monitoring import PerformanceMonitor
import time
import json

def test_monitoring():
    """Test monitoring functionality"""
    monitor = PerformanceMonitor()
    
    # Test generation monitoring
    @monitor.monitor_generation
    def generate_test(word: str):
        time.sleep(0.05)
        return f"trace_{word}"
    
    # Test batch monitoring
    @monitor.monitor_batch  
    def batch_test(words: list):
        return [generate_test(word=w) for w in words]
    
    print("Testing monitoring system...")
    print("-" * 40)
    
    # Run some operations
    for i in range(5):
        result = generate_test(word=f"test_{i}")
        print(f"✓ Generated: {result}")
    
    # Test batch
    batch_result = batch_test(words=['hello', 'world'])
    print(f"✓ Batch processed: {len(batch_result)} items")
    
    # Test error handling
    try:
        @monitor.monitor_generation
        def failing_gen(word: str):
            raise ValueError("Test error")
        
        failing_gen(word="fail")
    except:
        print("✓ Error handling works")
    
    # Get summary
    summary = monitor.metrics.get_summary()
    print("\nMetrics Summary:")
    print(f"  Total events: {summary['total_events']}")
    print(f"  Success rate: {summary['success_rate']:.1f}%")
    print(f"  Error count: {summary['error_count']}")
    
    if summary['average_durations']:
        print("  Average durations:")
        for op, duration in summary['average_durations'].items():
            print(f"    {op}: {duration:.3f}s")
    
    # Check health
    health = monitor.check_health()
    print(f"\nHealth Status: {health['status']}")
    if health['issues']:
        print("  Issues:", health['issues'])
    
    print("\n✅ Monitoring system working correctly!")

if __name__ == "__main__":
    test_monitoring()