"""
Advanced routing examples for HybridVectorDB Phase 3.
"""

import numpy as np
import time
from hybridvectordb import HybridVectorDB, Config
from hybridvectordb.router import RoutingDecision
from hybridvectordb.profiler import profiler, optimizer


def main():
    """Demonstrate advanced routing functionality."""
    
    print("=== HybridVectorDB Phase 3 Advanced Routing Examples ===\n")
    
    # Example 1: Basic Advanced Routing
    print("--- Example 1: Basic Advanced Routing ---")
    config = Config(
        dimension=128,
        index_type="flat",
        metric_type="l2",
        use_gpu=True,
        batch_threshold=32,
        k_threshold=50
    )
    
    db = HybridVectorDB(config)
    print(f"Database: {db}")
    print(f"Routing strategy: {db.router.current_strategy}")
    
    # Generate data
    np.random.seed(42)
    vectors = np.random.random((1000, 128)).astype(np.float32)
    metadata = [{"source": f"doc_{i}"} for i in range(1000)]
    
    # Add vectors (will use advanced routing)
    print("Adding 1000 vectors...")
    db.add(vectors, metadata=metadata)
    
    # Test different query patterns
    print("\nTesting different query patterns:")
    
    # Small query (should use CPU)
    small_query = np.random.random((1, 128)).astype(np.float32)
    results1 = db.search(small_query, k=10)
    print(f"Small query (1x10): {results1.search_time_ms:.2f}ms ({results1.index_used})")
    
    # Medium batch (might use GPU)
    medium_batch = np.random.random((25, 128)).astype(np.float32)
    results2 = db.search(medium_batch, k=25)
    avg_time = sum(r.search_time_ms for r in results2) / len(results2)
    print(f"Medium batch (25x25): {avg_time:.2f}ms avg ({results2[0].index_used})")
    
    # Large batch (should use GPU)
    large_batch = np.random.random((100, 128)).astype(np.float32)
    results3 = db.search(large_batch, k=50)
    avg_time = sum(r.search_time_ms for r in results3) / len(results3)
    print(f"Large batch (100x50): {avg_time:.2f}ms avg ({results3[0].index_used})")
    
    print()
    
    # Example 2: Routing Strategy Comparison
    print("--- Example 2: Routing Strategy Comparison ---")
    strategies = ['performance_based', 'workload_based', 'resource_based', 'hybrid']
    
    for strategy in strategies:
        print(f"\nTesting strategy: {strategy}")
        db.set_routing_strategy(strategy)
        
        # Run some queries to collect data
        for i in range(10):
            query = np.random.random((20, 128)).astype(np.float32)
            db.search(query, k=30)
        
        # Get routing stats
        stats = db.get_routing_stats()
        print(f"  Strategy: {stats['strategy']}")
        print(f"  Total queries: {stats['total_queries']}")
        print(f"  GPU usage: {stats['gpu_usage_percent']:.1f}%")
        print(f"  Speedup: {stats['speedup']:.2f}x")
        print(f"  Adaptive thresholds: batch={stats['adaptive_thresholds']['batch_threshold']}, k={stats['adaptive_thresholds']['k_threshold']}")
    
    print()
    
    # Example 3: Performance Profiling
    print("--- Example 3: Performance Profiling ---")
    db.set_routing_strategy('hybrid')
    
    # Profile different operations
    with profiler.profile("small_batch_search", {"batch_size": 5}):
        query = np.random.random((5, 128)).astype(np.float32)
        db.search(query, k=10)
    
    with profiler.profile("large_batch_search", {"batch_size": 100}):
        query = np.random.random((100, 128)).astype(np.float32)
        db.search(query, k=50)
    
    # Get profiling stats
    perf_stats = db.get_performance_stats()
    print("Performance Statistics:")
    for op_name, stats in perf_stats.get('operations', {}).items():
        print(f"  {op_name}:")
        print(f"    Count: {stats['count']}")
        print(f"    Avg time: {stats['avg_duration_ms']:.2f}ms")
        print(f"    Min time: {stats['min_duration_ms']:.2f}ms")
        print(f"    Max time: {stats['max_duration_ms']:.2f}ms")
    
    print()
    
    # Example 4: Performance Optimization
    print("--- Example 4: Performance Optimization ---")
    
    # Optimize batch size
    print("Optimizing batch size...")
    batch_optimization = db.optimize_performance("batch_size")
    if 'recommendation' in batch_optimization:
        print(f"Batch size recommendation: {batch_optimization['recommendation']}")
        print(f"Optimal batch size: {batch_optimization['optimal_batch_size']}")
        print(f"Optimal throughput: {batch_optimization['optimal_throughput_vps']:.1f} vectors/sec")
    
    # Optimize k value
    print("\nOptimizing k value...")
    k_optimization = db.optimize_performance("k_value")
    if 'recommendation' in k_optimization:
        print(f"K-value recommendation: {k_optimization['recommendation']}")
    
    print()
    
    # Example 5: Adaptive Learning
    print("--- Example 5: Adaptive Learning ---")
    db.set_routing_strategy('performance_based')
    
    print("Running queries to enable adaptive learning...")
    initial_thresholds = {
        'batch': db.router.adaptive_batch_threshold,
        'k': db.router.adaptive_k_threshold
    }
    
    # Simulate workload with varying performance
    for i in range(30):
        batch_size = 10 + i * 3  # Varying batch sizes
        query = np.random.random((batch_size, 128)).astype(np.float32)
        db.search(query, k=20)
    
    final_thresholds = {
        'batch': db.router.adaptive_batch_threshold,
        'k': db.router.adaptive_k_threshold
    }
    
    print(f"Initial thresholds - Batch: {initial_thresholds['batch']}, K: {initial_thresholds['k']}")
    print(f"Final thresholds - Batch: {final_thresholds['batch']}, K: {final_thresholds['k']}")
    
    if final_thresholds['batch'] != initial_thresholds['batch']:
        print("✓ Adaptive thresholds adjusted based on performance")
    else:
        print("- Thresholds unchanged (insufficient data for adaptation)")
    
    print()
    
    # Example 6: Advanced Analytics
    print("--- Example 6: Advanced Analytics ---")
    routing_stats = db.get_routing_stats()
    
    print("Comprehensive Routing Analytics:")
    print(f"  Total operations: {routing_stats['total_queries']}")
    print(f"  CPU operations: {routing_stats['cpu_queries']} ({routing_stats['cpu_usage_percent']:.1f}%)")
    print(f"  GPU operations: {routing_stats['gpu_queries']} ({routing_stats['gpu_usage_percent']:.1f}%)")
    print(f"  Average CPU time: {routing_stats['avg_cpu_time_ms']:.2f}ms")
    print(f"  Average GPU time: {routing_stats['avg_gpu_time_ms']:.2f}ms")
    print(f"  GPU speedup: {routing_stats['speedup']:.2f}x")
    print(f"  CPU success rate: {routing_stats['cpu_success_rate']:.2f}")
    print(f"  GPU success rate: {routing_stats['gpu_success_rate']:.2f}")
    
    if 'decision_distribution' in routing_stats:
        print("  Decision distribution:")
        for decision, count in routing_stats['decision_distribution'].items():
            print(f"    {decision}: {count}")
    
    print()
    
    # Example 7: Export Performance Data
    print("--- Example 7: Export Performance Data ---")
    
    # Export to JSON
    db.export_performance_data("performance_data.json", "json")
    print("Performance data exported to performance_data.json")
    
    # Show final statistics
    final_stats = db.get_stats()
    print("\nFinal Database Statistics:")
    print(f"  Total vectors: {len(db)}")
    print(f"  Total queries: {final_stats['total_queries']}")
    print(f"  Average query time: {final_stats['avg_query_time_ms']:.2f}ms")
    print(f"  GPU usage percentage: {final_stats.get('gpu_usage_percent', 0):.1f}%")
    
    print("\n=== Advanced Routing Examples Complete ===")


if __name__ == "__main__":
    main()
