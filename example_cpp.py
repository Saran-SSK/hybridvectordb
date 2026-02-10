"""
C++ bindings examples for HybridVectorDB Phase 4.
"""

import numpy as np
import time
from hybridvectordb import HybridVectorDB, Config
from hybridvectordb._cpp import create_vector_database, Config as CppConfig


def main():
    """Demonstrate C++ bindings with zero-copy optimizations."""
    
    print("=== HybridVectorDB Phase 4 C++ Bindings Examples ===\n")
    
    # Example 1: Basic C++ Usage
    print("--- Example 1: Basic C++ Usage ---")
    cpp_config = CppConfig()
    cpp_config.dimension = 128
    cpp_config.index_type = "flat"
    cpp_config.metric_type = "l2"
    cpp_config.use_gpu = False
    cpp_config.batch_threshold = 32
    cpp_config.k_threshold = 50
    
    # Create C++ database
    db = create_vector_database(cpp_config)
    print(f"Created C++ database: {type(db)}")
    
    # Generate test data
    np.random.seed(42)
    vectors = np.random.random((1000, 128)).astype(np.float32)
    
    # Add vectors using C++ bindings
    from hybridvectordb._cpp import add_vectors_numpy
    ids = [f"vec_{i}" for i in range(1000)]
    
    start_time = time.time()
    added = add_vectors_numpy(db, vectors, ids, [])
    add_time = time.time() - start_time
    
    print(f"Added {added} vectors in {add_time:.4f}s")
    print(f"Add throughput: {added / add_time:.1f} vectors/sec")
    
    # Search using C++ bindings
    from hybridvectordb._cpp import search_vectors_numpy
    
    queries = np.random.random((10, 128)).astype(np.float32)
    start_time = time.time()
    results = search_vectors_numpy(db, queries, 10, False)
    search_time = time.time() - start_time
    
    print(f"Searched {len(results)} queries in {search_time:.4f}s")
    print(f"Search throughput: {len(results) / search_time:.1f} queries/sec")
    
    # Get C++ metrics
    metrics = db.get_metrics()
    print(f"C++ Metrics - Total queries: {metrics.total_queries}")
    print(f"C++ Metrics - CPU queries: {metrics.cpu_queries}")
    print(f"C++ Metrics - Avg CPU time: {metrics.avg_cpu_time_ms:.2f}ms")
    print(f"C++ Metrics - Speedup: {metrics.get_speedup():.2f}x")
    
    print()
    
    # Example 2: Zero-Copy Operations
    print("--- Example 2: Zero-Copy Operations ---")
    
    # Zero-copy vector access
    count = 0
    vectors_ptr = db.get_vectors_zero_copy(count)
    print(f"Zero-copy access to {count} vectors")
    print(f"Vector pointer: {vectors_ptr}")
    
    if vectors_ptr and count > 0:
        # Access first vector without copying
        first_vector = np.frombuffer(
            np.ascontiguousarray(vectors_ptr[:128], dtype=np.float32),
            dtype=np.float32
        )
        print(f"First vector (zero-copy): {first_vector[:5]}")
    
    print()
    
    # Example 3: Performance Optimization
    print("--- Example 3: Performance Optimization ---")
    
    # Optimize batch size
    optimization = db.optimize_performance("batch_size")
    if "optimal_batch_size" in optimization:
        print(f"Optimal batch size: {optimization['optimal_batch_size']}")
        print(f"Optimal throughput: {optimization['throughput_vps']:.1f} vectors/sec")
    
    # Optimize memory layout
    mem_optimization = db.optimize_performance("memory_layout")
    if "memory_optimized" in mem_optimization:
        print(f"Memory layout optimized: {mem_optimization['memory_optimized']}")
    
    print()
    
    # Example 4: Memory Usage Analysis
    print("--- Example 4: Memory Usage Analysis ---")
    
    memory_usage = db.get_memory_usage()
    print("Memory Usage:")
    for key, value in memory_usage.items():
        if isinstance(value, int):
            print(f"  {key}: {value:,} bytes ({value/1024/1024:.1f} MB)")
        else:
            print(f"  {key}: {value}")
    
    print()
    
    # Example 5: Benchmarking
    print("--- Example 5: C++ Benchmarking ---")
    
    benchmark_results = db.benchmark(5000, 1000, 20)
    print("Benchmark Results:")
    for key, value in benchmark_results.items():
        if isinstance(value, float):
            if "time" in key:
                print(f"  {key}: {value:.4f}ms")
            elif "throughput" in key:
                print(f"  {key}: {value:.1f}")
            else:
                print(f"  {key}: {value:.2f}")
        else:
            print(f"  {key}: {value}")
    
    print()
    
    # Example 6: Performance Comparison (Python vs C++)
    print("--- Example 6: Performance Comparison ---")
    
    # Python implementation
    python_config = Config(
        dimension=128,
        index_type="flat",
        metric_type="l2",
        use_gpu=False
    )
    python_db = HybridVectorDB(python_config)
    
    # Test data
    test_vectors = np.random.random((2000, 128)).astype(np.float32)
    test_queries = np.random.random((100, 128)).astype(np.float32)
    
    # Python performance
    start_time = time.time()
    python_db.add(test_vectors)
    python_results = python_db.search(test_queries, k=10)
    python_time = time.time() - start_time
    
    # C++ performance
    start_time = time.time()
    add_vectors_numpy(db, test_vectors, [], [])
    cpp_results = search_vectors_numpy(db, test_queries, 10, False)
    cpp_time = time.time() - start_time
    
    print(f"Python implementation: {python_time:.4f}s")
    print(f"C++ implementation: {cpp_time:.4f}s")
    
    if cpp_time > 0:
        speedup = python_time / cpp_time
        print(f"C++ speedup: {speedup:.2f}x")
    
    print()
    
    # Example 7: Advanced SIMD Optimizations
    print("--- Example 7: Advanced SIMD Optimizations ---")
    
    from hybridvectordb._cpp import optimization
    
    # Test SIMD optimizations
    test_data = np.random.random((1000, 128)).astype(np.float32)
    
    # Apply SIMD optimizations
    optimization.apply_simd_optimizations(
        test_data, 
        test_data.shape[0], 
        test_data.shape[1]
    )
    print("Applied SIMD optimizations to test data")
    
    # Optimize memory layout
    optimization.optimize_memory_layout(
        test_data,
        test_data.shape[0],
        test_data.shape[1]
    )
    print("Optimized memory layout for cache efficiency")
    
    print()
    
    # Example 8: Error Handling and Recovery
    print("--- Example 8: Error Handling and Recovery ---")
    
    try:
        # Test invalid operations
        from hybridvectordb._cpp import HybridVectorDBError
        
        # Invalid vector addition
        invalid_vectors = np.random.random((10, 64)).astype(np.float32)  # Wrong dimension
        add_vectors_numpy(db, invalid_vectors, [], [])
        
    except Exception as e:
        print(f"Caught expected error: {type(e).__name__}: {e}")
    
    print()
    
    # Example 9: Configuration Management
    print("--- Example 9: Configuration Management ---")
    
    # Update configuration
    new_config = CppConfig()
    new_config.dimension = 128
    new_config.index_type = "flat"
    new_config.metric_type = "l2"
    new_config.use_gpu = False
    new_config.batch_threshold = 64  # Changed
    new_config.k_threshold = 100  # Changed
    
    db.configure(new_config)
    print("Updated database configuration")
    
    # Test with new configuration
    new_queries = np.random.random((50, 128)).astype(np.float32)
    new_results = search_vectors_numpy(db, new_queries, 20, False)
    print(f"Search with new config: {len(new_results)} results")
    
    print()
    
    # Example 10: Metrics and Analytics
    print("--- Example 10: Metrics and Analytics ---")
    
    # Get comprehensive statistics
    stats = db.get_stats()
    print("Comprehensive Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Get performance metrics
    final_metrics = db.get_metrics()
    print("\nFinal Performance Metrics:")
    print(f"  Total queries: {final_metrics.total_queries}")
    print(f"  CPU queries: {final_metrics.cpu_queries}")
    print(f"  GPU queries: {final_metrics.gpu_queries}")
    print(f"  CPU success rate: {final_metrics.cpu_success_rate:.2%}")
    print(f"  GPU success rate: {final_metrics.gpu_success_rate:.2%}")
    print(f"  Overall speedup: {final_metrics.get_speedup():.2f}x")
    
    print("\n=== C++ Bindings Examples Complete ===")
    print("Phase 4: C++ bindings with zero-copy optimizations implemented!")


if __name__ == "__main__":
    main()
