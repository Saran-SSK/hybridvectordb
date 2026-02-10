"""
Comprehensive benchmarking examples for HybridVectorDB Phase 5.
"""

import numpy as np
import time
import json
from hybridvectordb import HybridVectorDB, Config
from hybridvectordb._cpp import create_vector_database, CppConfig, BenchmarkFramework, BenchmarkConfig, LoadGenerator

def main():
    """Demonstrate comprehensive benchmarking capabilities."""
    
    print("=== HybridVectorDB Phase 5 Comprehensive Benchmarking Examples ===\n")
    
    # Example 1: Basic Benchmarking
    print("--- Example 1: Basic Performance Benchmarking ---")
    
    # Configure benchmark
    benchmark_config = BenchmarkConfig()
    benchmark_config.num_vectors = 10000
    benchmark_config.num_queries = 1000
    benchmark_config.vector_dimension = 128
    benchmark_config.k_values = [10, 50, 100]
    benchmark_config.batch_sizes = [1, 10, 50, 100]
    benchmark_config.benchmark_iterations = 3
    benchmark_config.enable_profiling = True
    benchmark_config.output_file = "basic_benchmark"
    benchmark_config.output_format = "json"
    
    # Create C++ database for benchmarking
    cpp_config = CppConfig()
    cpp_config.dimension = benchmark_config.vector_dimension
    cpp_config.index_type = "flat"
    cpp_config.metric_type = "l2"
    cpp_config.use_gpu = False
    
    db = create_vector_database(cpp_config)
    
    # Create and run benchmark framework
    framework = BenchmarkFramework(benchmark_config)
    framework.run_benchmark()
    framework.generate_report()
    framework.visualize_results()
    
    print("Basic benchmark completed!")
    print("Results exported to: basic_benchmark.json")
    print("Visualization exported to: basic_benchmark_visualization.html")
    
    print()
    
    # Example 2: Load Generation
    print("--- Example 2: Synthetic Data Generation ---")
    
    from hybridvectordb._cpp import LoadConfig
    
    # Configure load generation
    load_config = LoadConfig()
    load_config.num_vectors = 50000
    load_config.vector_dimension = 256
    load_config.distribution = "clustered"
    load_config.num_clusters = 20
    load_config.cluster_std = 0.2
    load_config.noise_ratio = 0.05
    load_config.normalize_vectors = True
    load_config.output_file = "synthetic_vectors"
    load_config.output_format = "numpy"
    
    # Generate synthetic data
    generator = LoadGenerator(load_config)
    vectors = generator.generate_vectors()
    
    print(f"Generated {len(vectors)} vectors with {load_config.distribution} distribution")
    print(f"Vector dimension: {load_config.vector_dimension}")
    print(f"Number of clusters: {load_config.num_clusters}")
    
    # Validate generated data
    if generator.validate_data(vectors):
        print("✓ Generated data validation passed")
    else:
        print("✗ Generated data validation failed")
    
    # Export data
    generator.export_data(vectors)
    print(f"Data exported to: {load_config.output_file}.npy")
    
    print()
    
    # Example 3: Performance Comparison
    print("--- Example 3: Implementation Comparison ---")
    
    # Test data
    test_vectors = np.random.random((5000, 128)).astype(np.float32)
    test_queries = np.random.random((500, 128)).astype(np.float32)
    
    # Python implementation
    python_config = Config(
        dimension=128,
        index_type="flat",
        metric_type="l2",
        use_gpu=False
    )
    python_db = HybridVectorDB(python_config)
    
    # C++ implementation
    cpp_config = CppConfig()
    cpp_config.dimension = 128
    cpp_config.index_type = "flat"
    cpp_config.metric_type = "l2"
    cpp_config.use_gpu = False
    
    cpp_db = create_vector_database(cpp_config)
    
    # Benchmark Python
    print("Benchmarking Python implementation...")
    start_time = time.time()
    
    for vec in test_vectors:
        python_db.add(vec)
    
    python_times = []
    for query in test_queries:
        start = time.time()
        results = python_db.search(query, k=10)
        end = time.time()
        python_times.append((end - start) * 1000)  # Convert to ms
    
    python_total_time = time.time() - start_time
    
    # Benchmark C++
    print("Benchmarking C++ implementation...")
    start_time = time.time()
    
    # Convert to C++ format
    from hybridvectordb._cpp import VectorData
    
    for vec in test_vectors:
        vd = VectorData("test", vec.tolist())
        cpp_db.add_vectors(&vd, 1)
    
    cpp_times = []
    for query in test_queries:
        start = time.time()
        results = cpp_db.search_vectors(query, 1, 10, False)
        end = time.time()
        cpp_times.append((end - start) * 1000)  # Convert to ms
    
    cpp_total_time = time.time() - start_time
    
    # Calculate performance comparison
    python_avg_time = np.mean(python_times)
    cpp_avg_time = np.mean(cpp_times)
    speedup = python_avg_time / cpp_avg_time if cpp_avg_time > 0 else 1.0
    
    print(f"Python average search time: {python_avg_time:.2f}ms")
    print(f"C++ average search time: {cpp_avg_time:.2f}ms")
    print(f"C++ speedup: {speedup:.2f}x")
    
    # Export comparison results
    comparison_results = {
        "python": {
            "total_time_ms": python_total_time * 1000,
            "avg_search_time_ms": python_avg_time,
            "throughput_qps": len(test_queries) / python_total_time
        },
        "cpp": {
            "total_time_ms": cpp_total_time * 1000,
            "avg_search_time_ms": cpp_avg_time,
            "throughput_qps": len(test_queries) / cpp_total_time
        },
        "comparison": {
            "speedup": speedup,
            "python_faster": python_avg_time < cpp_avg_time,
            "cpp_faster": cpp_avg_time < python_avg_time
        }
    }
    
    with open("performance_comparison.json", "w") as f:
        json.dump(comparison_results, f, indent=2)
    
    print(f"Performance comparison exported to: performance_comparison.json")
    
    print()
    
    # Example 4: Automated Testing
    print("--- Example 4: Automated Testing Suite ---")
    
    from hybridvectordb._cpp import AutomatedTestSuite
    
    # Create test suite
    test_suite = AutomatedTestSuite()
    
    # Add custom tests
    test_suite.add_test("Custom Performance Test", lambda: test_custom_performance(cpp_db))
    test_suite.add_test("Custom Memory Test", lambda: test_memory_efficiency(cpp_db))
    test_suite.add_test("Custom Accuracy Test", lambda: test_search_accuracy(cpp_db))
    
    # Run tests
    test_suite.run_tests()
    
    # Run stress tests
    print("\nRunning stress tests...")
    test_suite.run_stress_tests()
    
    # Get test results
    results = test_suite.get_results()
    coverage = test_suite.get_coverage()
    
    print(f"Tests passed: {sum(results)}/{len(results)}")
    print(f"Test coverage: {coverage:.1f}%")
    
    print()
    
    # Example 5: Performance Dashboard
    print("--- Example 5: Real-time Performance Monitoring ---")
    
    # Create monitoring configuration
    monitor_config = BenchmarkConfig()
    monitor_config.num_vectors = 1000
    monitor_config.num_queries = 100
    monitor_config.vector_dimension = 128
    monitor_config.enable_profiling = True
    monitor_config.benchmark_iterations = 1
    
    # Create monitoring database
    monitor_db = create_vector_database(cpp_config)
    
    # Simulate real-time monitoring
    print("Starting real-time performance monitoring...")
    print("Press Ctrl+C to stop monitoring")
    
    try:
        iteration = 0
        while True:
            # Generate random workload
            vectors = np.random.random((100, 128)).astype(np.float32)
            queries = np.random.random((10, 128)).astype(np.float32)
            
            # Add vectors
            start_time = time.time()
            for vec in vectors:
                vd = VectorData(f"monitor_{iteration}", vec.tolist())
                monitor_db.add_vectors(&vd, 1)
            
            # Search queries
            search_times = []
            for query in queries:
                start = time.time()
                results = monitor_db.search_vectors(query, 1, 10, False)
                end = time.time()
                search_times.append((end - start) * 1000)
            
            # Get metrics
            metrics = monitor_db.get_metrics()
            
            print(f"Iteration {iteration + 1}:")
            print(f"  Add time: {time.time() - start_time:.3f}s")
            print(f"  Avg search time: {np.mean(search_times):.2f}ms")
            print(f"  Throughput: {len(queries) / (time.time() - start_time):.1f} q/s")
            print(f"  Memory usage: {metrics.memory_usage_bytes / 1024 / 1024:.1f}MB")
            print(f"  Total queries: {metrics.total_queries}")
            print(f"  C++ speedup: {metrics.get_speedup():.2f}x")
            
            iteration += 1
            time.sleep(1)  # Monitor every second
            
    except KeyboardInterrupt:
        print("\nMonitoring stopped by user")
    
    print()
    
    # Example 6: Comprehensive Analysis
    print("--- Example 6: Comprehensive Performance Analysis ---")
    
    # Load previous benchmark results
    try:
        with open("basic_benchmark.json", "r") as f:
            benchmark_data = json.load(f)
        
        print("Analyzing benchmark results...")
        
        # Extract metrics
        add_times = [r["add_time_ms"] for r in benchmark_data["results"]]
        search_times = [r["search_time_ms"] for r in benchmark_data["results"]]
        throughputs = [r["overall_throughput_ops"] for r in benchmark_data["results"]]
        
        # Calculate statistics
        print(f"Add time statistics:")
        print(f"  Mean: {np.mean(add_times):.2f}ms")
        print(f"  Std: {np.std(add_times):.2f}ms")
        print(f"  Min: {np.min(add_times):.2f}ms")
        print(f"  Max: {np.max(add_times):.2f}ms")
        print(f"  95th percentile: {np.percentile(add_times, 95):.2f}ms")
        
        print(f"\nSearch time statistics:")
        print(f"  Mean: {np.mean(search_times):.2f}ms")
        print(f"  Std: {np.std(search_times):.2f}ms")
        print(f"  Min: {np.min(search_times):.2f}ms")
        print(f"  Max: {np.max(search_times):.2f}ms")
        print(f"  95th percentile: {np.percentile(search_times, 95):.2f}ms")
        
        print(f"\nThroughput statistics:")
        print(f"  Mean: {np.mean(throughputs):.1f} ops/s")
        print(f"  Std: {np.std(throughputs):.1f} ops/s")
        print(f"  Min: {np.min(throughputs):.1f} ops/s")
        print(f"  Max: {np.max(throughputs):.1f} ops/s")
        
        # Performance analysis
        print(f"\nPerformance Analysis:")
        if np.mean(search_times) > 100:
            print("  ⚠️  High search latency detected")
            print("  Recommendation: Consider using GPU or IVF index")
        
        if np.mean(throughputs) < 1000:
            print("  ⚠️  Low throughput detected")
            print("  Recommendation: Consider increasing batch size")
        
        # Calculate performance score (0-100)
        latency_score = max(0, 100 - np.mean(search_times) / 2)
        throughput_score = min(100, np.mean(throughputs) / 100)
        overall_score = (latency_score + throughput_score) / 2
        
        print(f"\nPerformance Score: {overall_score:.1f}/100")
        
    except FileNotFoundError:
        print("No benchmark results found. Run basic benchmark first.")
    
    print("\n=== Comprehensive Benchmarking Examples Complete ===")
    print("Phase 5: Comprehensive benchmarking, load generation, and visualization implemented!")


def test_custom_performance(db):
    """Custom performance test."""
    vectors = np.random.random((1000, 128)).astype(np.float32)
    
    start_time = time.time()
    for vec in vectors:
        from hybridvectordb._cpp import VectorData
        vd = VectorData("perf_test", vec.tolist())
        db.add_vectors(&vd, 1)
    
    end_time = time.time()
    return (end_time - start_time) < 5.0  # Should complete in under 5 seconds


def test_memory_efficiency(db):
    """Test memory efficiency."""
    initial_memory = db.get_memory_usage()
    
    # Add vectors and check memory growth
    vectors = np.random.random((10000, 128)).astype(np.float32)
    from hybridvectordb._cpp import VectorData
    
    for vec in vectors:
        vd = VectorData("mem_test", vec.tolist())
        db.add_vectors(&vd, 1)
    
    final_memory = db.get_memory_usage()
    
    # Memory should grow proportionally
    initial_used = int(initial_memory.get("memory_pool_used", 0))
    final_used = int(final_memory.get("memory_pool_used", 0))
    expected_growth = len(vectors) * 128 * 4  # bytes per float
    actual_growth = final_used - initial_used
    
    # Memory efficiency should be reasonable
    efficiency = actual_growth / expected_growth if expected_growth > 0 else 1.0
    
    return efficiency < 3.0  # Less than 3x overhead


def test_search_accuracy(db):
    """Test search accuracy."""
    # Create known data
    vectors = np.random.random((1000, 128)).astype(np.float32)
    from hybridvectordb._cpp import VectorData
    
    for i, vec in enumerate(vectors):
        vd = VectorData(f"accuracy_{i}", vec.tolist())
        db.add_vectors(&vd, 1)
    
    # Search for exact matches
    correct = 0
    for i, vec in enumerate(vectors[:100]):  # Test first 100
        query = vec.reshape(1, -1)
        results = db.search_vectors(query, 1, 10, False)
        
        if not results.empty() and results[0].results[0].id == f"accuracy_{i}":
            correct += 1
    
    return correct / 100  # Should be 100% accuracy


if __name__ == "__main__":
    main()
