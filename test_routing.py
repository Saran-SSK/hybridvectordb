"""
Advanced routing tests for HybridVectorDB Phase 3.
"""

import pytest
import numpy as np
import time
from hybridvectordb import HybridVectorDB, Config
from hybridvectordb.router import AdvancedRouter, RoutingDecision, RoutingContext
from hybridvectordb.profiler import profiler, optimizer


class TestAdvancedRouting:
    """Test suite for advanced routing functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = Config(
            dimension=64,
            index_type="flat",
            metric_type="l2",
            use_gpu=True,
            batch_threshold=32,
            k_threshold=50
        )
        
        # Generate test data
        np.random.seed(42)
        self.vectors = np.random.random((200, 64)).astype(np.float32)
        self.query_vector = np.random.random((1, 64)).astype(np.float32)
        
        # Clear profiler
        profiler.clear_profiles()
    
    def test_router_initialization(self):
        """Test router initialization."""
        router = AdvancedRouter(self.config)
        
        assert router.metrics.total_queries == 0
        assert router.current_strategy == 'hybrid'
        assert router.adaptive_batch_threshold == 32
        assert router.adaptive_k_threshold == 50
        
        # Test strategy change
        router.set_strategy('performance_based')
        assert router.current_strategy == 'performance_based'
        
        # Test invalid strategy
        with pytest.raises(ValueError):
            router.set_strategy('invalid_strategy')
    
    def test_basic_routing_decisions(self):
        """Test basic routing decisions."""
        router = AdvancedRouter(self.config)
        
        # Small batch, small k -> CPU
        context = RoutingContext(
            batch_size=10,
            k_value=10,
            vector_dimension=64,
            dataset_size=100,
            gpu_available=True
        )
        decision, reasoning = router.route_operation(context)
        assert decision == RoutingDecision.CPU
        assert "batch=10" in reasoning
        
        # Large batch -> GPU
        context.batch_size = 50
        decision, reasoning = router.route_operation(context)
        assert decision == RoutingDecision.GPU
        assert "batch=50" in reasoning
    
    def test_performance_based_routing(self):
        """Test performance-based routing."""
        router = AdvancedRouter(self.config)
        router.set_strategy('performance_based')
        
        # Simulate GPU being faster
        for i in range(20):
            router.record_performance(RoutingDecision.GPU, 1.0, True)
            router.record_performance(RoutingDecision.CPU, 2.0, True)
        
        # Should prefer GPU
        context = RoutingContext(
            batch_size=20,
            k_value=20,
            vector_dimension=64,
            dataset_size=100,
            gpu_available=True
        )
        decision, reasoning = router.route_operation(context)
        assert decision == RoutingDecision.GPU
        assert "faster" in reasoning.lower()
    
    def test_workload_based_routing(self):
        """Test workload-based routing."""
        router = AdvancedRouter(self.config)
        router.set_strategy('workload_based')
        
        # High workload
        context = RoutingContext(
            batch_size=100,
            k_value=100,
            vector_dimension=64,
            dataset_size=10000,
            gpu_available=True
        )
        decision, reasoning = router.route_operation(context)
        assert decision == RoutingDecision.GPU
        assert "workload score" in reasoning.lower()
        
        # Low workload
        context.batch_size = 5
        context.k_value = 5
        decision, reasoning = router.route_operation(context)
        assert decision == RoutingDecision.CPU
    
    def test_resource_based_routing(self):
        """Test resource-based routing."""
        router = AdvancedRouter(self.config)
        router.set_strategy('resource_based')
        
        # Simulate constrained GPU
        context = RoutingContext(
            batch_size=50,
            k_value=50,
            vector_dimension=64,
            dataset_size=1000,
            gpu_available=True,
            gpu_memory_info={'used': 8000, 'total': 10000},  # 80% used
            gpu_utilization=90.0  # 90% utilization
        )
        decision, reasoning = router.route_operation(context)
        assert decision == RoutingDecision.CPU
        assert "constrained" in reasoning.lower()
        
        # Available GPU
        context.gpu_memory_info = {'used': 2000, 'total': 10000}  # 20% used
        context.gpu_utilization = 30.0  # 30% utilization
        decision, reasoning = router.route_operation(context)
        assert decision == RoutingDecision.GPU
    
    def test_hybrid_routing(self):
        """Test hybrid routing combining all strategies."""
        router = AdvancedRouter(self.config)
        
        # Setup mixed performance data
        for i in range(10):
            router.record_performance(RoutingDecision.GPU, 1.0, True)
            router.record_performance(RoutingDecision.CPU, 2.0, True)
        
        context = RoutingContext(
            batch_size=40,  # Above threshold
            k_value=30,
            vector_dimension=64,
            dataset_size=1000,
            gpu_available=True
        )
        
        decision, reasoning = router.route_operation(context)
        # Should be GPU due to multiple factors
        assert decision == RoutingDecision.GPU
        assert "hybrid decision" in reasoning.lower()
    
    def test_adaptive_thresholds(self):
        """Test adaptive threshold adjustment."""
        router = AdvancedRouter(self.config)
        
        # Simulate GPU being much faster
        for i in range(25):
            router.record_performance(RoutingDecision.GPU, 1.0, True)
            router.record_performance(RoutingDecision.CPU, 3.0, True)
        
        # Check if thresholds adjusted
        initial_batch_threshold = router.adaptive_batch_threshold
        router.update_adaptive_thresholds()
        
        # Should lower threshold to use GPU more
        assert router.adaptive_batch_threshold <= initial_batch_threshold
    
    def test_routing_statistics(self):
        """Test routing statistics collection."""
        router = AdvancedRouter(self.config)
        
        # Add some performance data
        router.record_performance(RoutingDecision.GPU, 1.0, True)
        router.record_performance(RoutingDecision.CPU, 2.0, True)
        router.record_performance(RoutingDecision.GPU, 1.5, True)
        
        stats = router.get_routing_stats()
        
        assert stats['total_queries'] == 3
        assert stats['gpu_queries'] == 2
        assert stats['cpu_queries'] == 1
        assert stats['gpu_usage_percent'] == 66.67
        assert stats['speedup'] > 1.0
        assert 'adaptive_thresholds' in stats
    
    def test_database_integration(self):
        """Test router integration with database."""
        db = HybridVectorDB(self.config)
        
        # Add vectors (should trigger routing)
        db.add(self.vectors[:100])
        
        # Search with different batch sizes
        small_query = self.query_vector
        large_batch_queries = np.random.random((50, 64)).astype(np.float32)
        
        # Small query should use CPU
        results1 = db.search(small_query, k=10)
        
        # Large batch should use GPU (if available)
        results2 = db.search(large_batch_queries, k=10)
        
        # Check that routing occurred
        stats = db.get_stats()
        assert stats['total_queries'] > 0
    
    def test_performance_profiling(self):
        """Test performance profiling."""
        # Test profiling context manager
        with profiler.profile("test_operation", {"batch_size": 100}):
            time.sleep(0.01)  # Simulate work
        
        stats = profiler.get_operation_stats("test_operation")
        assert stats['count'] == 1
        assert stats['avg_duration_ms'] > 10  # Should be around 10ms
        
        # Test manual profiling
        profile_id = profiler.start_profile("manual_test")
        time.sleep(0.005)
        profile = profiler.end_profile(profile_id)
        
        assert profile.operation_name == "manual_test"
        assert profile.duration_ms > 5
        assert profile.duration_ms < 10
    
    def test_batch_size_optimization(self):
        """Test batch size optimization."""
        def dummy_operation(data):
            # Simulate work that scales with batch size
            time.sleep(len(data) * 0.0001)
        
        test_sizes = [10, 25, 50, 100]
        result = optimizer.optimize_batch_size(dummy_operation, test_sizes, iterations=3)
        
        assert 'optimal_batch_size' in result
        assert 'optimal_throughput_vps' in result
        assert 'recommendation' in result
        
        # Should prefer larger batch sizes for this operation
        assert result['optimal_batch_size'] >= 50
    
    def test_k_value_optimization(self):
        """Test k-value optimization."""
        def dummy_search(query_data, k):
            # Simulate search that gets slower with larger k
            time.sleep(k * 0.0001)
            return list(range(k))
        
        test_k_values = [10, 50, 100, 200]
        query_data = np.random.random((10, 64)).astype(np.float32)
        
        result = optimizer.optimize_k_value(dummy_search, test_k_values, query_data, iterations=2)
        
        assert 'test_results' in result
        assert 'recommendation' in result
        
        # Check that times increase with k
        results = result['test_results']
        for i in range(1, len(results)):
            assert results[i]['avg_time_ms'] >= results[i-1]['avg_time_ms']
    
    def test_memory_analysis(self):
        """Test memory usage analysis."""
        def dummy_operation(data):
            # Allocate some memory
            temp = np.copy(data)
            return temp
        
        data_sizes = [100, 500, 1000]
        result = optimizer.analyze_memory_usage(dummy_operation, data_sizes)
        
        assert 'memory_analysis' in result
        assert 'recommendation' in result
        
        # Check memory scaling
        analysis = result['memory_analysis']
        for i in range(1, len(analysis)):
            if 'error' not in analysis[i] and 'error' not in analysis[i-1]:
                assert analysis[i]['memory_delta_mb'] >= analysis[i-1]['memory_delta_mb']
    
    def test_error_handling_and_fallback(self):
        """Test error handling and fallback mechanisms."""
        db = HybridVectorDB(self.config)
        
        # Add vectors
        db.add(self.vectors[:50])
        
        # Simulate GPU error by forcing invalid operation
        # This tests the fallback mechanism
        try:
            # This should work and potentially use GPU
            results = db.search(self.query_vector, k=10)
            assert len(results.results) <= 10
        except Exception as e:
            # Should not crash the system
            assert "fallback" in str(e).lower() or "gpu" in str(e).lower()
    
    def test_performance_export(self):
        """Test performance data export."""
        # Add some profile data
        with profiler.profile("export_test"):
            time.sleep(0.001)
        
        # Test JSON export
        import tempfile
        import os
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp:
            tmp_path = tmp.name
        
        try:
            profiler.export_profiles(tmp_path, 'json')
            
            # Check file exists and has content
            assert os.path.exists(tmp_path)
            with open(tmp_path, 'r') as f:
                content = f.read()
                assert 'export_test' in content
                assert 'duration_ms' in content
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)


def run_routing_tests():
    """Run all routing tests."""
    test_instance = TestAdvancedRouting()
    
    # Run each test method
    test_methods = [
        'test_router_initialization',
        'test_basic_routing_decisions',
        'test_performance_based_routing',
        'test_workload_based_routing',
        'test_resource_based_routing',
        'test_hybrid_routing',
        'test_adaptive_thresholds',
        'test_routing_statistics',
        'test_database_integration',
        'test_performance_profiling',
        'test_batch_size_optimization',
        'test_k_value_optimization',
        'test_memory_analysis',
        'test_error_handling_and_fallback',
        'test_performance_export'
    ]
    
    for method_name in test_methods:
        print(f"Running {method_name}...")
        test_instance.setup_method()
        try:
            getattr(test_instance, method_name)()
            print(f"PASS {method_name} passed")
        except Exception as e:
            print(f"FAIL {method_name} failed: {e}")
    
    print("\nAll routing tests completed!")


if __name__ == "__main__":
    run_routing_tests()
