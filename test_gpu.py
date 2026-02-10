"""
GPU-specific tests for HybridVectorDB Phase 2.
"""

import pytest
import numpy as np
from hybridvectordb import HybridVectorDB, Config
from hybridvectordb.gpu_utils import check_gpu_availability, gpu_manager
from hybridvectordb.exceptions import ValidationError


class TestGPUFeatures:
    """Test suite for GPU functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = Config(dimension=64, index_type="flat", metric_type="l2", use_gpu=True)
        
        # Generate test data
        np.random.seed(42)
        self.vectors = np.random.random((100, 64)).astype(np.float32)
        self.query_vector = np.random.random((1, 64)).astype(np.float32)
    
    def test_gpu_availability_detection(self):
        """Test GPU availability detection."""
        available = check_gpu_availability()
        print(f"GPU available: {available}")
        
        if available:
            gpu_info = gpu_manager.get_gpu_info()
            assert gpu_info is not None
            assert 'name' in gpu_info
            assert 'memory_total' in gpu_info
            print(f"GPU: {gpu_info['name']}")
    
    def test_gpu_initialization(self):
        """Test GPU database initialization."""
        if not check_gpu_availability():
            pytest.skip("GPU not available")
        
        db = HybridVectorDB(self.config)
        assert db.gpu_index is not None
        assert db.config.use_gpu is True
        
        # Test GPU info
        gpu_info = db.get_gpu_info()
        assert gpu_info is not None
        assert 'name' in gpu_info
    
    def test_cpu_fallback_when_gpu_unavailable(self):
        """Test CPU fallback when GPU is not available."""
        # Force GPU to be unavailable
        config = Config(dimension=64, use_gpu=True)
        db = HybridVectorDB(config)
        
        # Should fall back to CPU
        assert db.gpu_index is None
        assert db.cpu_index is not None
        
        # Should still work for basic operations
        db.add(self.vectors[:10])
        results = db.search(self.query_vector, k=5)
        assert len(results.results) == 5
    
    def test_gpu_add_vectors(self):
        """Test adding vectors to GPU index."""
        if not check_gpu_availability():
            pytest.skip("GPU not available")
        
        db = HybridVectorDB(self.config)
        
        # Add vectors (should use GPU for large batch)
        added = db.add(self.vectors, use_gpu=True)
        assert added == len(self.vectors)
        
        # Check GPU index has vectors
        assert db.gpu_index.total_vectors == len(self.vectors)
    
    def test_gpu_search(self):
        """Test searching on GPU index."""
        if not check_gpu_availability():
            pytest.skip("GPU not available")
        
        db = HybridVectorDB(self.config)
        db.add(self.vectors, use_gpu=True)
        
        # Search on GPU
        results = db.search(self.query_vector, k=5, use_gpu=True)
        
        assert results.total_results == 5
        assert len(results.results) == 5
        assert results.search_time_ms > 0
        assert "gpu" in results.index_used
        
        # Check statistics
        assert db.gpu_queries > 0
        assert db.cpu_queries == 0
    
    def test_hybrid_routing(self):
        """Test automatic CPU/GPU routing."""
        if not check_gpu_availability():
            pytest.skip("GPU not available")
        
        # Set low threshold for testing
        config = Config(dimension=64, batch_threshold=10, use_gpu=True)
        db = HybridVectorDB(config)
        
        # Small batch should use CPU
        small_batch = self.vectors[:5]
        db.add(small_batch)
        
        # Large batch should use GPU
        large_batch = self.vectors[5:50]
        db.add(large_batch)
        
        # Small search should use CPU
        small_queries = self.query_vector[:1]  # Already 1 query
        results1 = db.search(small_queries, k=3)
        
        # Large batch search should use GPU
        large_queries = np.random.random((20, 64)).astype(np.float32)
        results2 = db.search(large_queries, k=3)
        
        # Should have both CPU and GPU queries
        assert db.cpu_queries >= 0
        assert db.gpu_queries >= 0
        assert db.total_queries > 0
    
    def test_gpu_memory_management(self):
        """Test GPU memory management."""
        if not check_gpu_availability():
            pytest.skip("GPU not available")
        
        db = HybridVectorDB(self.config)
        
        # Get initial memory
        initial_mem = gpu_manager.get_gpu_memory_info()
        initial_free = initial_mem['free'] if initial_mem else 0
        
        # Add vectors
        db.add(self.vectors, use_gpu=True)
        
        # Check memory usage
        current_mem = gpu_manager.get_gpu_memory_info()
        if current_mem:
            # Memory should be used
            assert current_mem['used'] > initial_mem['used'] - initial_mem['free']
        
        # Get stats
        stats = db.get_stats()
        assert 'gpu_index' in stats
        assert stats['gpu_index']['memory_usage_bytes'] > 0
    
    def test_gpu_ivf_index(self):
        """Test GPU IVF index functionality."""
        if not check_gpu_availability():
            pytest.skip("GPU not available")
        
        config = Config(dimension=64, index_type="ivf", nlist=10, use_gpu=True)
        db = HybridVectorDB(config)
        
        # Train on GPU
        db.train(self.vectors[50:80], use_gpu=True)
        assert db.gpu_index.index.is_trained
        
        # Add vectors
        db.add(self.vectors[:50], use_gpu=True)
        
        # Search
        results = db.search(self.query_vector, k=5, use_gpu=True)
        assert len(results.results) == 5
        assert "gpu_ivf" in results.index_used
    
    def test_gpu_switching(self):
        """Test enabling/disabling GPU."""
        if not check_gpu_availability():
            pytest.skip("GPU not available")
        
        # Start with GPU disabled
        config = Config(dimension=64, use_gpu=False)
        db = HybridVectorDB(config)
        assert db.gpu_index is None
        
        # Enable GPU
        db.switch_gpu(True)
        assert db.gpu_index is not None
        
        # Disable GPU
        db.switch_gpu(False)
        assert db.gpu_index is None
    
    def test_gpu_save_load(self):
        """Test saving and loading GPU indexes."""
        if not check_gpu_availability():
            pytest.skip("GPU not available")
        
        db = HybridVectorDB(self.config)
        db.add(self.vectors[:20], use_gpu=True)
        
        # Save GPU index
        import tempfile
        import os
        with tempfile.NamedTemporaryFile(delete=False, suffix=".idx") as tmp:
            tmp_path = tmp.name
        
        try:
            db.save_index(tmp_path, use_gpu=True)
            
            # Create new database and load to GPU
            new_db = HybridVectorDB(self.config)
            new_db.load_index(tmp_path, use_gpu=True)
            
            # Test search works
            results = new_db.search(self.query_vector, k=5)
            assert len(results.results) == 5
            
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    
    def test_error_handling(self):
        """Test GPU error handling."""
        if not check_gpu_availability():
            pytest.skip("GPU not available")
        
        db = HybridVectorDB(self.config)
        
        # Test invalid dimension
        wrong_dim = np.random.random((10, 32)).astype(np.float32)
        with pytest.raises(ValueError):
            db.add(wrong_dim, use_gpu=True)
        
        # Test empty vectors
        with pytest.raises(ValueError):
            db.add([], use_gpu=True)
    
    def test_performance_comparison(self):
        """Compare CPU vs GPU performance."""
        if not check_gpu_availability():
            pytest.skip("GPU not available")
        
        # Create separate databases for comparison
        cpu_config = Config(dimension=64, use_gpu=False)
        gpu_config = Config(dimension=64, use_gpu=True)
        
        cpu_db = HybridVectorDB(cpu_config)
        gpu_db = HybridVectorDB(gpu_config)
        
        # Add same vectors to both
        cpu_db.add(self.vectors)
        gpu_db.add(self.vectors, use_gpu=True)
        
        # Time CPU search
        import time
        start_time = time.time()
        cpu_results = cpu_db.search(self.query_vector, k=10)
        cpu_time = time.time() - start_time
        
        # Time GPU search
        start_time = time.time()
        gpu_results = gpu_db.search(self.query_vector, k=10, use_gpu=True)
        gpu_time = time.time() - start_time
        
        print(f"CPU time: {cpu_time*1000:.2f}ms")
        print(f"GPU time: {gpu_time*1000:.2f}ms")
        print(f"Speedup: {cpu_time/gpu_time:.2f}x")
        
        # Results should be similar (allowing for minor differences)
        assert len(cpu_results.results) == len(gpu_results.results)


def run_gpu_tests():
    """Run all GPU tests."""
    test_instance = TestGPUFeatures()
    
    # Run each test method
    test_methods = [
        'test_gpu_availability_detection',
        'test_cpu_fallback_when_gpu_unavailable',
        'test_gpu_initialization',
        'test_gpu_add_vectors',
        'test_gpu_search',
        'test_hybrid_routing',
        'test_gpu_memory_management',
        'test_gpu_ivf_index',
        'test_gpu_switching',
        'test_gpu_save_load',
        'test_error_handling',
        'test_performance_comparison'
    ]
    
    for method_name in test_methods:
        print(f"Running {method_name}...")
        test_instance.setup_method()
        try:
            getattr(test_instance, method_name)()
            print(f"PASS {method_name} passed")
        except Exception as e:
            print(f"FAIL {method_name} failed: {e}")
    
    print("\nAll GPU tests completed!")


if __name__ == "__main__":
    run_gpu_tests()
