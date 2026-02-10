"""
Basic tests for HybridVectorDB Phase 1.
"""

import pytest
import numpy as np
from hybridvectordb import HybridVectorDB, Config
from hybridvectordb.exceptions import ValidationError, ConfigurationError


class TestHybridVectorDB:
    """Test suite for HybridVectorDB."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = Config(dimension=64, index_type="flat", metric_type="l2")
        self.db = HybridVectorDB(self.config)
        
        # Generate test data
        np.random.seed(42)
        self.vectors = np.random.random((100, 64)).astype(np.float32)
        self.query_vector = np.random.random((1, 64)).astype(np.float32)
    
    def test_initialization(self):
        """Test database initialization."""
        assert self.db.config.dimension == 64
        assert self.db.config.index_type == "flat"
        assert self.db.config.metric_type == "l2"
        assert len(self.db) == 0
    
    def test_add_vectors(self):
        """Test adding vectors."""
        # Add vectors
        added = self.db.add(self.vectors[:50])
        assert added == 50
        assert len(self.db) == 50
        
        # Add more vectors
        added = self.db.add(self.vectors[50:])
        assert added == 50
        assert len(self.db) == 100
    
    def test_add_with_metadata(self):
        """Test adding vectors with metadata."""
        metadata = [{"test": i} for i in range(10)]
        ids = [f"test_{i}" for i in range(10)]
        
        added = self.db.add(self.vectors[:10], metadata=metadata, ids=ids)
        assert added == 10
        assert len(self.db) == 10
    
    def test_search_single_query(self):
        """Test single query search."""
        # Add vectors first
        self.db.add(self.vectors)
        
        # Search
        results = self.db.search(self.query_vector, k=5)
        
        assert results.total_results == 5
        assert len(results.results) == 5
        assert results.search_time_ms > 0
        assert results.index_used == "cpu_flat"
        
        # Check result structure
        for result in results.results:
            assert hasattr(result, 'id')
            assert hasattr(result, 'distance')
            assert hasattr(result, 'metadata')
            assert isinstance(result.distance, float)
    
    def test_search_batch_queries(self):
        """Test batch query search."""
        # Add vectors first
        self.db.add(self.vectors)
        
        # Batch search
        batch_queries = np.random.random((5, 64)).astype(np.float32)
        results = self.db.search(batch_queries, k=3)
        
        assert len(results) == 5
        for response in results:
            assert response.total_results == 3
            assert len(response.results) == 3
    
    def test_different_metrics(self):
        """Test different distance metrics."""
        # Test L2
        l2_config = Config(dimension=64, metric_type="l2")
        l2_db = HybridVectorDB(l2_config)
        l2_db.add(self.vectors[:10])
        l2_results = l2_db.search(self.query_vector, k=3)
        assert l2_results.index_used == "cpu_flat"
        
        # Test Inner Product
        ip_config = Config(dimension=64, metric_type="inner_product")
        ip_db = HybridVectorDB(ip_config)
        ip_db.add(self.vectors[:10])
        ip_results = ip_db.search(self.query_vector, k=3)
        assert ip_results.index_used == "cpu_flat"
        
        # Test Cosine
        cosine_config = Config(dimension=64, metric_type="cosine")
        cosine_db = HybridVectorDB(cosine_config)
        cosine_db.add(self.vectors[:10])
        cosine_results = cosine_db.search(self.query_vector, k=3)
        assert cosine_results.index_used == "cpu_flat"
    
    def test_ivf_index(self):
        """Test IVF index functionality."""
        ivf_config = Config(dimension=64, index_type="ivf", nlist=10)
        ivf_db = HybridVectorDB(ivf_config)
        
        # Train index first
        ivf_db.train(self.vectors[50:80])
        
        # Add vectors after training
        ivf_db.add(self.vectors[:50])
        
        # Search
        results = ivf_db.search(self.query_vector, k=5)
        assert results.index_used == "cpu_ivf"
        assert ivf_db.cpu_index.index.is_trained
    
    def test_validation_errors(self):
        """Test validation errors."""
        # Wrong dimension
        wrong_dim_vectors = np.random.random((10, 32)).astype(np.float32)
        with pytest.raises(ValueError):
            self.db.add(wrong_dim_vectors)
        
        # Wrong query dimension
        with pytest.raises(ValueError):
            self.db.search(np.random.random((1, 32)).astype(np.float32))
        
        # Empty vectors list
        with pytest.raises(ValueError):
            self.db.add([])
        
        # Empty numpy array
        with pytest.raises(ValueError):
            self.db.add(np.array([]))
    
    def test_statistics(self):
        """Test database statistics."""
        # Add vectors
        self.db.add(self.vectors[:50])
        
        # Perform some searches
        self.db.search(self.query_vector, k=5)
        
        # Get stats
        stats = self.db.get_stats()
        
        assert stats['cpu_index']['total_vectors'] == 50
        assert stats['total_queries'] == 1
        assert stats['avg_query_time_ms'] > 0
        assert stats['phase'] == "1 (CPU only)"
    
    def test_save_load_index(self):
        """Test saving and loading index."""
        # Add vectors
        self.db.add(self.vectors[:20])
        original_count = len(self.db)
        
        # Save index
        import tempfile
        import os
        with tempfile.NamedTemporaryFile(delete=False, suffix=".idx") as tmp:
            tmp_path = tmp.name
        
        try:
            self.db.save_index(tmp_path)
            
            # Create new database and load
            new_db = HybridVectorDB(self.config)
            new_db.load_index(tmp_path)
            
            # The loaded index should have the same number of vectors
            # Note: FAISS doesn't preserve metadata in save/load for basic indexes
            assert new_db.cpu_index.index.ntotal == original_count
            
            # Update the total_vectors count to match loaded index
            new_db.cpu_index.total_vectors = new_db.cpu_index.index.ntotal
            
            # Test search works
            results = new_db.search(self.query_vector, k=5)
            assert len(results.results) == 5
            
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)


def run_tests():
    """Run all tests."""
    test_instance = TestHybridVectorDB()
    
    # Run each test method
    test_methods = [
        'test_initialization',
        'test_add_vectors',
        'test_add_with_metadata',
        'test_search_single_query',
        'test_search_batch_queries',
        'test_different_metrics',
        'test_ivf_index',
        'test_validation_errors',
        'test_statistics',
        'test_save_load_index'
    ]
    
    for method_name in test_methods:
        print(f"Running {method_name}...")
        test_instance.setup_method()
        try:
            getattr(test_instance, method_name)()
            print(f"PASS {method_name} passed")
        except Exception as e:
            print(f"FAIL {method_name} failed: {e}")
    
    print("\nAll tests completed!")


if __name__ == "__main__":
    run_tests()
