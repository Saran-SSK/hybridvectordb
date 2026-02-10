"""
GPU usage examples for HybridVectorDB Phase 2.
"""

import numpy as np
import time
from hybridvectordb import HybridVectorDB, Config
from hybridvectordb.gpu_utils import check_gpu_availability, gpu_manager


def main():
    """Demonstrate GPU functionality."""
    
    print("=== HybridVectorDB Phase 2 GPU Examples ===\n")
    
    # Check GPU availability
    gpu_available = check_gpu_availability()
    print(f"GPU Available: {gpu_available}")
    
    if gpu_available:
        gpu_info = gpu_manager.get_gpu_info()
        if gpu_info:
            print(f"GPU Name: {gpu_info['name']}")
            print(f"GPU Memory: {gpu_info['memory_free']/1024**3:.1f}GB free / "
                  f"{gpu_info['memory_total']/1024**3:.1f}GB total")
    
    print()
    
    # Example 1: Basic GPU usage
    print("--- Example 1: Basic GPU Usage ---")
    try:
        config = Config(
            dimension=128,
            index_type="flat",
            metric_type="l2",
            use_gpu=True,
            batch_threshold=32  # Use GPU for batches >= 32
        )
        
        db = HybridVectorDB(config)
        print(f"Database: {db}")
        
        # Generate sample data
        np.random.seed(42)
        vectors = np.random.random((1000, 128)).astype(np.float32)
        metadata = [{"source": f"doc_{i}"} for i in range(1000)]
        
        # Add vectors (will use GPU for large batch)
        print("Adding 1000 vectors...")
        start_time = time.time()
        added = db.add(vectors, metadata=metadata)
        add_time = time.time() - start_time
        print(f"Added {added} vectors in {add_time:.4f}s")
        
        # Single query (will use CPU due to small batch)
        query = np.random.random((1, 128)).astype(np.float32)
        results = db.search(query, k=10)
        print(f"Single query: {results.search_time_ms:.2f}ms (index: {results.index_used})")
        
        # Batch query (will use GPU)
        batch_queries = np.random.random((50, 128)).astype(np.float32)
        batch_results = db.search(batch_queries, k=5)
        avg_time = sum(r.search_time_ms for r in batch_results) / len(batch_results)
        print(f"Batch queries: {avg_time:.2f}ms avg (index: {batch_results[0].index_used})")
        
    except Exception as e:
        print(f"GPU example failed: {e}")
        print("Falling back to CPU-only mode")
    
    print()
    
    # Example 2: Forced GPU usage
    if gpu_available:
        print("--- Example 2: Forced GPU Usage ---")
        try:
            config = Config(dimension=128, use_gpu=True)
            db = HybridVectorDB(config)
            
            # Force GPU usage
            vectors = np.random.random((100, 128)).astype(np.float32)
            db.add(vectors, use_gpu=True)
            
            query = np.random.random((1, 128)).astype(np.float32)
            results = db.search(query, k=5, use_gpu=True)
            print(f"Forced GPU search: {results.search_time_ms:.2f}ms")
            print(f"Index used: {results.index_used}")
            
        except Exception as e:
            print(f"Forced GPU example failed: {e}")
    
    print()
    
    # Example 3: GPU IVF Index
    if gpu_available:
        print("--- Example 3: GPU IVF Index ---")
        try:
            config = Config(
                dimension=128,
                index_type="ivf",
                metric_type="l2",
                nlist=50,
                nprobe=5,
                use_gpu=True
            )
            db = HybridVectorDB(config)
            
            # Train on GPU
            train_vectors = np.random.random((500, 128)).astype(np.float32)
            db.train(train_vectors, use_gpu=True)
            print("GPU IVF index trained")
            
            # Add vectors
            vectors = np.random.random((200, 128)).astype(np.float32)
            db.add(vectors, use_gpu=True)
            
            # Search
            query = np.random.random((1, 128)).astype(np.float32)
            results = db.search(query, k=10, use_gpu=True)
            print(f"GPU IVF search: {results.search_time_ms:.2f}ms")
            print(f"Index used: {results.index_used}")
            
        except Exception as e:
            print(f"GPU IVF example failed: {e}")
    
    print()
    
    # Example 4: Performance Comparison
    if gpu_available:
        print("--- Example 4: CPU vs GPU Performance ---")
        try:
            # CPU-only database
            cpu_config = Config(dimension=128, use_gpu=False)
            cpu_db = HybridVectorDB(cpu_config)
            
            # GPU-enabled database
            gpu_config = Config(dimension=128, use_gpu=True)
            gpu_db = HybridVectorDB(gpu_config)
            
            # Same data for both
            vectors = np.random.random((1000, 128)).astype(np.float32)
            queries = np.random.random((100, 128)).astype(np.float32)
            
            # CPU performance
            cpu_db.add(vectors)
            start_time = time.time()
            cpu_results = cpu_db.search(queries, k=10)
            cpu_time = time.time() - start_time
            
            # GPU performance
            gpu_db.add(vectors, use_gpu=True)
            start_time = time.time()
            gpu_results = gpu_db.search(queries, k=10, use_gpu=True)
            gpu_time = time.time() - start_time
            
            print(f"CPU batch search: {cpu_time*1000:.2f}ms")
            print(f"GPU batch search: {gpu_time*1000:.2f}ms")
            if gpu_time > 0:
                speedup = cpu_time / gpu_time
                print(f"GPU speedup: {speedup:.2f}x")
            
        except Exception as e:
            print(f"Performance comparison failed: {e}")
    
    print()
    
    # Example 5: Memory Management
    if gpu_available:
        print("--- Example 5: GPU Memory Management ---")
        try:
            config = Config(dimension=512, use_gpu=True)  # Higher dimension
            db = HybridVectorDB(config)
            
            # Check initial memory
            mem_info = gpu_manager.get_gpu_memory_info()
            if mem_info:
                print(f"Initial GPU memory: {mem_info['used']/1024**3:.2f}GB used")
            
            # Add vectors and check memory usage
            vectors = np.random.random((5000, 512)).astype(np.float32)
            db.add(vectors, use_gpu=True)
            
            mem_info = gpu_manager.get_gpu_memory_info()
            if mem_info:
                print(f"After adding vectors: {mem_info['used']/1024**3:.2f}GB used")
            
            # Get database stats
            stats = db.get_stats()
            if 'gpu_index' in stats:
                gpu_stats = stats['gpu_index']
                print(f"Index memory usage: {gpu_stats['memory_usage_mb']:.2f}MB")
                print(f"GPU queries: {stats['gpu_queries']}")
                print(f"CPU queries: {stats['cpu_queries']}")
                print(f"GPU usage: {stats['gpu_usage_percent']:.1f}%")
            
        except Exception as e:
            print(f"Memory management example failed: {e}")
    
    print()
    
    # Example 6: Dynamic GPU Switching
    if gpu_available:
        print("--- Example 6: Dynamic GPU Switching ---")
        try:
            config = Config(dimension=128, use_gpu=False)
            db = HybridVectorDB(config)
            print(f"Started with GPU disabled: {db}")
            
            # Add some vectors on CPU
            vectors = np.random.random((100, 128)).astype(np.float32)
            db.add(vectors)
            print(f"Added {len(vectors)} vectors on CPU")
            
            # Enable GPU
            db.switch_gpu(True)
            print(f"Enabled GPU: {db}")
            
            # Add more vectors on GPU
            more_vectors = np.random.random((100, 128)).astype(np.float32)
            db.add(more_vectors, use_gpu=True)
            print(f"Added {len(more_vectors)} vectors on GPU")
            
            # Search will use hybrid routing
            query = np.random.random((1, 128)).astype(np.float32)
            results = db.search(query, k=5)
            print(f"Search result: {results.search_time_ms:.2f}ms ({results.index_used})")
            
            # Disable GPU
            db.switch_gpu(False)
            print(f"Disabled GPU: {db}")
            
        except Exception as e:
            print(f"GPU switching example failed: {e}")
    
    print("\n=== GPU Examples Complete ===")


if __name__ == "__main__":
    main()
