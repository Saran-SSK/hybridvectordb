"""
Example usage of HybridVectorDB (Phase 1).
"""

import numpy as np
from hybridvectordb import HybridVectorDB, Config

def main():
    """Demonstrate basic usage of HybridVectorDB."""
    
    # Create configuration
    config = Config(
        dimension=128,
        index_type="flat",  # Use "ivf" for approximate search
        metric_type="l2"    # Options: "l2", "inner_product", "cosine"
    )
    
    # Initialize database
    db = HybridVectorDB(config)
    print(f"Initialized: {db}")
    
    # Generate some sample data
    np.random.seed(42)
    num_vectors = 1000
    vectors = np.random.random((num_vectors, 128)).astype(np.float32)
    
    # Create metadata
    metadata = [
        {"source": f"doc_{i}", "category": "test" if i % 2 == 0 else "validation"}
        for i in range(num_vectors)
    ]
    
    # Create IDs
    ids = [f"vector_{i}" for i in range(num_vectors)]
    
    # Add vectors to database
    print(f"\nAdding {num_vectors} vectors...")
    added = db.add(vectors, metadata=metadata, ids=ids)
    print(f"Added {added} vectors to database")
    print(f"Database size: {len(db)} vectors")
    
    # Perform single query search
    print("\n--- Single Query Search ---")
    query_vector = np.random.random((1, 128)).astype(np.float32)
    results = db.search(query_vector, k=5)
    
    print(f"Search time: {results.search_time_ms:.2f} ms")
    print(f"Index used: {results.index_used}")
    print("Top 5 results:")
    for i, result in enumerate(results.results):
        print(f"  {i+1}. ID: {result.id}, Distance: {result.distance:.4f}, "
              f"Category: {result.metadata.get('category', 'N/A')}")
    
    # Perform batch query search
    print("\n--- Batch Query Search ---")
    batch_queries = np.random.random((10, 128)).astype(np.float32)
    batch_results = db.search(batch_queries, k=3)
    
    print(f"Processed {len(batch_results)} queries")
    for i, response in enumerate(batch_results[:3]):  # Show first 3
        print(f"Query {i}: {len(response.results)} results in {response.search_time_ms:.2f} ms")
    
    # Get database statistics
    print("\n--- Database Statistics ---")
    stats = db.get_stats()
    print(f"Total vectors: {stats['cpu_index']['total_vectors']}")
    print(f"Total queries: {stats['total_queries']}")
    print(f"Average query time: {stats['avg_query_time_ms']:.2f} ms")
    print(f"Index type: {stats['cpu_index']['index_type']}")
    print(f"Metric type: {stats['cpu_index']['metric_type']}")
    
    # Test different distance metrics
    print("\n--- Testing Different Metrics ---")
    
    # Cosine similarity
    cosine_config = Config(dimension=128, metric_type="cosine")
    cosine_db = HybridVectorDB(cosine_config)
    
    # Add normalized vectors for cosine
    normalized_vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    cosine_db.add(normalized_vectors[:100])  # Add smaller subset
    
    # Search with cosine
    cosine_query = normalized_vectors[0:1]
    cosine_results = cosine_db.search(cosine_query, k=5)
    print(f"Cosine search time: {cosine_results.search_time_ms:.2f} ms")
    print("Cosine results (distances are similarities):")
    for i, result in enumerate(cosine_results.results[:3]):
        print(f"  {i+1}. ID: {result.id}, Similarity: {result.distance:.4f}")
    
    # Test IVF index (approximate search)
    print("\n--- Testing IVF Index ---")
    ivf_config = Config(
        dimension=128,
        index_type="ivf",
        metric_type="l2",
        nlist=50,
        nprobe=5
    )
    ivf_db = HybridVectorDB(ivf_config)
    
    # Train IVF index first
    ivf_db.train(vectors[500:700])  # Train with some vectors
    
    # Add vectors after training
    ivf_db.add(vectors[:500])  # Add vectors after training
    
    # Search with IVF
    ivf_results = ivf_db.search(query_vector, k=5)
    print(f"IVF search time: {ivf_results.search_time_ms:.2f} ms")
    print(f"IVF index trained: {ivf_db.cpu_index.index.is_trained}")
    
    print("\n--- Example Complete ---")


if __name__ == "__main__":
    main()
