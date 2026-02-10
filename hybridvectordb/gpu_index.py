"""
GPU-based FAISS index implementation.
"""

import faiss
import numpy as np
from typing import List, Tuple, Optional
import time
import logging

from .config import Config
from .models import VectorData, SearchResult, SearchResponse
from .gpu_utils import gpu_manager, create_gpu_index, check_gpu_availability
from .exceptions import SearchError, IndexError

logger = logging.getLogger(__name__)


class GPUIndex:
    """FAISS GPU index wrapper."""
    
    def __init__(self, config: Config, gpu_id: int = 0):
        """Initialize GPU index with configuration."""
        self.config = config
        self.dimension = config.dimension
        self.metric_type = config.metric_type
        self.index_type = config.index_type
        self.gpu_id = gpu_id
        
        # Check GPU availability
        if not check_gpu_availability():
            raise RuntimeError("GPU not available. Install faiss-gpu and ensure CUDA is properly configured.")
        
        # Initialize GPU index
        self.index = None
        self.resources = None
        self._create_index()
        
        # Storage for metadata (kept on CPU)
        self.vector_ids: List[str] = []
        self.metadata_store: dict = {}
        
        # Statistics
        self.total_vectors = 0
        self.memory_usage_bytes = 0
        
        logger.info(f"Initialized GPU index: {type(self.index).__name__} on GPU {gpu_id}")
    
    def _create_index(self) -> None:
        """Create FAISS GPU index."""
        try:
            self.index, self.resources = create_gpu_index(
                dimension=self.dimension,
                metric_type=self.metric_type,
                index_type=self.index_type,
                gpu_id=self.gpu_id
            )
        except Exception as e:
            logger.error(f"Failed to create GPU index: {e}")
            raise RuntimeError(f"Failed to create GPU index: {e}")
    
    def _normalize_vectors(self, vectors: np.ndarray) -> np.ndarray:
        """Normalize vectors for cosine similarity."""
        if self.metric_type == "cosine":
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            norms[norms == 0] = 1  # Avoid division by zero
            return vectors / norms
        return vectors
    
    def _estimate_memory_usage(self, num_vectors: int) -> int:
        """Estimate memory usage for adding vectors."""
        return gpu_manager.estimate_memory_usage(
            num_vectors=num_vectors,
            dimension=self.dimension,
            index_type=self.index_type
        )
    
    def _check_memory_availability(self, num_vectors: int) -> bool:
        """Check if GPU has enough memory for adding vectors."""
        required_memory = self._estimate_memory_usage(num_vectors)
        return gpu_manager.can_use_gpu(memory_required=required_memory, gpu_id=self.gpu_id)
    
    def add_vectors(self, vector_data_list: List[VectorData]) -> int:
        """Add vectors to the GPU index."""
        if not vector_data_list:
            return 0
        
        # Check memory availability
        if not self._check_memory_availability(len(vector_data_list)):
            raise MemoryError(f"Insufficient GPU memory to add {len(vector_data_list)} vectors")
        
        # Convert to numpy array
        embeddings = np.array([vd.embedding for vd in vector_data_list], dtype=np.float32)
        
        # Normalize if using cosine similarity
        embeddings = self._normalize_vectors(embeddings)
        
        # Add to GPU index
        start_time = time.time()
        
        # For GPU indexes, we need to handle IDs differently
        if hasattr(self.index, 'add_with_ids'):
            faiss_ids = list(range(self.total_vectors, self.total_vectors + len(vector_data_list)))
            self.index.add_with_ids(embeddings, np.array(faiss_ids, dtype=np.int64))
        else:
            # Use regular add
            self.index.add(embeddings)
            faiss_ids = list(range(self.total_vectors, self.total_vectors + len(vector_data_list)))
        
        # Store metadata on CPU
        for i, vd in enumerate(vector_data_list):
            faiss_id = faiss_ids[i]
            self.vector_ids.append(vd.id)
            self.metadata_store[faiss_id] = vd.metadata or {}
        
        self.total_vectors += len(vector_data_list)
        self.memory_usage_bytes += self._estimate_memory_usage(len(vector_data_list))
        add_time = time.time() - start_time
        
        logger.info(f"Added {len(vector_data_list)} vectors to GPU in {add_time:.4f}s")
        return len(vector_data_list)
    
    def search(self, query_vectors: np.ndarray, k: int = 10) -> List[SearchResponse]:
        """Search for similar vectors on GPU."""
        if self.total_vectors == 0:
            raise ValueError("No vectors in index")
        
        # Normalize query vectors if using cosine similarity
        query_vectors = self._normalize_vectors(query_vectors)
        
        # Adjust k if necessary
        k = min(k, self.total_vectors)
        
        # Perform search on GPU
        start_time = time.time()
        distances, indices = self.index.search(query_vectors, k)
        search_time = (time.time() - start_time) * 1000  # Convert to ms
        
        # Convert to SearchResponse objects
        responses = []
        for i, query_vec in enumerate(query_vectors):
            results = []
            for j, idx in enumerate(indices[i]):
                if idx == -1:  # FAISS returns -1 for missing results
                    continue
                
                # Handle case where vector_ids might not be populated
                if idx < len(self.vector_ids):
                    vector_id = self.vector_ids[idx]
                else:
                    vector_id = f"gpu_vec_{idx}"
                
                distance = float(distances[i][j])
                metadata = self.metadata_store.get(idx, {})
                
                result = SearchResult(
                    id=vector_id,
                    distance=distance,
                    metadata=metadata
                )
                results.append(result)
            
            response = SearchResponse(
                query_id=f"query_{i}",
                results=results,
                total_results=len(results),
                search_time_ms=search_time,
                index_used=f"gpu_{self.index_type}"
            )
            responses.append(response)
        
        return responses
    
    def train(self, vectors: np.ndarray) -> None:
        """Train IVF index if needed."""
        if self.index_type == "ivf" and not self.index.is_trained:
            # Check memory availability for training
            if not self._check_memory_availability(len(vectors)):
                raise MemoryError(f"Insufficient GPU memory to train with {len(vectors)} vectors")
            
            vectors = self._normalize_vectors(vectors)
            self.index.train(vectors)
            logger.info(f"Trained GPU IVF index with {len(vectors)} vectors")
    
    def reset(self) -> None:
        """Reset the GPU index."""
        try:
            self.index.reset()
            self.vector_ids.clear()
            self.metadata_store.clear()
            self.total_vectors = 0
            self.memory_usage_bytes = 0
            logger.info("Reset GPU index")
        except Exception as e:
            logger.error(f"Failed to reset GPU index: {e}")
            raise IndexError(f"Failed to reset GPU index: {e}")
    
    def get_stats(self) -> dict:
        """Get index statistics."""
        stats = {
            "total_vectors": self.total_vectors,
            "index_type": self.index_type,
            "metric_type": self.metric_type,
            "dimension": self.dimension,
            "gpu_id": self.gpu_id,
            "is_trained": self.index.is_trained if hasattr(self.index, 'is_trained') else True,
            "ntotal": self.index.ntotal,
            "memory_usage_bytes": self.memory_usage_bytes,
            "memory_usage_mb": self.memory_usage_bytes / (1024 * 1024),
        }
        
        # Add GPU-specific stats
        gpu_info = gpu_manager.get_gpu_info(self.gpu_id)
        if gpu_info:
            stats.update({
                "gpu_name": gpu_info['name'],
                "gpu_memory_total": gpu_info['memory_total'],
                "gpu_memory_free": gpu_info['memory_free'],
                "gpu_memory_used": gpu_info['memory_used'],
                "gpu_utilization": gpu_manager.get_gpu_utilization(self.gpu_id),
            })
        
        return stats
    
    def __del__(self):
        """Cleanup GPU resources."""
        try:
            if hasattr(self, 'resources') and self.resources is not None:
                # FAISS GPU resources are automatically cleaned up
                pass
        except Exception as e:
            logger.warning(f"Error during GPU cleanup: {e}")
