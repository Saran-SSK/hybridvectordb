"""
CPU-based FAISS index implementation.
"""

import faiss
import numpy as np
from typing import List, Tuple, Optional
import time
import logging

from .config import Config
from .models import VectorData, SearchResult, SearchResponse

logger = logging.getLogger(__name__)


class CPUIndex:
    """FAISS CPU index wrapper."""
    
    def __init__(self, config: Config):
        """Initialize CPU index with configuration."""
        self.config = config
        self.dimension = config.dimension
        self.metric_type = config.metric_type
        self.index_type = config.index_type
        
        # Initialize FAISS index
        self.index = self._create_index()
        
        # Storage for metadata
        self.vector_ids: List[str] = []
        self.metadata_store: dict = {}
        
        # Statistics
        self.total_vectors = 0
        
        logger.info(f"Initialized CPU index: {type(self.index).__name__}")
    
    def _create_index(self) -> faiss.Index:
        """Create FAISS index based on configuration."""
        # Convert metric type to FAISS metric
        if self.metric_type == "l2":
            metric = faiss.METRIC_L2
        elif self.metric_type == "inner_product":
            metric = faiss.METRIC_INNER_PRODUCT
        elif self.metric_type == "cosine":
            # For cosine similarity, we normalize vectors and use inner product
            metric = faiss.METRIC_INNER_PRODUCT
        else:
            raise ValueError(f"Unsupported metric type: {self.metric_type}")
        
        if self.index_type == "flat":
            index = faiss.IndexFlat(self.dimension, metric)
        elif self.index_type == "ivf":
            # IVF index requires training
            quantizer = faiss.IndexFlat(self.dimension, metric)
            index = faiss.IndexIVFFlat(quantizer, self.dimension, self.config.nlist, metric)
        else:
            raise ValueError(f"Unsupported index type: {self.index_type}")
        
        return index
    
    def _normalize_vectors(self, vectors: np.ndarray) -> np.ndarray:
        """Normalize vectors for cosine similarity."""
        if self.metric_type == "cosine":
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            norms[norms == 0] = 1  # Avoid division by zero
            return vectors / norms
        return vectors
    
    def add_vectors(self, vector_data_list: List[VectorData]) -> int:
        """Add vectors to the index."""
        if not vector_data_list:
            return 0
        
        # Check if IVF index needs training
        if self.index_type == "ivf" and not self.index.is_trained:
            raise ValueError("IVF index must be trained before adding vectors. Call train() first.")
        
        # Convert to numpy array
        embeddings = np.array([vd.embedding for vd in vector_data_list], dtype=np.float32)
        
        # Normalize if using cosine similarity
        embeddings = self._normalize_vectors(embeddings)
        
        # Add to FAISS index
        start_time = time.time()
        if hasattr(self.index, 'add_with_ids') and self.index_type != "flat":
            faiss_ids = list(range(self.total_vectors, self.total_vectors + len(vector_data_list)))
            self.index.add_with_ids(embeddings, np.array(faiss_ids, dtype=np.int64))
        else:
            # For IndexFlat, use regular add and track IDs separately
            self.index.add(embeddings)
            faiss_ids = list(range(self.total_vectors, self.total_vectors + len(vector_data_list)))
        
        # Store metadata
        for i, vd in enumerate(vector_data_list):
            faiss_id = faiss_ids[i]
            self.vector_ids.append(vd.id)
            self.metadata_store[faiss_id] = vd.metadata or {}
        
        self.total_vectors += len(vector_data_list)
        add_time = time.time() - start_time
        
        logger.info(f"Added {len(vector_data_list)} vectors in {add_time:.4f}s")
        return len(vector_data_list)
    
    def search(self, query_vectors: np.ndarray, k: int = 10) -> List[SearchResponse]:
        """Search for similar vectors."""
        if self.total_vectors == 0:
            raise ValueError("No vectors in index")
        
        # Normalize query vectors if using cosine similarity
        query_vectors = self._normalize_vectors(query_vectors)
        
        # Adjust k if necessary
        k = min(k, self.total_vectors)
        
        # Perform search
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
                
                # Handle case where vector_ids might not be populated (e.g., after loading)
                if idx < len(self.vector_ids):
                    vector_id = self.vector_ids[idx]
                else:
                    vector_id = f"loaded_vec_{idx}"
                
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
                index_used=f"cpu_{self.index_type}"
            )
            responses.append(response)
        
        return responses
    
    def train(self, vectors: np.ndarray) -> None:
        """Train IVF index if needed."""
        if self.index_type == "ivf" and not self.index.is_trained:
            vectors = self._normalize_vectors(vectors)
            self.index.train(vectors)
            logger.info(f"Trained IVF index with {len(vectors)} vectors")
    
    def get_stats(self) -> dict:
        """Get index statistics."""
        return {
            "total_vectors": self.total_vectors,
            "index_type": self.index_type,
            "metric_type": self.metric_type,
            "dimension": self.dimension,
            "is_trained": self.index.is_trained if hasattr(self.index, 'is_trained') else True,
            "ntotal": self.index.ntotal
        }
