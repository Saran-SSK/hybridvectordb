"""
Core HybridVectorDB implementation.
"""

import numpy as np
from typing import List, Union, Optional, Dict, Any
import logging
import time

from .config import Config
from .models import VectorData, SearchResponse
from .cpu_index import CPUIndex
from .gpu_index import GPUIndex
from .gpu_utils import check_gpu_availability, gpu_manager
from .router import AdvancedRouter, RoutingDecision, RoutingContext
from .profiler import profiler, optimizer
from .exceptions import SearchError, IndexError

logger = logging.getLogger(__name__)


class HybridVectorDB:
    """
    Hybrid Vector Database with intelligent query routing.
    
    Phase 3: Advanced CPU/GPU implementation with adaptive routing.
    """
    
    def __init__(self, config: Config):
        """Initialize HybridVectorDB with configuration."""
        self.config = config
        
        # Initialize CPU index (always available)
        self.cpu_index = CPUIndex(config)
        
        # Initialize GPU index if available and enabled
        self.gpu_index = None
        if config.use_gpu and check_gpu_availability():
            try:
                self.gpu_index = GPUIndex(config, gpu_id=0)
                logger.info("GPU index initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize GPU index: {e}")
                self.gpu_index = None
        elif config.use_gpu:
            logger.warning("GPU requested but not available")
        
        # Initialize advanced router
        self.router = AdvancedRouter(config)
        
        # Statistics
        self.total_queries = 0
        self.total_search_time_ms = 0.0
        self.gpu_queries = 0
        self.cpu_queries = 0
        
        phase_text = "3 (Advanced CPU + GPU)" if self.gpu_index else "3 (Advanced CPU only, GPU unavailable)"
        logger.info(f"Initialized HybridVectorDB ({phase_text})")
        logger.info(f"Config: dimension={config.dimension}, index_type={config.index_type}")
        logger.info(f"GPU available: {self.gpu_index is not None}")
        logger.info(f"Routing strategy: {self.router.current_strategy}")
    
    def add(self, 
            vectors: Union[List[List[float]], np.ndarray], 
            metadata: Optional[List[Dict[str, Any]]] = None,
            ids: Optional[List[str]] = None,
            use_gpu: Optional[bool] = None) -> int:
        """
        Add vectors to database.
        
        Args:
            vectors: List of vector embeddings or numpy array
            metadata: Optional list of metadata dictionaries
            ids: Optional list of vector IDs
            use_gpu: Force GPU usage (None = auto-routing)
            
        Returns:
            Number of vectors added
        """
        # Handle empty input
        if vectors is None:
            raise ValueError("No vectors provided")
        
        if isinstance(vectors, np.ndarray):
            if vectors.size == 0:
                raise ValueError("Empty vectors array provided")
        elif isinstance(vectors, list):
            if len(vectors) == 0:
                raise ValueError("Empty vectors list provided")
        
        # Convert to numpy array if needed
        if isinstance(vectors, list):
            if len(vectors) == 0:
                raise ValueError("Empty vectors list provided")
            vectors = np.array(vectors, dtype=np.float32)
        
        # Validate dimensions
        if vectors.ndim != 2:
            raise ValueError(f"Vectors must be 2D array, got {vectors.ndim}D")
        
        if vectors.shape[1] != self.config.dimension:
            raise ValueError(f"Vector dimension mismatch: expected {self.config.dimension}, got {vectors.shape[1]}")
        
        # Create VectorData objects
        vector_data_list = []
        for i, embedding in enumerate(vectors):
            vector_id = ids[i] if ids and i < len(ids) else f"vec_{self._get_next_vector_id()}"
            vector_metadata = metadata[i] if metadata and i < len(metadata) else None
            
            vector_data = VectorData(
                id=vector_id,
                embedding=embedding.tolist(),
                metadata=vector_metadata
            )
            vector_data_list.append(vector_data)
        
        # Use advanced routing for decision
        if use_gpu is None:
            # Create routing context
            context = RoutingContext(
                batch_size=len(vector_data_list),
                k_value=0,  # Not applicable for add
                vector_dimension=self.config.dimension,
                dataset_size=self._get_total_dataset_size(),
                gpu_available=self.gpu_index is not None,
                gpu_memory_info=gpu_manager.get_gpu_memory_info(0) if self.gpu_index else None,
                gpu_utilization=gpu_manager.get_gpu_utilization(0) if self.gpu_index else None,
                operation_type="add"
            )
            
            # Get routing decision
            decision, reasoning = self.router.route_operation(context)
            logger.debug(f"Add routing decision: {decision.value} - {reasoning}")
        else:
            # Forced decision
            decision = RoutingDecision.GPU if use_gpu and self.gpu_index else RoutingDecision.CPU
            reasoning = f"Forced {'GPU' if decision == RoutingDecision.GPU else 'CPU'}"
        
        # Profile the operation
        profile_id = profiler.start_profile("add_vectors", {
            'batch_size': len(vector_data_list),
            'decision': decision.value,
            'reasoning': reasoning
        })
        
        try:
            # Execute operation based on decision
            if decision == RoutingDecision.GPU and self.gpu_index:
                try:
                    result = self.gpu_index.add_vectors(vector_data_list)
                    self.router.record_performance(decision, 0, True, context)
                    return result
                except Exception as e:
                    logger.warning(f"GPU add failed, falling back to CPU: {e}")
                    self.router.record_performance(RoutingDecision.GPU, 0, False, context)
                    decision = RoutingDecision.FALLBACK
            
            if decision in [RoutingDecision.CPU, RoutingDecision.FALLBACK]:
                result = self.cpu_index.add_vectors(vector_data_list)
                final_decision = RoutingDecision.CPU if decision == RoutingDecision.CPU else RoutingDecision.FALLBACK
                self.router.record_performance(final_decision, 0, True, context)
                return result
            
        finally:
            profiler.end_profile(profile_id)
        
        raise RuntimeError("Unexpected routing state")
    
    def _get_total_dataset_size(self) -> int:
        """Get total dataset size across all indexes."""
        total = self.cpu_index.total_vectors
        if self.gpu_index:
            total += self.gpu_index.total_vectors
        return total
    
    def search(self, 
              query_vectors: Union[List[List[float]], np.ndarray], 
              k: int = 10,
              use_gpu: Optional[bool] = None) -> Union[SearchResponse, List[SearchResponse]]:
        """
        Search for similar vectors.
        
        Args:
            query_vectors: Query vectors to search for
            k: Number of results to return
            use_gpu: Force GPU usage (None = auto-routing)
            
        Returns:
            SearchResponse or list of SearchResponse objects
        """
        # Convert to numpy array if needed
        if isinstance(query_vectors, list):
            query_vectors = np.array(query_vectors, dtype=np.float32)
        
        # Validate dimensions
        if query_vectors.shape[1] != self.config.dimension:
            raise ValueError(f"Query vector dimension mismatch: expected {self.config.dimension}, got {query_vectors.shape[1]}")
        
        # Use advanced routing for decision
        if use_gpu is None:
            # Create routing context
            context = RoutingContext(
                batch_size=len(query_vectors),
                k_value=k,
                vector_dimension=self.config.dimension,
                dataset_size=self._get_total_dataset_size(),
                gpu_available=self.gpu_index is not None,
                gpu_memory_info=gpu_manager.get_gpu_memory_info(0) if self.gpu_index else None,
                gpu_utilization=gpu_manager.get_gpu_utilization(0) if self.gpu_index else None,
                operation_type="search"
            )
            
            # Get routing decision
            decision, reasoning = self.router.route_operation(context)
            logger.debug(f"Search routing decision: {decision.value} - {reasoning}")
        else:
            # Forced decision
            decision = RoutingDecision.GPU if use_gpu and self.gpu_index else RoutingDecision.CPU
            reasoning = f"Forced {'GPU' if decision == RoutingDecision.GPU else 'CPU'}"
        
        # Profile the operation
        profile_id = profiler.start_profile("search_vectors", {
            'batch_size': len(query_vectors),
            'k': k,
            'decision': decision.value,
            'reasoning': reasoning
        })
        
        try:
            # Execute operation based on decision
            if decision == RoutingDecision.GPU and self.gpu_index:
                try:
                    start_time = time.time()
                    responses = self.gpu_index.search(query_vectors, k)
                    operation_time = (time.time() - start_time) * 1000
                    self.router.record_performance(decision, operation_time, True, context)
                    self._update_search_stats(responses)
                    return self._format_search_result(responses)
                except Exception as e:
                    logger.warning(f"GPU search failed, falling back to CPU: {e}")
                    self.router.record_performance(RoutingDecision.GPU, 0, False, context)
                    decision = RoutingDecision.FALLBACK
            
            if decision in [RoutingDecision.CPU, RoutingDecision.FALLBACK]:
                start_time = time.time()
                responses = self.cpu_index.search(query_vectors, k)
                operation_time = (time.time() - start_time) * 1000
                final_decision = RoutingDecision.CPU if decision == RoutingDecision.CPU else RoutingDecision.FALLBACK
                self.router.record_performance(final_decision, operation_time, True, context)
                self._update_search_stats(responses)
                return self._format_search_result(responses)
            
        finally:
            profiler.end_profile(profile_id)
        
        raise RuntimeError("Unexpected routing state")
    
    def _update_search_stats(self, responses):
        """Update search statistics."""
        self.total_queries += len(responses)
        for response in responses:
            self.total_search_time_ms += response.search_time_ms
            
            # Update CPU/GPU query counts
            if "gpu" in response.index_used:
                self.gpu_queries += 1
            else:
                self.cpu_queries += 1
    
    def _format_search_result(self, responses):
        """Format search result for return."""
        if len(responses) == 1:
            return responses[0]
        return responses
    
    def train(self, vectors: Optional[Union[List[List[float]], np.ndarray]] = None, 
               use_gpu: Optional[bool] = None) -> None:
        """
        Train the index (required for IVF indexes).
        
        Args:
            vectors: Training vectors (optional, will use existing vectors if not provided)
            use_gpu: Force GPU usage (None = auto-detect)
        """
        if self.config.index_type != "ivf":
            logger.info("Training not required for non-IVF indexes")
            return
        
        if vectors is not None:
            if isinstance(vectors, list):
                vectors = np.array(vectors, dtype=np.float32)
            
            # Decide where to train
            should_use_gpu = self._should_use_gpu_for_add(len(vectors), use_gpu)
            
            if should_use_gpu and self.gpu_index:
                try:
                    self.gpu_index.train(vectors)
                    logger.info("Trained GPU IVF index")
                except Exception as e:
                    logger.warning(f"GPU training failed, using CPU: {e}")
                    self.cpu_index.train(vectors)
            else:
                self.cpu_index.train(vectors)
        elif self.cpu_index.total_vectors > 0:
            logger.info("Using existing vectors for training")
            logger.warning("Automatic training from existing vectors not implemented in Phase 2")
        else:
            logger.warning("No vectors available for training")
    
    def _get_next_vector_id(self) -> int:
        """Get the next available vector ID across both indexes."""
        cpu_count = self.cpu_index.total_vectors
        gpu_count = self.gpu_index.total_vectors if self.gpu_index else 0
        return max(cpu_count, gpu_count)
    
    def get_routing_stats(self) -> Dict[str, Any]:
        """Get comprehensive routing statistics."""
        return self.router.get_routing_stats()
    
    def set_routing_strategy(self, strategy: str) -> None:
        """Set the routing strategy."""
        self.router.set_strategy(strategy)
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance profiling statistics."""
        return profiler.get_all_stats()
    
    def optimize_performance(self, operation_type: str = "search") -> Dict[str, Any]:
        """Run performance optimization."""
        if operation_type == "batch_size":
            # Optimize batch size
            test_sizes = [16, 32, 64, 128, 256]
            result = optimizer.optimize_batch_size(
                lambda data: self.search(data, k=10),
                test_sizes
            )
        elif operation_type == "k_value":
            # Optimize k value
            test_k_values = [10, 25, 50, 100, 200]
            query_data = np.random.random((50, self.config.dimension)).astype(np.float32)
            result = optimizer.optimize_k_value(
                lambda q, k: self.search(q, k=k),
                test_k_values,
                query_data
            )
        else:
            result = {"error": f"Unknown optimization type: {operation_type}"}
        
        return result
    
    def export_performance_data(self, filepath: str, format: str = 'json') -> None:
        """Export performance data to file."""
        profiler.export_profiles(filepath, format)
    
    def reset_routing_metrics(self) -> None:
        """Reset routing metrics."""
        self.router.reset_metrics()
        profiler.clear_profiles()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get database statistics."""
        cpu_stats = self.cpu_index.get_stats()
        
        stats = {
            "config": self.config.dict(),
            "cpu_index": cpu_stats,
            "total_queries": self.total_queries,
            "total_search_time_ms": self.total_search_time_ms,
            "avg_query_time_ms": self.total_search_time_ms / max(1, self.total_queries),
            "gpu_queries": self.gpu_queries,
            "cpu_queries": self.cpu_queries,
            "gpu_usage_percent": (self.gpu_queries / max(1, self.total_queries)) * 100,
            "phase": "2 (CPU + GPU)" if self.gpu_index else "2 (CPU only)"
        }
        
        # Add GPU stats if available
        if self.gpu_index:
            gpu_stats = self.gpu_index.get_stats()
            stats["gpu_index"] = gpu_stats
        
        return stats
    
    def get_gpu_info(self) -> Optional[Dict[str, Any]]:
        """Get GPU information."""
        if not self.gpu_index:
            return None
        return gpu_manager.get_gpu_info(0)
    
    def switch_gpu(self, enabled: bool) -> None:
        """Enable or disable GPU usage."""
        if enabled and not self.gpu_index:
            if check_gpu_availability():
                try:
                    self.gpu_index = GPUIndex(self.config, gpu_id=0)
                    logger.info("GPU enabled")
                except Exception as e:
                    logger.error(f"Failed to enable GPU: {e}")
                    raise RuntimeError(f"Failed to enable GPU: {e}")
            else:
                raise RuntimeError("GPU not available")
        elif not enabled and self.gpu_index:
            self.gpu_index = None
            logger.info("GPU disabled")
    
    def save_index(self, filepath: str, use_gpu: Optional[bool] = None) -> None:
        """Save the index to disk."""
        import faiss
        
        # Decide which index to save
        if use_gpu is None:
            use_gpu = self.gpu_index is not None
        
        if use_gpu and self.gpu_index:
            # Save GPU index (convert to CPU first)
            cpu_index = faiss.index_gpu_to_cpu(self.gpu_index.index)
            faiss.write_index(cpu_index, filepath)
            logger.info(f"Saved GPU index to {filepath}")
        else:
            # Save CPU index
            faiss.write_index(self.cpu_index.index, filepath)
            logger.info(f"Saved CPU index to {filepath}")
    
    def load_index(self, filepath: str, use_gpu: Optional[bool] = None) -> None:
        """Load the index from disk."""
        import faiss
        
        # Decide where to load
        if use_gpu is None:
            use_gpu = self.gpu_index is not None
        
        if use_gpu and self.gpu_index:
            # Load to GPU
            cpu_index = faiss.read_index(filepath)
            self.gpu_index.index = faiss.index_cpu_to_gpu(self.gpu_index.resources, 0, cpu_index)
            self.gpu_index.total_vectors = self.gpu_index.index.ntotal
            logger.info(f"Loaded index to GPU from {filepath}")
        else:
            # Load to CPU
            self.cpu_index.index = faiss.read_index(filepath)
            self.cpu_index.total_vectors = self.cpu_index.index.ntotal
            logger.info(f"Loaded index to CPU from {filepath}")
    
    def __len__(self) -> int:
        """Return number of vectors in the database."""
        total = self.cpu_index.total_vectors
        if self.gpu_index:
            total += self.gpu_index.total_vectors
        return total
    
    def __repr__(self) -> str:
        """String representation of database."""
        gpu_status = "GPU" if self.gpu_index else "CPU only"
        return f"HybridVectorDB(vectors={len(self)}, dimension={self.config.dimension}, mode={gpu_status}, strategy={self.router.current_strategy})"
