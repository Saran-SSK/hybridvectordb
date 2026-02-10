"""
GPU utilities and management for HybridVectorDB.
"""

import logging
import warnings
from typing import Optional, List, Dict, Any
import numpy as np

logger = logging.getLogger(__name__)

# Try to import GPU-related packages
try:
    import pynvml
    PYNVML_AVAILABLE = True
except ImportError:
    PYNVML_AVAILABLE = False
    warnings.warn("pynvml not available. GPU monitoring will be limited.")

try:
    import faiss
    FAISS_GPU_AVAILABLE = hasattr(faiss, 'StandardGpuResources')
except ImportError:
    FAISS_GPU_AVAILABLE = False
    warnings.warn("FAISS GPU not available. GPU acceleration disabled.")


class GPUManager:
    """Manages GPU resources and availability detection."""
    
    def __init__(self):
        """Initialize GPU manager."""
        self._gpu_available = None
        self._gpu_count = None
        self._gpu_info = None
        self._initialized = False
        
    def _initialize(self):
        """Initialize GPU detection."""
        if self._initialized:
            return
            
        self._initialized = True
        
        # Check FAISS GPU availability
        if not FAISS_GPU_AVAILABLE:
            self._gpu_available = False
            logger.warning("FAISS GPU not available")
            return
            
        # Check NVML availability
        if PYNVML_AVAILABLE:
            try:
                pynvml.nvmlInit()
                self._gpu_count = pynvml.nvmlDeviceGetCount()
                self._gpu_available = self._gpu_count > 0
                
                if self._gpu_available:
                    self._gpu_info = []
                    for i in range(self._gpu_count):
                        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                        info = {
                            'id': i,
                            'name': pynvml.nvmlDeviceGetName(handle).decode('utf-8'),
                            'memory_total': pynvml.nvmlDeviceGetMemoryInfo(handle).total,
                            'memory_free': pynvml.nvmlDeviceGetMemoryInfo(handle).free,
                            'memory_used': pynvml.nvmlDeviceGetMemoryInfo(handle).used,
                        }
                        self._gpu_info.append(info)
                    
                    logger.info(f"Found {self._gpu_count} GPU(s)")
                    for gpu in self._gpu_info:
                        logger.info(f"  GPU {gpu['id']}: {gpu['name']} "
                                  f"({gpu['memory_free']/1024**3:.1f}GB free / "
                                  f"{gpu['memory_total']/1024**3:.1f}GB total)")
                else:
                    logger.warning("No GPUs found")
                    
            except Exception as e:
                logger.error(f"Failed to initialize GPU detection: {e}")
                self._gpu_available = False
        else:
            # Fallback: try to create a GPU index to test availability
            try:
                test_resources = faiss.StandardGpuResources()
                self._gpu_available = True
                self._gpu_count = 1
                logger.info("GPU available (NVML monitoring disabled)")
            except Exception as e:
                logger.error(f"GPU not available: {e}")
                self._gpu_available = False
    
    @property
    def gpu_available(self) -> bool:
        """Check if GPU is available."""
        self._initialize()
        return self._gpu_available
    
    @property
    def gpu_count(self) -> int:
        """Get number of available GPUs."""
        self._initialize()
        return self._gpu_count or 0
    
    def get_gpu_info(self, gpu_id: int = 0) -> Optional[Dict[str, Any]]:
        """Get information about a specific GPU."""
        self._initialize()
        if not self._gpu_info or gpu_id >= len(self._gpu_info):
            return None
        return self._gpu_info[gpu_id].copy()
    
    def get_gpu_utilization(self, gpu_id: int = 0) -> Optional[float]:
        """Get current GPU utilization percentage."""
        if not PYNVML_AVAILABLE or not self.gpu_available:
            return None
            
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            return util.gpu
        except Exception as e:
            logger.error(f"Failed to get GPU utilization: {e}")
            return None
    
    def get_gpu_memory_info(self, gpu_id: int = 0) -> Optional[Dict[str, int]]:
        """Get GPU memory information in bytes."""
        if not PYNVML_AVAILABLE or not self.gpu_available:
            return None
            
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            return {
                'total': mem_info.total,
                'free': mem_info.free,
                'used': mem_info.used
            }
        except Exception as e:
            logger.error(f"Failed to get GPU memory info: {e}")
            return None
    
    def get_gpu_memory_usage(self, gpu_id: int = 0) -> Optional[float]:
        """Get GPU memory usage as percentage."""
        mem_info = self.get_gpu_memory_info(gpu_id)
        if not mem_info:
            return None
        return (mem_info['used'] / mem_info['total']) * 100.0
    
    def can_use_gpu(self, memory_required: int = 0, gpu_id: int = 0) -> bool:
        """Check if GPU can be used for given memory requirement."""
        if not self.gpu_available:
            return False
            
        if memory_required <= 0:
            return True
            
        mem_info = self.get_gpu_memory_info(gpu_id)
        if not mem_info:
            return False
            
        return mem_info['free'] >= memory_required
    
    def estimate_memory_usage(self, num_vectors: int, dimension: int, 
                           index_type: str = "flat") -> int:
        """Estimate GPU memory usage for given vectors."""
        # Basic estimation (this is approximate and depends on FAISS implementation)
        if index_type == "flat":
            # Flat index: vectors + overhead
            vector_bytes = num_vectors * dimension * 4  # float32
            overhead = vector_bytes * 0.1  # 10% overhead estimate
            return int(vector_bytes + overhead)
        elif index_type == "ivf":
            # IVF index: vectors + centroids + overhead
            vector_bytes = num_vectors * dimension * 4
            nlist = min(100, max(10, num_vectors // 1000))  # heuristic
            centroid_bytes = nlist * dimension * 4
            overhead = (vector_bytes + centroid_bytes) * 0.2  # 20% overhead
            return int(vector_bytes + centroid_bytes + overhead)
        else:
            # Default estimate
            return int(num_vectors * dimension * 4 * 1.5)


# Global GPU manager instance
gpu_manager = GPUManager()


def check_gpu_availability() -> bool:
    """Convenience function to check GPU availability."""
    return gpu_manager.gpu_available


def get_gpu_resources(gpu_id: int = 0):
    """Get FAISS GPU resources for a specific GPU."""
    if not gpu_manager.gpu_available:
        raise RuntimeError("GPU not available")
    
    try:
        resources = faiss.StandardGpuResources()
        # Configure resources
        resources.setTempMemory(256 * 1024 * 1024)  # 256MB temp memory
        return resources
    except Exception as e:
        logger.error(f"Failed to create GPU resources: {e}")
        raise RuntimeError(f"Failed to initialize GPU resources: {e}")


def create_gpu_index(dimension: int, metric_type: str = "l2", 
                    index_type: str = "flat", gpu_id: int = 0):
    """Create a GPU index."""
    if not gpu_manager.gpu_available:
        raise RuntimeError("GPU not available")
    
    try:
        resources = get_gpu_resources(gpu_id)
        
        # Convert metric type
        if metric_type == "l2":
            metric = faiss.METRIC_L2
        elif metric_type == "inner_product":
            metric = faiss.METRIC_INNER_PRODUCT
        elif metric_type == "cosine":
            metric = faiss.METRIC_INNER_PRODUCT
        else:
            raise ValueError(f"Unsupported metric type: {metric_type}")
        
        # Create index based on type
        if index_type == "flat":
            cpu_index = faiss.IndexFlat(dimension, metric)
            gpu_index = faiss.index_cpu_to_gpu(resources, gpu_id, cpu_index)
        elif index_type == "ivf":
            quantizer = faiss.IndexFlat(dimension, metric)
            cpu_index = faiss.IndexIVFFlat(quantizer, dimension, 100, metric)
            gpu_index = faiss.index_cpu_to_gpu(resources, gpu_id, cpu_index)
        else:
            raise ValueError(f"Unsupported index type: {index_type}")
        
        logger.info(f"Created GPU {index_type} index on GPU {gpu_id}")
        return gpu_index, resources
        
    except Exception as e:
        logger.error(f"Failed to create GPU index: {e}")
        raise RuntimeError(f"Failed to create GPU index: {e}")
