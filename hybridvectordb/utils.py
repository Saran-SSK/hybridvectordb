"""
Utility functions for HybridVectorDB.
"""

import numpy as np
from typing import List, Union
import time
import logging

logger = logging.getLogger(__name__)


def validate_vectors(vectors: Union[List[List[float]], np.ndarray], dimension: int) -> np.ndarray:
    """
    Validate and convert vectors to numpy array.
    
    Args:
        vectors: Input vectors
        dimension: Expected dimension
        
    Returns:
        Validated numpy array
        
    Raises:
        ValidationError: If vectors are invalid
    """
    from .exceptions import ValidationError
    
    # Convert to numpy array
    if isinstance(vectors, list):
        vectors = np.array(vectors, dtype=np.float32)
    elif not isinstance(vectors, np.ndarray):
        raise ValidationError("Vectors must be list or numpy array")
    
    # Validate shape
    if vectors.ndim != 2:
        raise ValidationError(f"Vectors must be 2D array, got {vectors.ndim}D")
    
    if vectors.shape[1] != dimension:
        raise ValidationError(f"Vector dimension mismatch: expected {dimension}, got {vectors.shape[1]}")
    
    # Check for NaN or infinite values
    if np.any(~np.isfinite(vectors)):
        raise ValidationError("Vectors contain NaN or infinite values")
    
    return vectors


def timer(func):
    """Decorator to time function execution."""
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        logger.debug(f"{func.__name__} executed in {end_time - start_time:.4f}s")
        return result
    return wrapper


def setup_logging(level: str = "INFO") -> None:
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
