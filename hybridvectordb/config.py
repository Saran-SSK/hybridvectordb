"""
Configuration module for HybridVectorDB.
"""

from typing import Optional
from pydantic import BaseModel, Field


class Config(BaseModel):
    """Configuration for HybridVectorDB."""
    
    # Index parameters
    dimension: int = Field(..., description="Dimension of vectors")
    index_type: str = Field(default="flat", description="Index type: 'flat', 'ivf'")
    metric_type: str = Field(default="l2", description="Distance metric: 'l2', 'inner_product', 'cosine'")
    
    # IVF parameters (only used if index_type='ivf')
    nlist: int = Field(default=100, description="Number of IVF clusters")
    nprobe: int = Field(default=10, description="Number of clusters to search")
    
    # Routing thresholds (for future hybrid implementation)
    batch_threshold: int = Field(default=32, description="Batch size threshold for CPU vs GPU")
    k_threshold: int = Field(default=50, description="Top-k threshold for CPU vs GPU")
    dataset_threshold: int = Field(default=100000, description="Dataset size threshold for GPU")
    gpu_util_limit: float = Field(default=0.8, description="GPU utilization limit")
    
    # Performance settings
    use_gpu: bool = Field(default=False, description="Enable GPU acceleration (Phase 2)")
    
    class Config:
        validate_assignment = True
