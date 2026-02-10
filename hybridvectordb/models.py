"""
Data models for HybridVectorDB.
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field, validator
import numpy as np


class VectorData(BaseModel):
    """Model for vector data with metadata."""
    
    id: str = Field(..., description="Unique identifier for the vector")
    embedding: List[float] = Field(..., description="Vector embedding")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Additional metadata")
    
    @validator('embedding')
    def validate_embedding(cls, v):
        if not isinstance(v, list):
            raise ValueError("Embedding must be a list of floats")
        if len(v) == 0:
            raise ValueError("Embedding cannot be empty")
        return v
    
    def to_numpy(self) -> np.ndarray:
        """Convert embedding to numpy array."""
        return np.array(self.embedding, dtype=np.float32)


class SearchResult(BaseModel):
    """Model for search results."""
    
    id: str = Field(..., description="Vector ID")
    distance: float = Field(..., description="Distance to query vector")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Vector metadata")


class SearchResponse(BaseModel):
    """Model for complete search response."""
    
    query_id: Optional[str] = Field(default=None, description="Query identifier")
    results: List[SearchResult] = Field(..., description="Search results")
    total_results: int = Field(..., description="Total number of results returned")
    search_time_ms: float = Field(..., description="Search time in milliseconds")
    index_used: str = Field(..., description="Which index was used for search")
