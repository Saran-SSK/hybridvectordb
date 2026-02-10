"""
HybridVectorDB - A hybrid CPU/GPU vector database with intelligent query routing.

Phase 5 Implementation:
- FAISS CPU index
- FAISS GPU index
- Python API with clean interface
- Advanced hybrid routing
- Dynamic threshold selection
- Performance optimization
- Adaptive learning
- C++ bindings with pybind11
- Zero-copy memory transfers
- SIMD optimizations
- High-performance threading
- Comprehensive benchmarking framework
- Load generation tools
- Performance visualization
- Automated testing suite
"""

from .core import HybridVectorDB
from .config import Config
from .models import VectorData, SearchResult
from .router import AdvancedRouter, RoutingDecision
from .profiler import profiler, optimizer

__version__ = "0.5.0"
__all__ = [
    "HybridVectorDB", 
    "Config", 
    "VectorData", 
    "SearchResult",
    "AdvancedRouter",
    "RoutingDecision",
    "profiler",
    "optimizer"
]

# Try to import C++ bindings
try:
    from . import _cpp
    __all__.extend([
        "create_vector_database",
        "optimization",
        "BenchmarkFramework",
        "BenchmarkConfig",
        "LoadGenerator",
        "LoadConfig",
        "PerformanceVisualizer",
        "AutomatedTestSuite"
    ])
    # Export C++ Config
    CppConfig = _cpp.Config
    __all__.append("CppConfig")
except ImportError:
    # C++ bindings not available
    pass
