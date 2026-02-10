"""
Custom exceptions for HybridVectorDB.
"""


class HybridVectorDBError(Exception):
    """Base exception for HybridVectorDB."""
    pass


class ConfigurationError(HybridVectorDBError):
    """Raised when configuration is invalid."""
    pass


class IndexError(HybridVectorDBError):
    """Raised when index operations fail."""
    pass


class ValidationError(HybridVectorDBError):
    """Raised when data validation fails."""
    pass


class SearchError(HybridVectorDBError):
    """Raised when search operations fail."""
    pass
