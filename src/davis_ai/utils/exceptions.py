# src/davis_ai/utils/exceptions.py

"""
Custom exceptions for the Davis AI project for robust error handling.
"""

class DavisError(Exception):
    """Base exception class for all custom errors in the Davis project."""
    pass

class DavisConfigError(DavisError):
    """Raised when a required configuration value is missing or invalid."""
    pass

class DavisMoveEncodingError(DavisError):
    """Raised when a chess move cannot be encoded into a policy index."""
    pass

class DavisMCTSError(DavisError):
    """Raised for errors occurring during the MCTS search process."""
    pass

class DavisDataError(DavisError):
    """Raised for issues related to data loading, processing, or validation."""
    pass