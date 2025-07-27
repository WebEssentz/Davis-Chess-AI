# src/davis_ai/board/__init__.py
"""
Davis AI Board Package

This package contains the high-performance board representation for Davis.
It is designed for speed and efficiency within the MCTS and self-play loops.
The core `Board` class handles state, move generation, and conversion to
neural network tensors.
"""

# Expose the primary Board class for convenient importing
from .board_state import Board

__all__ = [
    "Board"
]