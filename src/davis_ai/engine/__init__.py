# src/davis_ai/engine/__init__.py
"""
Davis AI Engine Package

This package houses the core decision-making components of Davis:
the deep learning model, the Monte Carlo Tree Search (MCTS) algorithm,
and the inference mechanism.

Developed by Ossie, this engine is the strategic heart of Davis.
"""

# Expose key classes for convenient imports from this package
from .model import DavisChessModel
from .inference import InferenceEngine
from .mcts import MonteCarloTreeSearch, MCTSNode

__all__ = [
    "DavisChessModel",
    "InferenceEngine",
    "MonteCarloTreeSearch",
    "MCTSNode"
]