# src/davis_ai/correction/__init__.py
"""
Davis AI Correction and Learning Loop Package

This package contains the master controller for the entire reinforcement
learning process. The `ReinforcementLearner` class orchestrates the
cycle of self-play, training, and evaluation.
"""

from .reinforcement_learner import ReinforcementLearner

__all__ = ["ReinforcementLearner"]