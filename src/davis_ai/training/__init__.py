# src/davis_ai/training/__init__.py
"""
Davis AI Training Package

This package contains all modules related to the AI's learning loop,
including self-play, our novel Echo Chamber Reinforcement (ECR) data pipeline,
the model trainer, and the evaluation system.
"""

from .self_play import SelfPlayWorker, GameRecord, run_self_play_and_save
from .data_pipeline import ChessDataset, create_dataloader, analyze_and_enhance_game
from .trainer import Trainer, run_training_cycle
from .evaluation import Evaluator, run_evaluation

__all__ = [
    # from self_play
    "SelfPlayWorker",
    "GameRecord",
    "run_self_play_and_save",
    # from data_pipeline
    "ChessDataset",
    "create_dataloader",
    "analyze_and_enhance_game",
    # from trainer
    "Trainer",
    "run_training_cycle",
    # from evaluation
    "Evaluator",
    "run_evaluation"
]