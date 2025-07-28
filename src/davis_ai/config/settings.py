# src/davis_ai/config/settings.py

"""
Centralized Configuration for Davis AI.

This file contains all hyperparameters and settings for training, evaluation,
and self-play. Adjusting these values will directly impact the AI's
learning process and performance.
"""

# A dictionary is used to allow for easy passing of the entire config object.
import os


DAVIS_CONFIG = {
    # --- Path Configuration ---
    "data_directory": "data/",
    "model_directory": "data/models/",
    "games_directory": "data/games/",
    "processed_data_directory": "data/processed/",
    "best_model_filename": "best_model.pth",
    "candidate_model_filename": "candidate.pth",

    # --- Self-Play and MCTS Configuration ---
    "simulations_per_move": 400,        # MCTS simulations for each move during self-play.
    "mcts_c_puct": 4.0,                 # UCB formula exploration constant.
    "mcts_dirichlet_alpha": 0.3,        # Dirichlet noise alpha for root node exploration.
    "mcts_dirichlet_epsilon": 0.25,     # Weight of Dirichlet noise in the root policy.
    "self_play_move_limit": 250,        # Max moves in a self-play game to prevent infinite loops.
    "mcts_top_k_moves": 5,           # MCTS Pruning: Only search the top 5 moves from the policy.
    "num_parallel_games": os.cpu_count() or 4, # Parallel Self-Play: Use all available CPU cores.

    # --- Training Configuration ---
    "learning_rate": 2e-4,              # Initial learning rate for the optimizer.
    "weight_decay": 1e-4,               # Weight decay for regularization (AdamW).
    "training_epochs": 10,              # Number of epochs to train on a new batch of games.
    "batch_size": 1024,                 # Batch size for training the neural network.

    # --- ECR (Echo Chamber Reinforcement) Analyst Configuration ---
    "ecr_analyst_simulations": 800,     # Deeper MCTS search for post-game blunder analysis.

    # --- Evaluation Configuration ---
    "eval_num_games": 100,              # Number of games to play between candidate and best model.
    "eval_win_threshold": 0.55,         # Candidate must win >55% of decisive games to be promoted.

    # --- Main Loop Configuration ---
    "num_generations": 500,             # Total number of training generations to run.
    "games_per_generation": 5000,       # Number of self-play games to generate in each generation.
}