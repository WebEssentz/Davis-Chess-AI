# The "First Spark" Configuration for Free Tier Colab
DAVIS_CONFIG = {
    # --- Path Configuration (Leave as is) ---
    "data_directory": "data/",
    "model_directory": "data/models/",
    "games_directory": "data/games/",
    "processed_data_directory": "data/processed/",
    "best_model_filename": "best_model.pth",
    "candidate_model_filename": "candidate.pth",

    # --- Self-Play and MCTS Configuration ---
    "simulations_per_move": 50,         # Drastically reduced. Fast, but enough for basic patterns.
    "mcts_c_puct": 2.5,                 # A bit less exploration to focus the learning.
    "mcts_dirichlet_alpha": 0.3,
    "mcts_dirichlet_epsilon": 0.25,
    "self_play_move_limit": 100,        # Shorter games to get more data faster.

    # --- Training Configuration ---
    "learning_rate": 1e-3,              # A slightly higher LR for faster learning on a small dataset.
    "weight_decay": 1e-4,
    "training_epochs": 5,               # Fewer epochs are needed for a smaller dataset.
    "batch_size": 256,                  # Smaller batch size for less data.

    # --- ECR (Echo Chamber Reinforcement) Analyst Configuration ---
    "ecr_analyst_simulations": 100,     # Still 2x the self-play sims.

    # --- Evaluation Configuration ---
    "eval_num_games": 20,               # Enough games to get a statistically meaningful result.
    "eval_win_threshold": 0.55,         # Keep the professional threshold.

    # --- Main Loop Configuration ---
    "num_generations": 5,               # Your proposed 5 generations.
    "games_per_generation": 20,         # More than 10, to get a decent training batch.
}