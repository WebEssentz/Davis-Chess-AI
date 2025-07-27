# tests/integration/test_game_flow.py

import os
from davis_ai.correction import ReinforcementLearner
from davis_ai.config import DAVIS_CONFIG

def test_full_learning_loop_one_generation(tmp_path):
    """
    Runs the entire reinforcement learning loop for a single generation
    to ensure all components integrate correctly.
    """
    # Create a temporary directory structure for this test
    data_dir = tmp_path / "data"
    model_dir = data_dir / "models"
    games_dir = data_dir / "games"
    processed_dir = data_dir / "processed"

    # Override the main config with a tiny configuration for a very fast test run
    test_config = DAVIS_CONFIG.copy()
    test_config.update({
        "data_directory": str(data_dir),
        "model_directory": str(model_dir),
        "games_directory": str(games_dir),
        "processed_data_directory": str(processed_dir),
        "num_generations": 1,
        "games_per_generation": 1,
        "simulations_per_move": 4, # Drastically reduced for speed
        "ecr_analyst_simulations": 4,
        "training_epochs": 1,
        "batch_size": 2,
        "eval_num_games": 2,
    })
    
    # Initialize and run the learner
    learner = ReinforcementLearner(config=test_config)
    learner.run_loop() # This is the core of the integration test

    # --- Assertions ---
    # Check that the necessary files and directories were created
    assert os.path.exists(model_dir)
    assert os.path.exists(os.path.join(model_dir, "best_model.pth"))
    
    # Check that a game record was created
    gen1_games_dir = os.path.join(games_dir, "gen_1")
    assert os.path.exists(gen1_games_dir)
    assert len(os.listdir(gen1_games_dir)) == 1

    # Check that processed data was created
    assert os.path.exists(os.path.join(processed_dir, "gen_1_data.pkl"))