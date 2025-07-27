# src/davis_ai/correction/reinforcement_learner.py

import os
import shutil
import pickle
from typing import Dict

from ..config import DAVIS_CONFIG
from ..engine import DavisChessModel, InferenceEngine
from ..training import (
    run_self_play_and_save,
    analyze_and_enhance_game,
    run_training_cycle,
    run_evaluation
)

class ReinforcementLearner:
    """
    The master class that orchestrates the entire reinforcement learning loop for Davis.
    It manages generations of self-play, training, and evaluation to continuously
    improve the AI model.
    """
    def __init__(self, config: Dict):
        self.config = config
        self._setup_directories()

    def _setup_directories(self):
        """Ensures all necessary data directories exist."""
        os.makedirs(self.config["model_directory"], exist_ok=True)
        os.makedirs(self.config["games_directory"], exist_ok=True)
        os.makedirs(self.config["processed_data_directory"], exist_ok=True)

    def _get_initial_model(self) -> str:
        """
        Ensures an initial model exists. If not, it creates and saves a new one.
        """
        best_model_path = os.path.join(self.config["model_directory"], self.config["best_model_filename"])
        if not os.path.exists(best_model_path):
            print("No 'best_model.pth' found. Creating a new initial model.")
            # Use default architecture for the first model.
            initial_model = DavisChessModel() 
            initial_model.save_checkpoint(best_model_path)
        return best_model_path

    def run_loop(self):
        """
        Starts and runs the main training loop for the configured number of generations.
        """
        print("--- LAUNCHING DAVIS REINFORCEMENT LEARNER ---")
        best_model_path = self._get_initial_model()

        for generation in range(self.config["num_generations"]):
            print(f"\n\n{'='*20} GENERATION {generation+1}/{self.config['num_generations']} {'='*20}")
            
            # --- 1. Self-Play Phase ---
            print(f"\n[PHASE 1] Starting {self.config['games_per_generation']} self-play games...")
            current_games_dir = os.path.join(self.config["games_directory"], f"gen_{generation+1}")
            # For simplicity, we run games sequentially. For massive speedups, this would use multiprocessing.
            for i in range(self.config['games_per_generation']):
                print(f"  > Playing game {i+1}/{self.config['games_per_generation']}...")
                run_self_play_and_save(best_model_path, current_games_dir, self.config)
            
            # --- 2. Data Processing Phase with ECR ---
            print(f"\n[PHASE 2] Processing game data with Echo Chamber Reinforcement (ECR)...")
            all_training_samples = []
            analyst_engine = InferenceEngine(model_path=best_model_path)
            game_files = [os.path.join(current_games_dir, f) for f in os.listdir(current_games_dir)]
            
            for game_path in game_files:
                with open(game_path, 'rb') as f:
                    game_record = pickle.load(f)
                enhanced_samples = analyze_and_enhance_game(game_record, analyst_engine, self.config)
                all_training_samples.extend(enhanced_samples)
                
            processed_data_path = os.path.join(self.config['processed_data_directory'], f"gen_{generation+1}_data.pkl")
            with open(processed_data_path, 'wb') as f:
                pickle.dump(all_training_samples, f)
            print(f"  > Total training samples, including ECR: {len(all_training_samples)}")

            # --- 3. Training Phase ---
            print(f"\n[PHASE 3] Training a new candidate model...")
            candidate_model_path = os.path.join(self.config["model_directory"], self.config["candidate_model_filename"])
            run_training_cycle(best_model_path, processed_data_path, candidate_model_path, self.config)

            # --- 4. Evaluation Phase ---
            print(f"\n[PHASE 4] Evaluating candidate model against the current best...")
            is_candidate_better = run_evaluation(candidate_model_path, best_model_path, self.config)
            
            # --- 5. Promotion/Correction Phase ---
            if is_candidate_better:
                print("\n[PHASE 5] PROMOTION: New model is superior. Updating 'best_model.pth'.")
                shutil.copy(candidate_model_path, best_model_path)
            else:
                print("\n[PHASE 5] CORRECTION: New model is not better. Discarding candidate.")

        print("\n--- DAVIS REINFORCEMENT LEARNING COMPLETE ---")


if __name__ == '__main__':
    print("\n--- Testing Reinforcement Learner End-to-End Loop ---")
    
    # Create a temporary config for a very short test run
    test_config = DAVIS_CONFIG.copy()
    test_config.update({
        "data_directory": "temp_test_data/",
        "model_directory": "temp_test_data/models/",
        "games_directory": "temp_test_data/games/",
        "processed_data_directory": "temp_test_data/processed/",
        "num_generations": 1,
        "games_per_generation": 2,
        "simulations_per_move": 8,
        "ecr_analyst_simulations": 10,
        "training_epochs": 1,
        "batch_size": 4,
        "eval_num_games": 2,
    })
    
    # Run the learner
    learner = ReinforcementLearner(config=test_config)
    learner.run_loop()
    
    print("\nReinforcement Learner loop completed a full generation.")
    
    # Clean up the temporary directory
    shutil.rmtree(test_config["data_directory"])
    print("Cleaned up temporary test directory.")