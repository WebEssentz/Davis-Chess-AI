# src/davis_ai/correction/reinforcement_learner.py (Definitive Version)

import os
import pickle
from typing import Dict
import multiprocessing as mp
from functools import partial

from ..config import DAVIS_CONFIG
from ..engine import DavisChessModel, InferenceEngine
from ..training import (
    run_self_play_and_save,
    analyze_and_enhance_game,
    run_training_cycle,
    run_evaluation
)

# This helper function must be at the top level for multiprocessing
def play_one_game(game_num: int, model_path: str, save_dir: str, config: dict):
    """ Wrapper function for a single game of self-play. """
    print(f"  > Starting self-play game {game_num}/{config['games_per_generation']}...")
    # Suppress verbose output from workers by redirecting stdout temporarily
    # This keeps the main log clean.
    # old_stdout = sys.stdout
    # sys.stdout = open(os.devnull, 'w')
    run_self_play_and_save(model_path, save_dir, config)
    # sys.stdout = old_stdout

class ReinforcementLearner:
    """ The master class that orchestrates the entire reinforcement learning loop. """
    def __init__(self, config: Dict):
        self.config = config
        self._setup_directories()

    def _setup_directories(self):
        """ Ensures all necessary data directories exist. """
        os.makedirs(self.config["model_directory"], exist_ok=True)
        os.makedirs(self.config["games_directory"], exist_ok=True)
        os.makedirs(self.config["processed_data_directory"], exist_ok=True)

    def _get_initial_model(self) -> str:
        """ Ensures an initial model exists. If not, it creates a new one. """
        best_model_path = os.path.join(self.config["model_directory"], self.config["best_model_filename"])
        if not os.path.exists(best_model_path):
            print("No 'best_model.pth' found. Creating a new initial model.")
            initial_model = DavisChessModel() 
            initial_model.save_checkpoint(best_model_path, compile_for_inference=True)
        return best_model_path

    def run_loop(self, start_generation: int = 1):
        """ Starts and runs the main training loop with parallel self-play. """
        print("--- LAUNCHING DAVIS REINFORCEMENT LEARNER (HYPER-EFFICIENT) ---")
        best_model_path = self._get_initial_model()

        for generation in range(start_generation, self.config["num_generations"] + 1):
            print(f"\n\n{'='*20} GENERATION {generation}/{self.config['num_generations']} {'='*20}")
            
            # --- PHASE 1: Parallel Self-Play ---
            print(f"\n[PHASE 1] Starting {self.config['games_per_generation']} self-play games in parallel...")
            current_games_dir = os.path.join(self.config["games_directory"], f"gen_{generation}")
            num_workers = self.config.get('num_parallel_games', os.cpu_count() or 1)
            games_to_play = range(1, self.config['games_per_generation'] + 1)
            
            worker_func = partial(play_one_game, model_path=best_model_path, save_dir=current_games_dir, config=self.config)
            
            with mp.Pool(processes=num_workers) as pool:
                pool.map(worker_func, games_to_play)
            print("  > All self-play games have been completed.")

            # --- PHASE 2: Data Processing (All Generations) ---
            print(f"\n[PHASE 2] Processing game data from all generations with ECR...")
            all_training_samples = []
            analyst_engine = InferenceEngine(model_path=best_model_path)
            
            for gen_num in range(1, generation + 1):
                games_dir_to_process = os.path.join(self.config["games_directory"], f"gen_{gen_num}")
                if os.path.exists(games_dir_to_process):
                    game_files = [os.path.join(games_dir_to_process, f) for f in os.listdir(games_dir_to_process)]
                    for game_path in game_files:
                        with open(game_path, 'rb') as f:
                            game_record = pickle.load(f)
                        enhanced_samples = analyze_and_enhance_game(game_record, analyst_engine, self.config)
                        all_training_samples.extend(enhanced_samples)

            processed_data_path = os.path.join(self.config['processed_data_directory'], f"gen_{generation}_data.pkl")
            with open(processed_data_path, 'wb') as f:
                pickle.dump(all_training_samples, f)
            print(f"  > Total training samples aggregated: {len(all_training_samples)}")

            # --- PHASE 3: Training ---
            print(f"\n[PHASE 3] Training a new candidate model...")
            candidate_model_path = os.path.join(self.config["model_directory"], self.config["candidate_model_filename"])
            run_training_cycle(best_model_path, processed_data_path, candidate_model_path, self.config)

            # --- PHASE 4: Evaluation ---
            print(f"\n[PHASE 4] Evaluating candidate model against the current best...")
            is_candidate_better = run_evaluation(candidate_model_path, best_model_path, self.config)
            
            # --- PHASE 5: Promotion/Correction ---
            if is_candidate_better:
                print("\n[PHASE 5] PROMOTION: New model is superior. Compiling and updating 'best_model.pth'.")
                winning_model = DavisChessModel.load_checkpoint(candidate_model_path)
                winning_model.save_checkpoint(best_model_path, compile_for_inference=True)
            else:
                print("\n[PHASE 5] CORRECTION: New model is not better. Discarding candidate.")

        print("\n--- DAVIS REINFORCEMENT LEARNING COMPLETE ---")