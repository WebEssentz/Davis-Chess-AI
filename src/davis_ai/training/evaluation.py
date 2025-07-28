# src/davis_ai/training/evaluation.py

import chess
import os
from typing import Dict

from ..engine import InferenceEngine, MonteCarloTreeSearch
from ..board import Board

class Evaluator:
    """
    Pits two model versions against each other to determine which is stronger.
    """
    def __init__(self, candidate_engine: InferenceEngine, best_engine: InferenceEngine, config: dict):
        # CORRECTED: Pass the entire config dictionary.
        self.candidate_mcts = MonteCarloTreeSearch(candidate_engine, config=config)
        self.best_mcts = MonteCarloTreeSearch(best_engine, config=config)
        self.num_games = config.get('eval_num_games', 50)
        self.simulations_per_move = config.get('simulations_per_move', 100)
        self.move_limit = config.get('move_limit', 200)

    def evaluate(self) -> Dict[str, float]:
        """
        Runs a match between the two models.

        Returns:
            A dictionary with the score results.
        """
        scores = {"candidate_wins": 0, "best_wins": 0, "draws": 0}
        
        # Play half the games with candidate as White
        print(f"\n--- Playing {self.num_games // 2} games with Candidate as White ---")
        for i in range(self.num_games // 2):
            outcome = self._play_single_game(white_player=self.candidate_mcts, black_player=self.best_mcts)
            self._update_scores("candidate", scores, outcome)
            print(f"Game {i+1}: Candidate (W) vs Best (B) -> Outcome: {outcome}")

        # Play the other half with candidate as Black
        print(f"\n--- Playing {self.num_games // 2} games with Candidate as Black ---")
        for i in range(self.num_games // 2):
            outcome = self._play_single_game(white_player=self.best_mcts, black_player=self.candidate_mcts)
            self._update_scores("candidate", scores, -outcome) # Outcome is from White's perspective
            print(f"Game {i+1+(self.num_games//2)}: Best (W) vs Candidate (B) -> Outcome: {outcome}")

        return scores

    def _play_single_game(self, white_player: MonteCarloTreeSearch, black_player: MonteCarloTreeSearch) -> float:
        """Plays one game and returns the outcome from White's perspective (1.0, -1.0, 0.0)."""
        board = Board()
        move_count = 0
        while not board.is_game_over() and move_count < self.move_limit:
            player = white_player if board.turn == chess.WHITE else black_player
            
            # During evaluation, we are deterministic. Temperature is 0.
            root_node = player.search(board, self.simulations_per_move)
            move = player.select_move(root_node, temperature=0)
            
            board = board.apply_move(move)
            move_count += 1
            
        return board.get_game_outcome() or 0.0

    @staticmethod
    def _update_scores(candidate_player_id: str, scores: Dict, outcome: float):
        """Updates the score dictionary based on the game outcome."""
        if outcome == 1.0:
            scores[f"{candidate_player_id}_wins"] += 1
        elif outcome == -1.0:
            scores["best_wins"] += 1
        else:
            scores["draws"] += 1

def run_evaluation(candidate_path: str, best_path: str, config: dict) -> bool:
    """
    Main entry point to run an evaluation match and decide if the candidate model is better.
    """
    print("\n--- Starting Model Evaluation ---")
    print(f"Candidate: {candidate_path}")
    print(f"Current Best: {best_path}")

    candidate_engine = InferenceEngine(candidate_path)
    best_engine = InferenceEngine(best_path)
    
    evaluator = Evaluator(candidate_engine, best_engine, config)
    scores = evaluator.evaluate()
    
    print("\n--- Evaluation Results ---")
    print(scores)
    
    total_games = scores["candidate_wins"] + scores["best_wins"]
    if total_games == 0:
        print("No decisive games. Keeping current model.")
        return False
        
    win_rate = scores["candidate_wins"] / total_games
    win_threshold = config.get('eval_win_threshold', 0.55)
    
    print(f"Candidate Win Rate (vs Best, excluding draws): {win_rate:.2%}")
    print(f"Required Win Rate Threshold: {win_threshold:.2%}")
    
    if win_rate > win_threshold:
        print("\nVERDICT: Candidate model is significantly better. Promoting to new 'best'.")
        return True
    else:
        print("\nVERDICT: Candidate model is not better. Keeping current 'best'.")
        return False

if __name__ == '__main__':
    # --- Example Usage ---
    print("\n--- Testing Evaluator ---")
    
    # 1. Create two dummy models to act as candidate and best
    from ..engine import DavisChessModel
    best_model_file = "best.pth"
    candidate_model_file = "candidate.pth"
    model1 = DavisChessModel(input_channels=17, num_residual_blocks=1, num_filters=8)
    model1.save_checkpoint(best_model_file)
    model2 = DavisChessModel(input_channels=17, num_residual_blocks=1, num_filters=8) # Same architecture
    model2.save_checkpoint(candidate_model_file)
    
    # 2. Define evaluation config
    test_config = {
        'eval_num_games': 4, # Very low for a quick test
        'simulations_per_move': 8,
        'eval_win_threshold': 0.51 # 51% for testing
    }
    
    # 3. Run evaluation
    is_candidate_better = run_evaluation(candidate_model_file, best_model_file, test_config)
    print(f"Did candidate win? -> {is_candidate_better}")

    # Clean up
    os.remove(best_model_file)
    os.remove(candidate_model_file)