# src/davis_ai/training/self_play.py

import chess
import torch
import uuid
import pickle
import os
from dataclasses import dataclass, field
from typing import List, Tuple

from davis_ai.engine.mcts import MCTSNode

from ..engine import InferenceEngine, MonteCarloTreeSearch
from ..board import Board

@dataclass
class GameRecord:
    """Stores all data from a single self-play game."""
    game_id: str
    history: List[Tuple[str, torch.Tensor, int]] = field(default_factory=list) # FEN, Policy, Turn
    outcome: float = 0.0

class SelfPlayWorker:
    """
    A worker that orchestrates a single game of self-play between two instances
    of the AI, generating a GameRecord.
    """
    def __init__(self, inference_engine: InferenceEngine, config: dict):
        # CORRECTED: The MCTS engine now takes the entire config dictionary directly.
        self.mcts = MonteCarloTreeSearch(
            inference_engine=inference_engine,
            config=config 
        )
        self.simulations_per_move = config.get('simulations_per_move', 100)
        self.move_limit = config.get('move_limit', 200)

    def play_game(self) -> GameRecord:
        """
        Executes a single game of self-play from the starting position.
        
        Returns:
            A GameRecord object containing the full history and outcome.
        """
        board = Board()
        game_record = GameRecord(game_id=str(uuid.uuid4()))
        
        move_count = 0
        while not board.is_game_over() and move_count < self.move_limit:
            # For early moves, use a higher temperature for exploration.
            # Later in the game, reduce temperature to play more deterministically.
            temperature = 1.0 if move_count < 30 else 0.1

            # Perform the MCTS search
            root_node = self.mcts.search(board, self.simulations_per_move)

            # Create the policy target for training
            policy_target = self._calculate_policy_target(root_node)
            game_record.history.append((board.fen, policy_target, board.turn))

            # Select and play the move
            move = self.mcts.select_move(root_node, temperature=temperature)
            board = board.apply_move(move)
            move_count += 1
            
        game_record.outcome = board.get_game_outcome() or 0.0 # Or 0.0 if move limit hit
        return game_record

    def _calculate_policy_target(self, root_node: 'MCTSNode') -> torch.Tensor:
        """
        Calculates the policy vector from the root node's children visit counts.
        The policy vector size is 4672, matching the model output.
        """
        policy = torch.zeros(4672, dtype=torch.float32)
        if not root_node.children:
            return policy # No legal moves

        total_visits = sum(child.visits for child in root_node.children.values())
        if total_visits == 0:
            return policy # Should not happen if search is run

        from ..utils.move_encoder import move_to_policy_index
        for move, child in root_node.children.items():
            try:
                policy_idx = move_to_policy_index(move)
                policy[policy_idx] = child.visits / total_visits
            except (ValueError, IndexError):
                continue
        return policy

def run_self_play_and_save(model_path: str, save_dir: str, config: dict):
    """
    Main entry point to initialize a worker, play a game, and save the result.
    """
    print(f"Starting self-play worker for model: {model_path}")
    os.makedirs(save_dir, exist_ok=True)
    
    engine = InferenceEngine(model_path=model_path)
    worker = SelfPlayWorker(engine, config)
    
    game_record = worker.play_game()
    
    save_path = os.path.join(save_dir, f"{game_record.game_id}.pkl")
    with open(save_path, 'wb') as f:
        pickle.dump(game_record, f)
        
    print(f"Game {game_record.game_id} finished with outcome {game_record.outcome}. Saved to {save_path}")

if __name__ == '__main__':
    # --- Example Usage ---
    # This block would typically be called by a master training script.
    
    # 1. Create a dummy model for the worker to use
    from ..engine import DavisChessModel
    dummy_model_file = "dummy_selfplay_model.pth"
    model = DavisChessModel(input_channels=17, num_residual_blocks=1, num_filters=32)
    model.save_checkpoint(dummy_model_file)
    
    # 2. Define configuration
    test_config = {
        'simulations_per_move': 16, # Low for a quick test
        'mcts_c_puct': 1.25,
        'move_limit': 50
    }
    
    # 3. Run a single game and save it
    run_self_play_and_save(
        model_path=dummy_model_file,
        save_dir="temp_games/",
        config=test_config
    )
    
    # Clean up
    os.remove(dummy_model_file)
    # Note: temp_games/ directory and the game file within it will remain for inspection.