# src/davis_ai/training/self_play.py (Final, Corrected Version)

import chess
import torch
import uuid
import pickle
import os
from dataclasses import dataclass, field
from typing import List, Tuple

# --- CORRECTED IMPORT ---
# We now use a consistent, relative import style for all internal modules.
from ..engine import InferenceEngine, MonteCarloTreeSearch, MCTSNode
from ..board import Board

@dataclass
class GameRecord:
    """Stores all data from a single self-play game."""
    game_id: str
    history: List[Tuple[str, torch.Tensor, int]] = field(default_factory=list)
    outcome: float = 0.0

class SelfPlayWorker:
    """ A worker that orchestrates a single game of self-play. """
    def __init__(self, inference_engine: InferenceEngine, config: dict):
        self.mcts = MonteCarloTreeSearch(
            inference_engine=inference_engine,
            config=config 
        )
        self.simulations_per_move = config.get('simulations_per_move', 100)
        self.move_limit = config.get('move_limit', 200)

    def play_game(self) -> GameRecord:
        """ Executes a single game of self-play. """
        board = Board()
        game_record = GameRecord(game_id=str(uuid.uuid4()))
        move_count = 0
        while not board.is_game_over() and move_count < self.move_limit:
            temperature = 1.0 if move_count < 30 else 0.1
            root_node = self.mcts.search(board, self.simulations_per_move)
            policy_target = self._calculate_policy_target(root_node)
            game_record.history.append((board.fen, policy_target, board.turn))
            move = self.mcts.select_move(root_node, temperature=temperature)
            board = board.apply_move(move)
            move_count += 1
        game_record.outcome = board.get_game_outcome() or 0.0
        return game_record

    def _calculate_policy_target(self, root_node: MCTSNode) -> torch.Tensor:
        """ Calculates the policy vector from the root node's children visit counts. """
        policy = torch.zeros(4672, dtype=torch.float32)
        if not root_node.children:
            return policy
        total_visits = sum(child.visits for child in root_node.children.values())
        if total_visits == 0:
            return policy
        from ..utils.move_encoder import move_to_policy_index
        for move, child in root_node.children.items():
            try:
                policy_idx = move_to_policy_index(move)
                policy[policy_idx] = child.visits / total_visits
            except (ValueError, IndexError):
                continue
        return policy

def run_self_play_and_save(model_path: str, save_dir: str, config: dict):
    """ Main entry point to initialize a worker, play a game, and save the result. """
    # This function is now primarily called by the multiprocessing worker,
    # so verbose printing can be reduced.
    os.makedirs(save_dir, exist_ok=True)
    engine = InferenceEngine(model_path=model_path)
    worker = SelfPlayWorker(engine, config)
    game_record = worker.play_game()
    save_path = os.path.join(save_dir, f"{game_record.game_id}.pkl")
    with open(save_path, 'wb') as f:
        pickle.dump(game_record, f)
    # This print will show up in the worker logs, which is fine.
    print(f"Game {game_record.game_id} finished with outcome {game_record.outcome}. Saved to {save_path}")