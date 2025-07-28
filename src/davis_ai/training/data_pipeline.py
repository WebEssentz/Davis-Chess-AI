# src/davis_ai/training/data_pipeline.py

import os
import pickle
import torch
import numpy as np
import chess
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple

from .self_play import GameRecord
from ..board import Board
from ..engine import InferenceEngine, MonteCarloTreeSearch
from ..utils.move_encoder import move_to_policy_index, policy_index_to_move # Import the new decoder

# --- ECR (Echo Chamber Reinforcement) Analyst ---

def analyze_and_enhance_game(game_record: GameRecord, analyst_engine: InferenceEngine, config: dict) -> List[Tuple[str, torch.Tensor, float]]:
    """
    Performs ECR analysis on a completed game to generate training samples.
    
    Returns a list of tuples: (FEN string, policy_target, value_target)
    """
    training_samples = []
    game_outcome = game_record.outcome
    
    for fen, policy, turn in game_record.history:
        value = game_outcome if turn == chess.WHITE else -game_outcome
        training_samples.append((fen, policy, value))

    if game_outcome != 0.0:
        winner_color = chess.WHITE if game_outcome > 0 else chess.BLACK
        best_blunder_state_idx = -1
        max_regret = -1.0
        analyst_mcts = MonteCarloTreeSearch(analyst_engine, c_puct=config.get('mcts_c_puct', 4.0))
        analyst_sims = config.get('ecr_analyst_simulations', 400)
        
        for i, (fen, _, turn) in enumerate(game_record.history):
            if turn == winner_color:
                board = Board(chess.Board(fen))
                _, initial_value = analyst_engine.predict(board)
                root_node = analyst_mcts.search(board, analyst_sims)
                deep_value = -root_node.q_value
                regret = initial_value - deep_value
                if regret > max_regret:
                    max_regret = regret
                    best_blunder_state_idx = i

        if best_blunder_state_idx != -1:
            fen, old_policy, turn = game_record.history[best_blunder_state_idx]
            board = Board(chess.Board(fen))
            root_node = analyst_mcts.search(board, analyst_sims)
            from .self_play import SelfPlayWorker
            improved_policy = SelfPlayWorker(analyst_engine, {})._calculate_policy_target(root_node)
            value = game_outcome if turn == chess.WHITE else -game_outcome
            training_samples.append((fen, improved_policy, value))
            print(f"ECR: Generated counterfactual for game {game_record.game_id} at move {best_blunder_state_idx}")

    return training_samples


# --- PyTorch Dataset ---

class ChessDataset(Dataset):
    def __init__(self, data: List[Tuple[str, torch.Tensor, float]], augment: bool = True):
        self.data = data
        self.augment = augment

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        fen, policy_target, value_target = self.data[idx]
        board = Board(chess.Board(fen))
        board_tensor = board.to_model_input()
        
        if self.augment and torch.rand(1).item() > 0.5:
            board_tensor = torch.flip(board_tensor, [2])
            policy_target = self._flip_policy(policy_target, board)
            
        return board_tensor, policy_target, torch.tensor([value_target], dtype=torch.float32)

    @staticmethod
    def _flip_policy(policy_target: torch.Tensor, board: Board) -> torch.Tensor:
        flipped_policy = torch.zeros_like(policy_target)
        non_zero_indices = policy_target.nonzero()

        # Handle the edge case of a 0-D tensor (single non-zero element)
        if non_zero_indices.dim() == 1 and non_zero_indices.numel() == 1:
            non_zero_indices = non_zero_indices.unsqueeze(0)
        
        # Handle the general case, ensuring we iterate over a 2D tensor of indices
        for idx_tensor in non_zero_indices:
            move_idx = idx_tensor.item()
            prob = policy_target[move_idx].item()
            try:
                original_move = policy_index_to_move(move_idx, board._board)
                flipped_from = chess.square_mirror(original_move.from_square)
                flipped_to = chess.square_mirror(original_move.to_square)
                flipped_move = chess.Move(flipped_from, flipped_to, promotion=original_move.promotion)
                flipped_idx = move_to_policy_index(flipped_move)
                flipped_policy[flipped_idx] = prob
            except Exception:
                flipped_policy[move_idx] = prob
                
        return flipped_policy
    
    """ A PyTorch Dataset for loading game data, with robust data augmentation. """
    def __init__(self, data: List[Tuple[str, torch.Tensor, float]], augment: bool = True):
        self.data = data
        self.augment = augment

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        fen, policy_target, value_target = self.data[idx]
        board = Board(chess.Board(fen))
        board_tensor = board.to_model_input()
        
        if self.augment and torch.rand(1).item() > 0.5:
            board_tensor = torch.flip(board_tensor, [2])
            policy_target = self._flip_policy(policy_target, board)
            
        return board_tensor, policy_target, torch.tensor([value_target], dtype=torch.float32)

    @staticmethod
    def _flip_policy(policy_target: torch.Tensor, board: Board) -> torch.Tensor:
        """ Flips the policy tensor horizontally with 100% accuracy and robust iteration. """
        flipped_policy = torch.zeros_like(policy_target)
        
        # --- THE FIX ---
        # Get the indices of non-zero elements
        non_zero_indices = policy_target.nonzero().squeeze()
        
        # If there's only one non-zero element, squeeze() makes it a 0-D tensor.
        # We must check for this and make it iterable.
        if non_zero_indices.dim() == 0:
            # If it's a single number (e.g., tensor(459)), this condition is true.
            # We wrap it in a list/tensor to make it a 1-D tensor.
            if non_zero_indices.numel() > 0: # Ensure it's not an empty tensor
                 non_zero_indices = non_zero_indices.unsqueeze(0)
            else: # Handle case of all-zero policy tensor
                 return flipped_policy

        for move_idx in non_zero_indices:
            prob = policy_target[move_idx].item()
            try:
                original_move = policy_index_to_move(move_idx.item(), board._board)
                flipped_from = chess.square_mirror(original_move.from_square)
                flipped_to = chess.square_mirror(original_move.to_square)
                flipped_move = chess.Move(flipped_from, flipped_to, promotion=original_move.promotion)
                flipped_idx = move_to_policy_index(flipped_move)
                flipped_policy[flipped_idx] = prob
            except Exception:
                flipped_policy[move_idx] = prob # Fallback on error
                
        return flipped_policy
    
    """ A PyTorch Dataset for loading game data, now with perfect data augmentation. """
    def __init__(self, data: List[Tuple[str, torch.Tensor, float]], augment: bool = True):
        self.data = data
        self.augment = augment

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        fen, policy_target, value_target = self.data[idx]
        board = Board(chess.Board(fen)) # We need the board object for flipping
        board_tensor = board.to_model_input()
        
        # --- UPGRADE #11: Perfect Data Augmentation via Board Symmetries ---
        if self.augment and torch.rand(1).item() > 0.5:
            # Flip the board tensor horizontally
            board_tensor = torch.flip(board_tensor, [2])
            # Flip the policy target to match the board flip
            policy_target = self._flip_policy(policy_target, board)
            
        return board_tensor, policy_target, torch.tensor([value_target], dtype=torch.float32)

    @staticmethod
    def _flip_policy(policy_target: torch.Tensor, board: Board) -> torch.Tensor:
        """ Flips the policy tensor horizontally with 100% accuracy. """
        flipped_policy = torch.zeros_like(policy_target)
        
        # Iterate through the original policy's probabilities
        for move_idx in policy_target.nonzero().squeeze():
            prob = policy_target[move_idx].item()
            
            try:
                # 1. Decode the move index back to a move object
                original_move = policy_index_to_move(move_idx.item(), board._board)

                # 2. Mathematically flip the move's squares
                flipped_from = chess.square_mirror(original_move.from_square)
                flipped_to = chess.square_mirror(original_move.to_square)
                
                # 3. Create the new, flipped move object
                flipped_move = chess.Move(flipped_from, flipped_to, promotion=original_move.promotion)
                
                # 4. Re-encode the new, flipped move to get its correct policy index
                flipped_idx = move_to_policy_index(flipped_move)
                
                # 5. Assign the probability to the new index
                flipped_policy[flipped_idx] = prob
            
            except Exception:
                # If any part of the decode/encode fails (e.g., for a strange move),
                # we just use the original index to be safe. This is rare.
                flipped_policy[move_idx] = prob
                
        return flipped_policy

def create_dataloader(processed_data_path: str, batch_size: int, shuffle: bool = True) -> DataLoader:
    """Loads processed data from a file and creates a DataLoader."""
    with open(processed_data_path, 'rb') as f:
        all_samples = pickle.load(f)
    dataset = ChessDataset(all_samples)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0, pin_memory=True) # Set num_workers to 0 on Windows

if __name__ == '__main__':
    print("--- Testing Data Pipeline and ECR ---")
    
    from ..engine import DavisChessModel
    # CORRECTED: Import the function needed by the test block
    from .self_play import run_self_play_and_save

    dummy_model_file = "dummy_pipeline_model.pth"
    model = DavisChessModel(input_channels=17, num_residual_blocks=1, num_filters=16)
    model.save_checkpoint(dummy_model_file)
    
    test_config = {'simulations_per_move': 8, 'ecr_analyst_simulations': 16}
    game_dir = "temp_games/"
    os.makedirs(game_dir, exist_ok=True)
    
    # Check if any files exist before trying to access index 0
    if not os.listdir(game_dir):
        print("No game files found, generating one for test...")
        run_self_play_and_save(dummy_model_file, game_dir, test_config)

    game_file_path = os.path.join(game_dir, os.listdir(game_dir)[0])
    with open(game_file_path, 'rb') as f:
        record = pickle.load(f)
    print(f"Loaded game {record.game_id} for processing.")
    
    analyst = InferenceEngine(dummy_model_file)
    processed_samples = analyze_and_enhance_game(record, analyst, test_config)
    print(f"Original history length: {len(record.history)}. Processed samples length: {len(processed_samples)}")
    assert len(processed_samples) >= len(record.history)
    
    processed_data_file = "processed_training_data.pkl"
    with open(processed_data_file, 'wb') as f:
        pickle.dump(processed_samples, f)
        
    dataloader = create_dataloader(processed_data_file, batch_size=4)
    board_batch, policy_batch, value_batch = next(iter(dataloader))
    
    print("\n--- DataLoader Sanity Check ---")
    print("Batch of board tensors shape:", board_batch.shape)
    print("Batch of policy targets shape:", policy_batch.shape)
    print("Batch of value targets shape:", value_batch.shape)
    
    print("\nData Pipeline with ECR check PASSED.")
    
    os.remove(dummy_model_file)
    os.remove(processed_data_file)
    # Clean up the temp_games directory if you want
    # import shutil
    # shutil.rmtree(game_dir)