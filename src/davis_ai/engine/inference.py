# src/davis_ai/engine/inference.py

import torch
from typing import Tuple

from .model import DavisChessModel
# CORRECTED IMPORT: Use the new, optimized Board class
from ..board import Board

class InferenceEngine:
    """
    The Inference Engine for Davis.
    This class loads a trained DavisChessModel and performs efficient, batched
    predictions. It handles all device management (CPU/GPU) and ensures the
    model is in evaluation mode. This is the sole entry point for getting
    predictions from a model file.
    """
    def __init__(self, model_path: str, device: str = None):
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Initializing Davis InferenceEngine on device: {self.device}")
        
        try:
            self.model = DavisChessModel.load_checkpoint(model_path)
            self.model.to(self.device)
            self.model.eval()
        except FileNotFoundError:
            raise FileNotFoundError(f"FATAL: Model checkpoint not found at: {model_path}")
        except Exception as e:
            raise RuntimeError(f"FATAL: Failed to load DavisChessModel from {model_path}: {e}")

    # CORRECTED TYPE HINT AND VARIABLE NAME
    def predict(self, board: Board) -> Tuple[torch.Tensor, float]:
        """
        Performs a prediction on a single Board object using the loaded model.

        Args:
            board (Board): The board state to evaluate.

        Returns:
            Tuple[torch.Tensor, float]:
                - policy_probs (torch.Tensor): Softmax-activated probabilities for all moves.
                - value (float): Predicted game outcome value (-1 to 1).
        """
        model_input = board.to_model_input()
        
        model_input = model_input.unsqueeze(0).to(self.device)

        with torch.no_grad():
            policy_logits, value_tensor = self.model(model_input)

        policy_probs = torch.nn.functional.softmax(policy_logits, dim=-1)
        
        return policy_probs.squeeze(0).cpu(), value_tensor.item()