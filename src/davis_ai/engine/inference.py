# src/davis_ai/engine/inference.py (Upgraded)

import torch
import os
from typing import Tuple

from .model import DavisChessModel
from ..board import Board

class InferenceEngine:
    def __init__(self, model_path: str, device: str = None):
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Initializing Davis InferenceEngine on device: {self.device}")
        self.model = self._load_model(model_path)
        self.model.eval()

    def _load_model(self, model_path: str):
        """ Loads a compiled TorchScript model if available, otherwise loads the standard model. """
        script_path = model_path.replace('.pth', '.pt')
        
        # --- UPGRADE #21: Prioritize loading the faster, compiled model ---
        if os.path.exists(script_path):
            try:
                print(f"Loading compiled TorchScript model from {script_path}")
                model = torch.jit.load(script_path, map_location=self.device)
                return model
            except Exception as e:
                print(f"Failed to load TorchScript model, falling back to standard. Error: {e}")

        # Fallback to standard model loading
        try:
            model = DavisChessModel.load_checkpoint(model_path)
            model.to(self.device)
            return model
        except FileNotFoundError:
            raise FileNotFoundError(f"FATAL: Model checkpoint not found at: {model_path}")
        except Exception as e:
            raise RuntimeError(f"FATAL: Failed to load DavisChessModel from {model_path}: {e}")

    def predict(self, board: Board) -> Tuple[torch.Tensor, float]:
        """ Performs a prediction on a single Board object using the loaded model. """
        model_input = board.to_model_input().unsqueeze(0).to(self.device)
        with torch.no_grad():
            policy_logits, value_tensor = self.model(model_input)
        policy_probs = torch.nn.functional.softmax(policy_logits, dim=-1)
        return policy_probs.squeeze(0).cpu(), value_tensor.item()