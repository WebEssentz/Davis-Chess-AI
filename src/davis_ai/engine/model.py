# src/davis_ai/engine/model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any

class ResidualBlock(nn.Module):
    """
    A standard residual block for the DavisChessModel.
    Enhances feature learning by allowing gradients to flow more easily.
    """
    def __init__(self, num_channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(num_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = F.relu(out)
        return out

class DavisChessModel(nn.Module):
    """
    The deep learning model architecture for Davis, Ossie's Chess AI.
    Inspired by AlphaZero, it uses a ResNet-like structure to predict
    move probabilities (policy) and game outcome (value).
    """
    def __init__(self, input_channels: int = 17, num_residual_blocks: int = 19, num_filters: int = 256, policy_output_size: int = 4672):
        super().__init__()
        self.config = {
            'input_channels': input_channels,
            'num_residual_blocks': num_residual_blocks,
            'num_filters': num_filters,
            'policy_output_size': policy_output_size
        }

        # Initial convolutional block
        self.initial_conv = nn.Sequential(
            nn.Conv2d(input_channels, num_filters, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_filters),
            nn.ReLU(inplace=True)
        )

        # Residual tower
        self.residual_blocks = nn.ModuleList(
            [ResidualBlock(num_filters) for _ in range(num_residual_blocks)]
        )

        # Policy head: Predicts probabilities for each possible move
        self.policy_head = nn.Sequential(
            nn.Conv2d(num_filters, 2, kernel_size=1, bias=False),
            nn.BatchNorm2d(2),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(2 * 8 * 8, policy_output_size)
        )

        # Value head: Predicts the outcome of the game from the current state
        self.value_head = nn.Sequential(
            nn.Conv2d(num_filters, 1, kernel_size=1, bias=False),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(1 * 8 * 8, num_filters),
            nn.ReLU(inplace=True),
            nn.Linear(num_filters, 1),
            nn.Tanh()
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.initial_conv(x)
        for block in self.residual_blocks:
            x = block(x)
        
        policy_logits = self.policy_head(x)
        value = self.value_head(x)
        return policy_logits, value

    def save_checkpoint(self, path: str):
        """
        Saves the model checkpoint, including architecture config and state_dict.
        """
        checkpoint = {
            'config': self.config,
            'state_dict': self.state_dict()
        }
        torch.save(checkpoint, path)
        print(f"DavisChessModel checkpoint saved to {path}")

    @staticmethod
    def load_checkpoint(path: str) -> 'DavisChessModel':
        """
        Loads a model from a checkpoint file.
        """
        checkpoint = torch.load(path, map_location=torch.device('cpu'))
        model = DavisChessModel(**checkpoint['config'])
        model.load_state_dict(checkpoint['state_dict'])
        print(f"DavisChessModel loaded from checkpoint {path}")
        return model

if __name__ == '__main__':
    import os
    print("--- Initializing and Testing DavisChessModel Checkpointing ---")
    
    # Create a small model for testing
    model = DavisChessModel(input_channels=17, num_residual_blocks=1, num_filters=32)
    print(f"Model architecture created with config: {model.config}")

    # Create a dummy input tensor
    dummy_input = torch.randn(1, 17, 8, 8)
    policy_logits, value = model(dummy_input)
    print(f"Sanity Check -> Policy Logits Shape: {policy_logits.shape}, Value: {value.item():.4f}")

    # Test saving and loading
    dummy_path = "davis_test_model.pth"
    model.save_checkpoint(dummy_path)
    
    loaded_model = DavisChessModel.load_checkpoint(dummy_path)
    print(f"Loaded model config: {loaded_model.config}")
    
    # Verify the loaded model gives the same output
    loaded_model.eval()
    model.eval()
    with torch.no_grad():
        original_output = model(dummy_input)[0]
        loaded_output = loaded_model(dummy_input)[0]
    
    assert torch.allclose(original_output, loaded_output), "Model output mismatch after loading!"
    print("Checkpoint save/load test PASSED.")

    os.remove(dummy_path)
    print(f"Cleaned up dummy file: {dummy_path}")