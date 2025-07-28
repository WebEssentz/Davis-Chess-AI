# src/davis_ai/engine/model.py (Upgraded)

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any

class ResidualBlock(nn.Module):
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
    def __init__(self, input_channels: int = 17, num_residual_blocks: int = 19, num_filters: int = 256, policy_output_size: int = 4672):
        super().__init__()
        self.config = {
            'input_channels': input_channels,
            'num_residual_blocks': num_residual_blocks,
            'num_filters': num_filters,
            'policy_output_size': policy_output_size
        }
        self.initial_conv = nn.Sequential(
            nn.Conv2d(input_channels, num_filters, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_filters),
            nn.ReLU(inplace=True)
        )
        self.residual_blocks = nn.ModuleList([ResidualBlock(num_filters) for _ in range(num_residual_blocks)])
        self.policy_head = nn.Sequential(
            nn.Conv2d(num_filters, 2, kernel_size=1, bias=False),
            nn.BatchNorm2d(2),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(2 * 8 * 8, policy_output_size)
        )
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

    def save_checkpoint(self, path: str, compile_for_inference: bool = False):
        """ Saves the model, optionally compiling a TorchScript version for speed. """
        checkpoint = {'config': self.config, 'state_dict': self.state_dict()}
        torch.save(checkpoint, path)
        print(f"DavisChessModel checkpoint saved to {path}")

        # --- UPGRADE #21: Compile and save a TorchScript version for faster inference ---
        if compile_for_inference:
            try:
                self.eval() # Model must be in eval mode
                scripted_model = torch.jit.script(self)
                script_path = path.replace('.pth', '.pt')
                scripted_model.save(script_path)
                print(f"Compiled TorchScript model saved to {script_path}")
            except Exception as e:
                print(f"Could not compile model to TorchScript: {e}")

    @staticmethod
    def load_checkpoint(path: str) -> 'DavisChessModel':
        """ Loads a model from a checkpoint file. """
        checkpoint = torch.load(path, map_location=torch.device('cpu'))
        model = DavisChessModel(**checkpoint['config'])
        model.load_state_dict(checkpoint['state_dict'])
        print(f"DavisChessModel loaded from checkpoint {path}")
        return model