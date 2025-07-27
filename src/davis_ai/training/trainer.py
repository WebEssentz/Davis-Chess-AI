# src/davis_ai/training/trainer.py

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from ..engine import DavisChessModel

class Trainer:
    """
    Manages the deep learning training loop for Davis.
    It handles optimization, loss calculation, and model checkpointing.
    """
    def __init__(self, model: DavisChessModel, config: dict):
        self.config = config
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = model.to(self.device)
        
        self.lr = config.get('learning_rate', 1e-3)
        self.epochs = config.get('epochs', 10)
        self.weight_decay = config.get('weight_decay', 1e-4)
        
        self.optimizer = AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=self.epochs)
        # Corrected for FutureWarning
        self.scaler = torch.amp.GradScaler('cuda', enabled=(self.device == 'cuda'))

    def train(self, dataloader: DataLoader):
        """
        Runs the training loop for a given number of epochs.
        """
        self.model.train()
        
        for epoch in range(self.epochs):
            total_loss, total_policy_loss, total_value_loss = 0, 0, 0
            
            for board_batch, policy_target_batch, value_target_batch in dataloader:
                board_batch = board_batch.to(self.device)
                policy_target_batch = policy_target_batch.to(self.device)
                value_target_batch = value_target_batch.to(self.device)

                self.optimizer.zero_grad()
                
                # Corrected for FutureWarning
                with torch.amp.autocast('cuda', enabled=(self.device == 'cuda')):
                    policy_logits, value_pred = self.model(board_batch)
                    
                    value_loss = F.mse_loss(value_pred, value_target_batch)
                    policy_loss = F.cross_entropy(policy_logits, policy_target_batch)
                    loss = value_loss + policy_loss

                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()

                total_loss += loss.item()
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()

            avg_loss = total_loss / len(dataloader)
            avg_policy_loss = total_policy_loss / len(dataloader)
            avg_value_loss = total_value_loss / len(dataloader)
            
            self.scheduler.step()
            
            print(
                f"Epoch {epoch+1}/{self.epochs} | "
                f"Avg Loss: {avg_loss:.4f} | "
                f"Policy Loss: {avg_policy_loss:.4f} | "
                f"Value Loss: {avg_value_loss:.4f} | "
                f"LR: {self.scheduler.get_last_lr()[0]:.6f}"
            )

def run_training_cycle(model_path: str, processed_data_path: str, new_model_save_path: str, config: dict):
    """
    Main entry point to load a model, train it on processed data, and save the new version.
    """
    print(f"\n--- Starting Training Cycle ---")
    print(f"Loading base model from: {model_path}")
    model = DavisChessModel.load_checkpoint(model_path)
    
    print(f"Loading data from: {processed_data_path}")
    from .data_pipeline import create_dataloader
    dataloader = create_dataloader(processed_data_path, batch_size=config.get('batch_size', 256))
    
    trainer = Trainer(model, config)
    trainer.train(dataloader)
    
    print(f"Training complete. Saving new model to: {new_model_save_path}")
    model.save_checkpoint(new_model_save_path)


if __name__ == '__main__':
    import os
    import pickle
    print("\n--- Testing Trainer ---")
    
    # ADD THIS IMPORT
    from ..board import Board
    
    dummy_model_file = "dummy_trainer_model.pth"
    model = DavisChessModel(input_channels=17, num_residual_blocks=1, num_filters=16)
    model.save_checkpoint(dummy_model_file)
    
    processed_data_file = "dummy_processed_data.pkl"
    dummy_data = [(Board().fen, torch.randn(4672), 1.0) for _ in range(64)]
    with open(processed_data_file, 'wb') as f:
        pickle.dump(dummy_data, f)
        
    test_config = {
        'learning_rate': 1e-3,
        'epochs': 2,
        'batch_size': 16
    }
    
    new_model_path = "dummy_trained_model.pth"
    run_training_cycle(dummy_model_file, processed_data_file, new_model_path, test_config)
    
    assert os.path.exists(new_model_path)
    print("\nTrainer sanity check PASSED.")
    
    os.remove(dummy_model_file)
    os.remove(processed_data_file)
    os.remove(new_model_path)