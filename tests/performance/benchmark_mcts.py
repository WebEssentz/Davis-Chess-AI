# tests/performance/benchmark_mcts.py

import time
import torch
from davis_ai.engine import DavisChessModel, InferenceEngine, MonteCarloTreeSearch
from davis_ai.board import Board

def run_mcts_benchmark():
    """
    Measures the performance of the MCTS search in nodes per second.
    """
    print("\n--- Running MCTS Performance Benchmark ---")
    
    # Setup a small, fast model
    model = DavisChessModel(input_channels=17, num_residual_blocks=1, num_filters=16)
    
    # Create a dummy path and save the model
    # (Not using a fixture here as it's a runnable script, not a typical pytest test)
    dummy_model_path = "temp_benchmark_model.pth"
    model.save_checkpoint(dummy_model_path)
    
    engine = InferenceEngine(model_path=dummy_model_path)
    mcts = MonteCarloTreeSearch(inference_engine=engine)
    board = Board()
    
    num_sims = 100
    
    print(f"Benchmarking with {num_sims} simulations...")
    start_time = time.perf_counter()
    mcts.search(board, num_simulations=num_sims)
    end_time = time.perf_counter()
    
    duration = end_time - start_time
    nodes_per_second = num_sims / duration
    
    print(f"Total time for {num_sims} simulations: {duration:.4f} seconds")
    print(f"PERFORMANCE: {nodes_per_second:.2f} nodes/second")
    
    # Clean up
    import os
    os.remove(dummy_model_path)

if __name__ == '__main__':
    # This allows running the benchmark directly: `python tests/performance/benchmark_mcts.py`
    run_mcts_benchmark()