# In tests/units/test_mcts.py (Final Corrected Version)

import torch
import chess
from unittest.mock import MagicMock

from davis_ai.engine import MonteCarloTreeSearch, InferenceEngine
from davis_ai.board import Board

# Create a default dummy config for all tests in this file
DUMMY_CONFIG = {
    'mcts_c_puct': 4.0,
    'mcts_dirichlet_alpha': 0.3,
    'mcts_dirichlet_epsilon': 0.25,
    'mcts_top_k_moves': 5
}

def test_mcts_predict_calls_and_root_visits():
    """ Tests that N simulations result in N predict calls and N root visits. """
    mock_engine = MagicMock(spec=InferenceEngine)
    mock_policy = torch.ones(4672) / 4672
    mock_engine.predict.return_value = (mock_policy, 0.5)

    # CORRECTED: Pass the dummy config
    mcts = MonteCarloTreeSearch(inference_engine=mock_engine, config=DUMMY_CONFIG)
    board = Board()
    num_sims = 50
    
    root_node = mcts.search(board, num_simulations=num_sims)
    
    assert mock_engine.predict.call_count == num_sims
    assert root_node.visits == num_sims

def test_mcts_expansion():
    """ Tests that the root node is correctly expanded with children. """
    mock_engine = MagicMock(spec=InferenceEngine)
    mock_policy = torch.ones(4672) / 4672
    mock_engine.predict.return_value = (mock_policy, 0.0)

    # CORRECTED: Pass the dummy config
    mcts = MonteCarloTreeSearch(inference_engine=mock_engine, config=DUMMY_CONFIG)
    board = Board()
    
    root_node = mcts.search(board, num_simulations=1)
    
    legal_moves = board.get_legal_moves()
    assert len(root_node.children) > 0
    assert len(root_node.children) <= len(legal_moves)

def test_mcts_backpropagation_and_child_visits():
    """ Tests value backpropagation and that child nodes are visited correctly. """
    mock_engine = MagicMock(spec=InferenceEngine)
    mock_engine.predict.return_value = (torch.ones(4672)/4672, 0.8)

    # CORRECTED: Pass the dummy config
    mcts = MonteCarloTreeSearch(inference_engine=mock_engine, config=DUMMY_CONFIG)
    board = Board()
    
    num_sims = 2
    root_node = mcts.search(board, num_simulations=num_sims)

    assert root_node.visits == 2
    assert abs(root_node.value_sum) < 1e-6
    child_visits = [child.visits for child in root_node.children.values()]
    assert 1 in child_visits