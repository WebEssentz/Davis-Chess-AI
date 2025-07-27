import torch
import chess
from unittest.mock import MagicMock

from davis_ai.engine import MonteCarloTreeSearch, InferenceEngine
from davis_ai.board import Board

def test_mcts_predict_calls_and_root_visits():
    """
    Tests that N simulations result in N predict calls and N root visits.
    """
    mock_engine = MagicMock(spec=InferenceEngine)
    mock_policy = torch.ones(4672) / 4672
    mock_engine.predict.return_value = (mock_policy, 0.5)

    mcts = MonteCarloTreeSearch(inference_engine=mock_engine)
    board = Board()
    num_sims = 50
    
    root_node = mcts.search(board, num_simulations=num_sims)
    
    assert mock_engine.predict.call_count == num_sims
    assert root_node.visits == num_sims

def test_mcts_expansion():
    """Tests that the root node is correctly expanded with children."""
    mock_engine = MagicMock(spec=InferenceEngine)
    mock_policy = torch.ones(4672) / 4672
    mock_engine.predict.return_value = (mock_policy, 0.0)

    mcts = MonteCarloTreeSearch(inference_engine=mock_engine)
    board = Board()
    
    root_node = mcts.search(board, num_simulations=1)
    
    legal_moves = board.get_legal_moves()
    assert len(root_node.children) > 0
    assert len(root_node.children) <= len(legal_moves)

def test_mcts_backpropagation_and_child_visits():
    """
    Tests value backpropagation and that child nodes are visited correctly.
    """
    mock_engine = MagicMock(spec=InferenceEngine)
    mock_engine.predict.return_value = (torch.ones(4672)/4672, 0.8)

    mcts = MonteCarloTreeSearch(inference_engine=mock_engine)
    board = Board()
    
    num_sims = 2
    root_node = mcts.search(board, num_simulations=num_sims)

    # After 2 simulations, root is visited twice
    assert root_node.visits == 2
    # The value should sum to ~0 as explained before
    assert abs(root_node.value_sum) < 1e-6
    # One of the children must have been visited exactly once
    child_visits = [child.visits for child in root_node.children.values()]
    assert 1 in child_visits