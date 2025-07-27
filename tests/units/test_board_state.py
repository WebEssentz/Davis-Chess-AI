import torch
import chess
from davis_ai.board import Board

def test_board_initialization():
    """Tests that the board initializes to the starting position."""
    board = Board()
    assert board.fen == chess.STARTING_FEN
    assert board.turn == chess.WHITE

def test_apply_move_immutability():
    """Tests that apply_move creates a new object and doesn't modify the original."""
    board1 = Board()
    move = chess.Move.from_uci("e2e4")
    board2 = board1.apply_move(move)
    assert board1.fen == chess.STARTING_FEN
    assert " b " in board2.fen

def test_game_over_checkmate():
    """Tests game over detection for a checkmate position."""
    # Use a board state that is one move away from a real checkmate
    board = Board(chess.Board("rnbqkbnr/pppp1ppp/8/4p3/6P1/5P2/PPPPP2P/RNBQKBNR b KQkq - 0 2"))
    move = chess.Move.from_uci("d8h4")
    
    # Ensure the move is legal before applying
    assert move in board.get_legal_moves()
    
    mated_board = board.apply_move(move)
    assert mated_board.is_game_over()
    assert mated_board.get_game_outcome() == -1.0

def test_game_over_stalemate():
    """Tests game over detection for a stalemate position."""
    # This FEN is a classic stalemate: Black to move with no legal options
    stale_fen = "7k/5Q2/6K1/8/8/8/8/8 b - - 0 1"
    board = Board(chess.Board(stale_fen))
    assert board.is_game_over()
    assert board.get_game_outcome() == 0.0


def test_tensor_conversion_shape_and_type():
    """Tests the shape and data type of the tensor conversion."""
    board = Board()
    tensor = board.to_model_input()
    assert tensor.shape == (17, 8, 8)
    assert tensor.dtype == torch.float32

def test_tensor_piece_planes():
    """Tests that pieces are correctly placed on the tensor planes."""
    board = Board()
    tensor = board.to_model_input()
    assert tensor[0, 1, 4] == 1.0
    assert tensor[6, 6, 3] == 1.0
    for i in range(12):
        assert tensor[i, 3, 4] == 0.0