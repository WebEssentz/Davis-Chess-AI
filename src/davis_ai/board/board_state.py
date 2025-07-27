# src/davis_ai/board/board_state.py

import chess
import torch
import numpy as np
from typing import List, Optional
from functools import lru_cache

class Board:
    """
    An optimized, immutable wrapper around the python-chess board.
    """
    def __init__(self, board: Optional[chess.Board] = None):
        self._board = board if board is not None else chess.Board()
        self._fen = self._board.fen()
        self._hash = hash(self._fen)

    @property
    def fen(self) -> str:
        return self._fen

    @property
    def turn(self) -> chess.Color:
        return self._board.turn

    def get_legal_moves(self) -> List[chess.Move]:
        return list(self._board.legal_moves)

    def apply_move(self, move: chess.Move) -> 'Board':
        new_board = self._board.copy(stack=False)
        new_board.push(move)
        return Board(new_board)

    def is_game_over(self) -> bool:
        """
        Checks if the game has ended by any means by explicitly checking all conditions.
        This is the most robust way to check for game termination.
        """
        # Use the canonical is_game_over() method from python-chess,
        # which correctly checks for stalemate, checkmate, and other draw conditions.
        return self._board.is_game_over(claim_draw=True)

    def get_game_outcome(self) -> Optional[float]:
        """
        Returns outcome from the current player's perspective: 1.0 win, -1.0 loss, 0.0 draw.
        """
        if not self.is_game_over():
            return None

        result = self._board.result(claim_draw=True)
        
        if result == "1-0":
            return 1.0 if self.turn == chess.WHITE else -1.0
        elif result == "0-1":
            return -1.0 if self.turn == chess.WHITE else 1.0
        else:
            return 0.0

    @staticmethod
    @lru_cache(maxsize=100000)
    def _get_tensor_from_fen(fen: str) -> torch.Tensor:
        """Converts a FEN string into a (17, 8, 8) tensor using efficient bitboard operations."""
        board = chess.Board(fen)
        tensor = np.zeros((17, 8, 8), dtype=np.float32)

        for piece_type in chess.PIECE_TYPES:
            white_pieces = board.pieces(piece_type, chess.WHITE)
            black_pieces = board.pieces(piece_type, chess.BLACK)
            for square_idx in range(64):
                row, col = square_idx // 8, square_idx % 8
                if (white_pieces >> square_idx) & 1:
                    tensor[piece_type - 1, row, col] = 1.0
                if (black_pieces >> square_idx) & 1:
                    tensor[6 + piece_type - 1, row, col] = 1.0

        if board.turn == chess.WHITE: tensor[12] = 1.0
        if board.has_kingside_castling_rights(chess.WHITE): tensor[13] = 1.0
        if board.has_queenside_castling_rights(chess.WHITE): tensor[14] = 1.0
        if board.has_kingside_castling_rights(chess.BLACK): tensor[15] = 1.0
        if board.has_queenside_castling_rights(chess.BLACK): tensor[16] = 1.0
            
        return torch.from_numpy(tensor)

    def to_model_input(self) -> torch.Tensor:
        return Board._get_tensor_from_fen(self._fen)

    def __hash__(self):
        return self._hash

    def __eq__(self, other):
        return isinstance(other, Board) and self._fen == other._fen

    def __repr__(self):
        return f"<Board: {self.fen}>"