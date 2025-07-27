# src/davis_ai/utils/constants.py

"""
Holds static constants used throughout the Davis AI project.
This includes neural network architecture details and move encoding specifics.
"""

import chess

# --- Board Representation ---
# The number of channels for the neural network input tensor.
# Based on the AlphaZero paper: 8 past positions for each player + 1 for turn.
# Our implementation uses a simpler, yet effective representation:
# 6 piece types for White + 6 for Black + 5 game state planes = 17 channels.
INPUT_CHANNELS = 17
BOARD_DIMENSIONS = (8, 8)

# --- Move Encoding (for 4672-move policy head) ---
# Based on AlphaZero's move representation (73 planes x 64 squares = 4672 moves).
# See: https://arxiv.org/pdf/1712.01815.pdf (Page 13, "Representing moves")

# 1. Queen Moves (56 planes): Moves in 8 directions, up to 7 squares.
# plane_idx = direction_idx * 7 + (distance - 1)
# 8 directions: N, NE, E, SE, S, SW, W, NW.
QUEEN_MOVE_PLANES = 56
QUEEN_MOVE_MAX_DISTANCE = 7

# 2. Knight Moves (8 planes): 8 possible knight moves from any square.
KNIGHT_MOVE_PLANES = 8
KNIGHT_MOVE_OFFSETS = [
    # (row_delta, col_delta)
    (2, 1), (2, -1), (-2, 1), (-2, -1),
    (1, 2), (1, -2), (-1, 2), (-1, -2)
]

# 3. Underpromotion Moves (9 planes): Pawn promotions to Knight, Bishop, or Rook.
# 3 promo pieces x 3 directions (capture left, straight, capture right).
UNDERPROMOTION_PLANES = 9
# Queen promotions are handled by the standard queen move planes.
UNDERPROMOTION_PIECES = [chess.KNIGHT, chess.BISHOP, chess.ROOK]

# --- Total Policy Head Size ---
# (56 queen move planes + 8 knight move planes + 9 underpromo planes) * 64 squares
POLICY_HEAD_SIZE = (QUEEN_MOVE_PLANES + KNIGHT_MOVE_PLANES + UNDERPROMOTION_PLANES) * 64
# This should equal 73 * 64 = 4672