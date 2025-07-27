# src/davis_ai/utils/move_encoder.py

import chess
import numpy as np

# Use a relative import because we are in the same 'utils' package
from . import constants 
from .exceptions import DavisMoveEncodingError

# --- Move to Policy Index Mapping ---

# Pre-calculate knight move plane mapping for faster lookup
# This maps the (row_delta, col_delta) to the specific knight move plane index
_KNIGHT_MOVE_MAP = {offset: i for i, offset in enumerate(constants.KNIGHT_MOVE_OFFSETS)}

def _get_queen_move_direction_plane(dx: int, dy: int) -> int:
    """Calculates the direction index (0-7) for a queen-like move."""
    if dx > 0 and dy == 0: return 0  # North
    if dx > 0 and dy > 0: return 1  # Northeast
    if dx == 0 and dy > 0: return 2  # East
    if dx < 0 and dy > 0: return 3  # Southeast
    if dx < 0 and dy == 0: return 4  # South
    if dx < 0 and dy < 0: return 5  # Southwest
    if dx == 0 and dy < 0: return 6  # West
    if dx > 0 and dy < 0: return 7  # Northwest
    return -1 # Should not happen

def move_to_policy_index(move: chess.Move) -> int:
    """
    Converts a chess.Move object to its corresponding policy index (0-4671).
    This encoding is based on the AlphaZero methodology.
    """
    from_square = move.from_square
    to_square = move.to_square

    # --- 1. Underpromotion Moves ---
    if move.promotion and move.promotion != chess.QUEEN:
        try:
            promo_piece_idx = constants.UNDERPROMOTION_PIECES.index(move.promotion)
            
            # CORRECTED LOGIC: Use column difference for direction
            from_col = chess.square_file(from_square)
            to_col = chess.square_file(to_square)
            
            if from_col > to_col: promo_dir_idx = 0  # Capture left
            elif from_col == to_col: promo_dir_idx = 1 # Forward
            else: promo_dir_idx = 2 # Capture right

            plane_idx = promo_piece_idx * 3 + promo_dir_idx
            base_offset = constants.QUEEN_MOVE_PLANES + constants.KNIGHT_MOVE_PLANES
            
            return (base_offset + plane_idx) * 64 + from_square
        except (ValueError, IndexError):
            raise DavisMoveEncodingError(f"Could not encode underpromotion move: {move.uci()}")

    from_row, from_col = from_square // 8, from_square % 8
    to_row, to_col = to_square // 8, to_square % 8
    dx, dy = to_row - from_row, to_col - from_col

    # --- 2. Knight Moves ---
    if (abs(dx), abs(dy)) in [(1, 2), (2, 1)]:
        plane_idx = _KNIGHT_MOVE_MAP.get((dx, dy), -1)
        if plane_idx != -1:
            base_offset = constants.QUEEN_MOVE_PLANES
            return (base_offset + plane_idx) * 64 + from_square
        
    # --- 3. Queen-like Moves (including normal promotions) ---
    distance = max(abs(dx), abs(dy))
    if distance > 0:
        direction_idx = _get_queen_move_direction_plane(dx, dy)
        if direction_idx != -1:
            plane_idx = direction_idx * constants.QUEEN_MOVE_MAX_DISTANCE + (distance - 1)
            return plane_idx * 64 + from_square

    raise DavisMoveEncodingError(f"Could not encode move: {move.uci()}")

def policy_index_to_move(policy_index: int, board: chess.Board) -> chess.Move:
    """
    Converts a policy index back into a chess.Move object.
    This is complex and generally not needed for MCTS, as we iterate through
    legal moves and encode them, rather than decoding the model's full output.
    """
    raise NotImplementedError("Decoding the full policy head is not the standard approach.")