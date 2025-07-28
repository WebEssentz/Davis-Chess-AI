# src/davis_ai/utils/move_encoder.py (Final, Unified Version)
# Generated with 💚 by Avurna AI (2025)

import chess
import numpy as np
from . import constants 
from .exceptions import DavisMoveEncodingError

# --- SHARED HELPERS ---

# Pre-calculate knight move plane mapping for faster lookup
# Maps the (row_delta, col_delta) to the specific knight move plane index
_KNIGHT_MOVE_MAP = {offset: i for i, offset in enumerate(constants.KNIGHT_MOVE_OFFSETS)}

# Map of queen directions, crucial for both encoding and decoding
# The order is critical: N, NE, E, SE, S, SW, W, NW
_QUEEN_DIRECTIONS_MAP = [ # (row_delta, col_delta)
    (1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1)
]

# --- ENCODING: Move Object -> Policy Index ---

def _get_queen_move_direction_plane(dx: int, dy: int) -> int:
    """Calculates the direction index (0-7) for a queen-like move."""
    if dy > 0 and dx == 0: return 0  # N
    if dy > 0 and dx > 0: return 1  # NE
    if dy == 0 and dx > 0: return 2  # E
    if dy < 0 and dx > 0: return 3  # SE
    if dy < 0 and dx == 0: return 4  # S
    if dy < 0 and dx < 0: return 5  # SW
    if dy == 0 and dx < 0: return 6  # W
    if dy > 0 and dx < 0: return 7  # NW
    return -1 # Should not happen

def move_to_policy_index(move: chess.Move) -> int:
    """
    Converts a chess.Move object to its corresponding policy index (0-4671).
    """
    from_square = move.from_square
    to_square = move.to_square

    # --- 1. Underpromotion Moves ---
    if move.promotion and move.promotion != chess.QUEEN:
        try:
            promo_piece_idx = constants.UNDERPROMOTION_PIECES.index(move.promotion)
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
    dx, dy = to_col - from_col, to_row - from_row

    # --- 2. Knight Moves ---
    if (abs(dx), abs(dy)) in [(1, 2), (2, 1)]:
        # Note: We use (dy, dx) for knight moves to match the row/col delta convention
        plane_idx = _KNIGHT_MOVE_MAP.get((dy, dx), -1)
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

# --- DECODING: Policy Index -> Move Object ---

def policy_index_to_move(policy_index: int, board: chess.Board) -> chess.Move:
    """
    Decodes a policy index back into a chess.Move object for a given board.
    """
    plane = policy_index // 64
    from_square = policy_index % 64
    from_row, from_col = from_square // 8, from_square % 8

    # --- 1. Queen-like Moves (Planes 0-55) ---
    if 0 <= plane < constants.QUEEN_MOVE_PLANES:
        direction_idx = plane // constants.QUEEN_MOVE_MAX_DISTANCE
        distance = (plane % constants.QUEEN_MOVE_MAX_DISTANCE) + 1
        
        dy, dx = _QUEEN_DIRECTIONS_MAP[direction_idx]
        to_row, to_col = from_row + dy, from_col + dx
        to_square = to_row * 8 + to_col

        if not (0 <= to_row < 8 and 0 <= to_col < 8):
            raise DavisMoveEncodingError(f"Decoded queen move is out of bounds: {policy_index}")

        piece = board.piece_at(from_square)
        if piece and piece.piece_type == chess.PAWN and (to_row == 0 or to_row == 7):
            return chess.Move(from_square, to_square, promotion=chess.QUEEN)
        else:
            return chess.Move(from_square, to_square)

    # --- 2. Knight Moves (Planes 56-63) ---
    knight_base = constants.QUEEN_MOVE_PLANES
    if knight_base <= plane < knight_base + constants.KNIGHT_MOVE_PLANES:
        move_idx = plane - knight_base
        dy, dx = constants.KNIGHT_MOVE_OFFSETS[move_idx]
        to_row, to_col = from_row + dy, from_col + dx
        to_square = to_row * 8 + to_col

        if not (0 <= to_row < 8 and 0 <= to_col < 8):
            raise DavisMoveEncodingError(f"Decoded knight move is out of bounds: {policy_index}")
            
        return chess.Move(from_square, to_square)

    # --- 3. Underpromotion Moves (Planes 64-72) ---
    underpromo_base = knight_base + constants.KNIGHT_MOVE_PLANES
    if underpromo_base <= plane < underpromo_base + constants.UNDERPROMOTION_PLANES:
        plane_offset = plane - underpromo_base
        promo_piece_idx = plane_offset // 3
        promo_dir_idx = plane_offset % 3

        promotion_piece = constants.UNDERPROMOTION_PIECES[promo_piece_idx]
        dy = 1 if board.turn == chess.WHITE else -1
        dx = promo_dir_idx - 1 # -1 (left), 0 (fwd), 1 (right)
        
        to_row, to_col = from_row + dy, from_col + dx
        to_square = to_row * 8 + to_col
        
        if not (0 <= to_row < 8 and 0 <= to_col < 8):
            raise DavisMoveEncodingError(f"Decoded underpromotion move is out of bounds: {policy_index}")

        return chess.Move(from_square, to_square, promotion=promotion_piece)

    raise DavisMoveEncodingError(f"Could not decode policy index: {policy_index}")