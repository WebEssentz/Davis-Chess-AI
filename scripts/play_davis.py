# scripts/play_davis.py

import sys
import os
import argparse
import chess

# Ensure the 'src' directory is in the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.davis_ai.engine import InferenceEngine, MonteCarloTreeSearch
from src.davis_ai.board import Board
from src.davis_ai.config import DAVIS_CONFIG
from src.davis_ai.utils import DavisConfigError

def print_board(board: chess.Board):
    """Prints the board to the console with rank and file labels."""
    # Use the unicode string representation from python-chess for a nice view
    print("\n" + str(board))
    print("-----------------")

def get_human_move(board: Board) -> chess.Move:
    """Prompts the human for a move and validates it."""
    while True:
        try:
            move_uci = input("Enter your move in UCI format (e.g., e2e4): ")
            move = chess.Move.from_uci(move_uci)
            if move in board.get_legal_moves():
                return move
            else:
                print("Invalid move. Please try again.")
        except ValueError:
            print("Invalid UCI format. Please try again (e.g., e2e4, g1f3, e7e8q).")
        except KeyboardInterrupt:
            print("\nExiting game.")
            sys.exit()

def main():
    """
    Main entry point for playing a game against Davis AI.
    """
    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(
        description="Play a game of chess against Davis AI."
    )
    parser.add_argument(
        "model_path",
        type=str,
        nargs='?', # Makes the model path optional
        default=os.path.join(DAVIS_CONFIG["model_directory"], DAVIS_CONFIG["best_model_filename"]),
        help="Path to the model checkpoint to play against. Defaults to the 'best_model.pth'."
    )
    parser.add_argument(
        "-s", "--sims",
        type=int,
        default=DAVIS_CONFIG["simulations_per_move"],
        help="Number of MCTS simulations for the AI's moves."
    )
    args = parser.parse_args()

    # --- Setup ---
    print("===================================")
    print("      Welcome to Project Davis     ")
    print("===================================")
    
    try:
        if not os.path.exists(args.model_path):
            raise DavisConfigError(f"Model path not found: {args.model_path}")
        
        print(f"Loading model: {args.model_path}")
        engine = InferenceEngine(model_path=args.model_path)
        
        mcts_config = DAVIS_CONFIG.copy()
        mcts_config['simulations_per_move'] = args.sims
        mcts = MonteCarloTreeSearch(engine, mcts_config)
        
        board = Board()
        
    except DavisConfigError as e:
        print(f"Error: {e}")
        sys.exit(1)

    # --- Game Loop ---
    while not board.is_game_over():
        print_board(board._board)
        
        if board.turn == chess.WHITE:
            print("Davis AI is thinking...")
            root_node = mcts.search(board, num_simulations=args.sims)
            # For competitive play, temperature is always 0
            ai_move = mcts.select_move(root_node, temperature=0)
            print(f"Davis AI plays: {ai_move.uci()}")
            board = board.apply_move(ai_move)
        else: # Human's turn (assuming human plays Black for this example)
            human_move = get_human_move(board)
            board = board.apply_move(human_move)

    # --- Game Over ---
    print_board(board._board)
    print("\n===================================")
    print("             GAME OVER             ")
    print(f"Result: {board._board.result(claim_draw=True)}")
    print("===================================")


if __name__ == '__main__':
    main()