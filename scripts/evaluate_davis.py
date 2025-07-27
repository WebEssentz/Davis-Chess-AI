# scripts/evaluate_davis.py

import sys
import os
import argparse

# Ensure the 'src' directory is in the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.davis_ai.training import run_evaluation
from src.davis_ai.config import DAVIS_CONFIG
from src.davis_ai.utils import setup_logger, DavisConfigError

def main():
    """
    Main entry point for running a head-to-head evaluation between two Davis models.
    """
    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(
        description="Run a tournament between two Davis AI models to evaluate their relative strength."
    )
    parser.add_argument(
        "model1_path",
        type=str,
        help="Path to the first model checkpoint file (e.g., 'data/models/candidate.pth')."
    )
    parser.add_argument(
        "model2_path",
        type=str,
        help="Path to the second model checkpoint file (e.g., 'data/models/best_model.pth')."
    )
    parser.add_argument(
        "-n", "--num_games",
        type=int,
        help="Number of games to play in the tournament. Overrides the config value."
    )
    parser.add_argument(
        "-s", "--sims",
        type=int,
        help="Number of MCTS simulations per move. Overrides the config value."
    )
    args = parser.parse_args()

    # --- Logger Setup ---
    logger = setup_logger(log_dir="logs", log_file="evaluation_run.log")
    logger.info("=====================================================")
    logger.info("           STARTING DAVIS AI EVALUATION RUN          ")
    logger.info("=====================================================")

    # --- Configuration and Validation ---
    try:
        if not os.path.exists(args.model1_path):
            raise DavisConfigError(f"Model 1 path not found: {args.model1_path}")
        if not os.path.exists(args.model2_path):
            raise DavisConfigError(f"Model 2 path not found: {args.model2_path}")

        # Override config with command-line arguments if provided
        eval_config = DAVIS_CONFIG.copy()
        if args.num_games:
            eval_config['eval_num_games'] = args.num_games
            logger.info(f"Overriding number of games to: {args.num_games}")
        if args.sims:
            # Note: The evaluation script uses 'simulations_per_move' from the config
            eval_config['simulations_per_move'] = args.sims
            logger.info(f"Overriding simulations per move to: {args.sims}")

    except DavisConfigError as e:
        logger.error(f"Configuration Error: {e}")
        sys.exit(1)

    # --- Run Evaluation ---
    try:
        # We treat model1 as the "candidate" and model2 as the "best" for the purpose of the function
        # The result will show the win rate of model1 vs model2.
        run_evaluation(
            candidate_path=args.model1_path,
            best_path=args.model2_path,
            config=eval_config
        )
    except Exception as e:
        logger.exception("A fatal error occurred during the evaluation tournament.")
    finally:
        logger.info("=====================================================")
        logger.info("            DAVIS AI EVALUATION RUN ENDED            ")
        logger.info("=====================================================")

if __name__ == '__main__':
    main()