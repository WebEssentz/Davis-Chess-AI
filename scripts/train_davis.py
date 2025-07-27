# scripts/train_davis.py

import sys
import os

# This is a common pattern to ensure the script can find the 'src' directory
# when run from the command line.
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.davis_ai.correction import ReinforcementLearner
from src.davis_ai.config import DAVIS_CONFIG
from src.davis_ai.utils import setup_logger

def main():
    """
    The main entry point for starting the Davis AI training process.
    """
    # 1. Setup the logger to capture all output to a file
    # This is crucial for long training runs.
    logger = setup_logger(log_dir="logs", log_file="training_run.log")
    logger.info("==================================================")
    logger.info("           STARTING DAVIS AI TRAINING RUN           ")
    logger.info("==================================================")

    # 2. Initialize the master Reinforcement Learner with the global config
    try:
        learner = ReinforcementLearner(config=DAVIS_CONFIG)
    except Exception as e:
        logger.exception("Failed to initialize ReinforcementLearner. Exiting.")
        return

    # 3. Start the main training loop
    # This will run for the number of generations specified in the config.
    # It will handle creating an initial model if one doesn't exist.
    try:
        logger.info("Starting the main reinforcement learning loop...")
        learner.run_loop()
        logger.info("Training loop completed successfully.")
    except KeyboardInterrupt:
        logger.warning("Training run interrupted by user (Ctrl+C). Exiting.")
    except Exception as e:
        logger.exception("A fatal error occurred during the training loop.")
    finally:
        logger.info("==================================================")
        logger.info("            DAVIS AI TRAINING RUN ENDED             ")
        logger.info("==================================================")


if __name__ == '__main__':
    # This makes the script executable from the command line.
    main()