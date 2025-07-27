# src/davis_ai/utils/logger.py

import logging
import os
import sys

LOG_LEVEL = logging.INFO

def setup_logger(log_dir: str = "logs", log_file: str = "davis_ai.log"):
    """
    Sets up a centralized logger for the Davis AI project.

    This will create a logger that outputs to both the console and a file.
    
    Args:
        log_dir: The directory to save log files in.
        log_file: The name of the log file.

    Returns:
        A configured logging.Logger instance.
    """
    # Create log directory if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)
    
    # Get the root logger for the project
    logger = logging.getLogger("davis_ai")
    logger.setLevel(LOG_LEVEL)

    # Prevent duplicate handlers if this function is called multiple times
    if logger.hasHandlers():
        logger.handlers.clear()

    # Create a formatter
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(name)s] - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Create a handler for console output (stdout)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(LOG_LEVEL)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Create a handler for file output
    file_handler = logging.FileHandler(os.path.join(log_dir, log_file), mode='a')
    file_handler.setLevel(LOG_LEVEL)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    return logger

# Example usage
if __name__ == '__main__':
    # You would typically call setup_logger() once at the start of your main script.
    # Then, in other modules, you get the logger by name.
    
    # --- In main script (e.g., train_davis.py) ---
    main_logger = setup_logger()
    main_logger.info("Logger has been set up. This is an informational message.")
    
    # --- In another module (e.g., davis_ai/engine/mcts.py) ---
    # You would just do this at the top of the file:
    # import logging
    # module_logger = logging.getLogger("davis_ai")
    
    # Simulating a log from another module:
    module_logger = logging.getLogger("davis_ai.mcts") # Use dotted notation for hierarchy
    module_logger.debug("This is a debug message. It won't show because level is INFO.")
    module_logger.warning("This is a warning from the MCTS module.")
    
    try:
        raise ValueError("Simulating an error.")
    except ValueError:
        module_logger.exception("An exception occurred! The logger will automatically add traceback info.")