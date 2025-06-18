# utils.py
# For miscellaneous utility functions, potentially logging setup if not too complex.

import logging
import sys
import torch
import config # For LOG_LEVEL

def setup_logging(log_level_str: str = config.LOG_LEVEL, log_file: str = "alphazero_chess.log"):
    """
    Sets up basic logging for the project.
    """
    log_level = getattr(logging, log_level_str.upper(), logging.INFO)
    
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.FileHandler(log_file, mode='a'), # Append to log file
            logging.StreamHandler(sys.stdout)      # Also print to console
        ]
    )
    # Suppress overly verbose logs from libraries if needed
    # logging.getLogger("PIL").setLevel(logging.WARNING)
    # logging.getLogger("matplotlib").setLevel(logging.WARNING)
    
    logger = logging.getLogger(config.PROJECT_NAME)
    logger.info("Logging setup complete.")
    return logger

def get_device(device_str: str = config.DEVICE) -> torch.device:
    """Gets the torch device based on configuration and availability."""
    if device_str.lower() == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    elif device_str.lower() == "mps" and torch.backends.mps.is_available(): # For Apple Silicon
        return torch.device("mps")
    else:
        if device_str.lower() == "cuda":
            print("CUDA specified but not available. Falling back to CPU.")
        elif device_str.lower() == "mps":
            print("MPS specified but not available. Falling back to CPU.")
        return torch.device("cpu")

# Add other utility functions as needed:
# - Elo calculation based on match results
# - Functions to save/load generic checkpoints
# - Plotting utilities (if not using TensorBoard/WandB directly for everything)

if __name__ == '__main__':
    logger = setup_logging(log_level_str="DEBUG")
    
    logger.debug("This is a debug message.")
    logger.info("This is an info message.")
    logger.warning("This is a warning message.")
    logger.error("This is an error message.")
    
    device = get_device()
    logger.info(f"Selected device: {device}")

    device_cuda = get_device("cuda") # Test fallback if CUDA not avail
    logger.info(f"Attempted CUDA, selected device: {device_cuda}")

    print("\nUtils testing completed. Check 'alphazero_chess.log' for file output.")
