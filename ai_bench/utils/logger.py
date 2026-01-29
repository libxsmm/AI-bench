import logging
import os


def setup_logger(name="ai_bench", level=logging.INFO):
    logger = logging.getLogger(name)
    if not logger.hasHandlers():
        # Disable propagation due to possible message duplication from PyTorch's logger.
        logger.propagate = False
        handler = logging.StreamHandler()
        formatter = logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    log_level = os.environ.get("AIBENCH_LOG", logging.getLevelName(level)).upper()
    logger.setLevel(log_level)
    return logger
