import logging
import os

def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.propagate = False

    # Prevent duplicate handlers if called multiple times
    if logger.handlers:
        return logger

    if os.getenv(name) == "1":
        logger.setLevel(logging.INFO)

        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
        )
        handler.setFormatter(formatter)

        logger.addHandler(handler)
    else:
        # Disable all logging for this logger
        logger.addHandler(logging.NullHandler())

    return logger


perf_logger = get_logger("PERF_LOGGER")
