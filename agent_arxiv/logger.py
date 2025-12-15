import logging
from typing import Optional

_BASE_LOGGER: Optional[logging.Logger] = None


def _configure_base_logger() -> logging.Logger:
    logger = logging.getLogger("agent_arxiv")
    if logger.handlers:
        return logger

    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    return logger


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Return a project logger, creating it if needed."""
    global _BASE_LOGGER
    if _BASE_LOGGER is None:
        _BASE_LOGGER = _configure_base_logger()

    if name and name != _BASE_LOGGER.name:
        return _BASE_LOGGER.getChild(name)

    return _BASE_LOGGER
