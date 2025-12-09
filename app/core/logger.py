# app/core/logger.py
import logging
import sys

_logger = logging.getLogger("agnel_ocr")
if not _logger.handlers:
    _logger.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    _logger.addHandler(handler)


def get_logger(name: str | None = None) -> logging.Logger:
    if name:
        return _logger.getChild(name)
    return _logger
