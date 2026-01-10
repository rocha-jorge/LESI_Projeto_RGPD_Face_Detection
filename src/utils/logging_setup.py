import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

def init_logging(log_dir: Path, log_file: Path, level: int = logging.INFO) -> None:
    """Initialize structured logging with a rotating file handler and console output.

    - log_dir: directory to store log files (created if missing)
    - log_file: full path to the main log file
    - level: logging level, default INFO
    """
    log_dir.mkdir(exist_ok=True)
    logger = logging.getLogger()
    logger.setLevel(level)

    file_handler = RotatingFileHandler(log_file, maxBytes=2_000_000, backupCount=3, encoding="utf-8")
    file_formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    file_handler.setFormatter(file_formatter)

    # File + console logging: single file on disk, live terminal output
    logger.handlers = []
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_formatter = logging.Formatter("%(levelname)s | %(message)s")
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
