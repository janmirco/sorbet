import logging
from pathlib import Path


def log_start(msg: str) -> None:
    logging.info(f"Starting {msg}...")


def log_end(msg: str) -> None:
    logging.info(f"Finished {msg}.")


def log_cwd() -> None:
    logging.info(f"Current working directory: {Path.cwd()}")
