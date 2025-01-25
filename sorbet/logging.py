import logging
from pathlib import Path


def start(msg: str) -> None:
    logging.info(f"Starting {msg}...")


def end(msg: str) -> None:
    logging.info(f"Finished {msg}.")


def cwd() -> None:
    logging.info(f"Current working directory: {Path.cwd()}")
