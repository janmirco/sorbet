import logging
from pathlib import Path

from main import main
from sorbet.logging import log_end, log_start
from sorbet.paths import setup_paths

if __name__ == "__main__":
    output_dir = setup_paths()
    logging.basicConfig(
        level=logging.INFO,
        format="[%(levelname)s] %(asctime)s - %(message)s",
        handlers=[
            logging.StreamHandler(),  # terminal output
            logging.FileHandler(output_dir / Path("app.log")),  # file output
        ],
    )
    sorbet_logo = r"""
       _____            __         __
      / ___/____  _____/ /_  ___  / /_
      \__ \/ __ \/ ___/ __ \/ _ \/ __/
     ___/ / /_/ / /  / /_/ /  __/ /_
    /____/\____/_/  /_.___/\___/\__/
    """
    logging.info(sorbet_logo)
    section = "main"
    log_start(section)
    main()
    log_end(section)
