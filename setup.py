import logging
from pathlib import Path

import sorbet
from main import main

if __name__ == "__main__":
    output_dir = sorbet.paths.setup()
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
    sorbet.logging.start(section)
    main()
    sorbet.logging.end(section)
