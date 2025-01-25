from pathlib import Path


def setup() -> Path:
    output_dir = Path.cwd() / Path("output")
    if not output_dir.exists():
        output_dir.mkdir()
    return output_dir
