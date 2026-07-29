import subprocess
from pathlib import Path


def initialize_repo(path: Path) -> None:
    subprocess.run(["git", "init", "-b", "main"], cwd=str(path), check=True)
