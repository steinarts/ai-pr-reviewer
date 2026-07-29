import subprocess
from pathlib import Path


def initialize_repo(path: Path) -> None:
    subprocess.run(["git", "init", "-b", "main"], cwd=str(path), check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=str(path), check=True)
    subprocess.run(["git", "config", "user.name", "Test User"], cwd=str(path), check=True)
