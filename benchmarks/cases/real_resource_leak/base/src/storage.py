from pathlib import Path


def write_status(path: Path, payload: str, should_skip: bool) -> None:
    with path.open("w", encoding="utf-8") as handle:
        if should_skip:
            return
        handle.write(payload)
