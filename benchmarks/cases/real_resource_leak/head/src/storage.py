from pathlib import Path


def write_status(path: Path, payload: str, should_skip: bool) -> None:
    handle = path.open("w", encoding="utf-8")
    if should_skip:
        # Defect: early return leaks the opened file handle.
        return
    try:
        handle.write(payload)
    finally:
        handle.close()
