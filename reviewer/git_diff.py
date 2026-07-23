from __future__ import annotations

import fnmatch
import re
import subprocess
from pathlib import Path

from .models import DiffFile, DiffSnapshot

MAX_FILES_DEFAULT = 30
MAX_DIFF_LINES_DEFAULT = 3000

EXCLUDED_GLOBS = {
    "*.env",
    ".env",
    ".env.*",
    "*.db",
    "*.sqlite",
    "*.lock",
    "poetry.lock",
    "package-lock.json",
    "yarn.lock",
    "pnpm-lock.yaml",
    "*.png",
    "*.jpg",
    "*.jpeg",
    "*.gif",
    "*.webp",
    "*.pdf",
    "*.zip",
    "*.exe",
    "*.dll",
    "*.so",
}

EXCLUDED_PARTS = {
    "node_modules",
    ".venv",
    "__pycache__",
    ".git",
    "secrets",
    "generated",
}


class GitDiffError(RuntimeError):
    pass


def _run_git(args: list[str], cwd: Path) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=str(cwd),
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise GitDiffError(result.stderr.strip() or result.stdout.strip() or "git command failed")
    return result.stdout


def _is_excluded(path: Path) -> bool:
    normalized = path.as_posix()
    parts = set(normalized.split("/"))
    if parts & EXCLUDED_PARTS:
        return True
    return any(fnmatch.fnmatch(normalized, pattern) for pattern in EXCLUDED_GLOBS)


def _validate_ref(ref: str, cwd: Path) -> None:
    _run_git(["rev-parse", "--verify", ref], cwd)


def _parse_name_status(text: str) -> list[DiffFile]:
    files: list[DiffFile] = []
    for raw in text.splitlines():
        if not raw.strip():
            continue
        cols = raw.split("\t")
        status = cols[0]
        code = status[0]

        if code == "R" and len(cols) >= 3:
            old_path, new_path = cols[1], cols[2]
            files.append(DiffFile(status=status, old_path=Path(old_path), path=Path(new_path)))
            continue

        if len(cols) < 2:
            continue

        path = cols[1]
        files.append(DiffFile(status=status, old_path=None, path=Path(path)))
    return files


def _parse_changed_lines(diff_text: str) -> dict[str, set[int]]:
    changed: dict[str, set[int]] = {}
    current_file: str | None = None
    current_new_line = 0

    hunk_re = re.compile(r"@@ -\d+(?:,\d+)? \+(\d+)(?:,\d+)? @@")

    for line in diff_text.splitlines():
        if line.startswith("+++ b/"):
            current_file = line.removeprefix("+++ b/")
            changed.setdefault(current_file, set())
            continue

        if line.startswith("@@"):
            match = hunk_re.search(line)
            if match:
                current_new_line = int(match.group(1))
            continue

        if current_file is None:
            continue

        if line.startswith("+") and not line.startswith("+++"):
            changed[current_file].add(current_new_line)
            current_new_line += 1
            continue

        if line.startswith("-") and not line.startswith("---"):
            continue

        if not line.startswith("\\"):
            current_new_line += 1

    return changed


def collect_diff(
    base: str,
    head: str,
    cwd: Path,
    max_files: int = MAX_FILES_DEFAULT,
    max_diff_lines: int = MAX_DIFF_LINES_DEFAULT,
) -> DiffSnapshot:
    _validate_ref(base, cwd)
    _validate_ref(head, cwd)

    name_status = _run_git(["diff", "--name-status", f"{base}...{head}"], cwd)
    files = _parse_name_status(name_status)

    filtered = [f for f in files if not _is_excluded(f.path)]
    filtered = [f for f in filtered if not f.status.startswith("D")]

    if len(filtered) > max_files:
        raise GitDiffError(f"Too many changed files ({len(filtered)}), limit is {max_files}")

    paths = [f.path.as_posix() for f in filtered]
    if not paths:
        return DiffSnapshot(
            base=base,
            head=head,
            changed_files=[],
            diff_text="",
            diff_lines=0,
        )

    diff_args = ["diff", "--unified=80", "--no-color", f"{base}...{head}"]
    diff_args.extend(["--", *paths])

    diff_text = _run_git(diff_args, cwd)
    diff_lines = len(diff_text.splitlines())

    if diff_lines > max_diff_lines:
        raise GitDiffError(f"Too many diff lines ({diff_lines}), limit is {max_diff_lines}")

    changed_lines_map = _parse_changed_lines(diff_text)
    for file in filtered:
        file.changed_lines = changed_lines_map.get(file.path.as_posix(), set())

    return DiffSnapshot(
        base=base,
        head=head,
        changed_files=filtered,
        diff_text=diff_text,
        diff_lines=diff_lines,
    )
