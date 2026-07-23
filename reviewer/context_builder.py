from __future__ import annotations

import re
import subprocess
from pathlib import Path

from .models import DiffSnapshot, ReviewContext

MAX_CONTEXT_CHARS = 120_000


def _git_show_file(head: str, path: str, cwd: Path) -> str:
    result = subprocess.run(
        ["git", "show", f"{head}:{path}"],
        cwd=str(cwd),
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return ""
    return result.stdout


def _find_symbol_names(file_content: str) -> list[str]:
    symbols: list[str] = []
    pattern = re.compile(r"^\s*(def|class)\s+([A-Za-z_][A-Za-z0-9_]*)", re.MULTILINE)
    for match in pattern.finditer(file_content):
        symbols.append(match.group(2))
    return symbols[:30]


def _snippet_around_changes(content: str, changed_lines: set[int], window: int = 80) -> str:
    if not content or not changed_lines:
        return ""
    lines = content.splitlines()
    lo = max(1, min(changed_lines) - window)
    hi = min(len(lines), max(changed_lines) + window)
    selected = lines[lo - 1 : hi]
    rendered = [f"{idx + lo:5d}: {line}" for idx, line in enumerate(selected)]
    return "\n".join(rendered)


def build_context(snapshot: DiffSnapshot, head: str, cwd: Path) -> ReviewContext:
    contexts: dict[str, str] = {}
    used_chars = 0

    for file in snapshot.changed_files:
        file_path = file.path.as_posix()
        content = _git_show_file(head, file_path, cwd)
        symbols = _find_symbol_names(content)
        snippet = _snippet_around_changes(content, file.changed_lines)

        block = (
            f"FILE: {file_path}\n"
            f"STATUS: {file.status}\n"
            f"CHANGED_LINES: {sorted(file.changed_lines)}\n"
            f"SYMBOLS: {symbols}\n"
            f"SNIPPET:\n{snippet}\n"
        )

        block_len = len(block)
        if used_chars + block_len > MAX_CONTEXT_CHARS:
            break

        contexts[file_path] = block
        used_chars += block_len

    return ReviewContext(
        base=snapshot.base,
        head=snapshot.head,
        diff_text=snapshot.diff_text,
        file_contexts=contexts,
    )
