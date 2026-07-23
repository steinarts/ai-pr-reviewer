from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from reviewer.git_diff import GitDiffError, collect_diff


def _run(cmd: list[str], cwd: Path) -> None:
    subprocess.run(cmd, cwd=str(cwd), check=True, capture_output=True, text=True)


def _init_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    _run(["git", "init", "-b", "main"], repo)
    _run(["git", "config", "user.email", "test@example.com"], repo)
    _run(["git", "config", "user.name", "Test User"], repo)

    (repo / "app.py").write_text("def add(a, b):\n    return a + b\n", encoding="utf-8")
    _run(["git", "add", "."], repo)
    _run(["git", "commit", "-m", "init"], repo)

    (repo / "app.py").write_text(
        "def add(a, b):\n    if a is None:\n        return b\n    return a + b\n",
        encoding="utf-8",
    )
    _run(["git", "add", "."], repo)
    _run(["git", "commit", "-m", "change"], repo)
    return repo


def test_collect_diff_reads_files_and_lines(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path)
    snap = collect_diff(base="HEAD~1", head="HEAD", cwd=repo)

    assert snap.diff_lines > 0
    assert len(snap.changed_files) == 1
    assert snap.changed_files[0].path == Path("app.py")
    assert len(snap.changed_files[0].changed_lines) > 0


def test_collect_diff_handles_empty_diff(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path)
    snap = collect_diff(base="HEAD", head="HEAD", cwd=repo)

    assert snap.changed_files == []
    assert snap.diff_lines == 0
    assert snap.diff_text == ""


def test_collect_diff_invalid_ref_raises(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path)

    with pytest.raises(GitDiffError):
        collect_diff(base="does-not-exist", head="HEAD", cwd=repo)


def test_collect_diff_excludes_env_files(tmp_path: Path) -> None:
    repo = tmp_path / "repo_env"
    repo.mkdir()
    _run(["git", "init", "-b", "main"], repo)
    _run(["git", "config", "user.email", "test@example.com"], repo)
    _run(["git", "config", "user.name", "Test User"], repo)

    (repo / "a.py").write_text("x = 1\n", encoding="utf-8")
    _run(["git", "add", "."], repo)
    _run(["git", "commit", "-m", "init"], repo)

    (repo / ".env").write_text("SECRET=1\n", encoding="utf-8")
    _run(["git", "add", "."], repo)
    _run(["git", "commit", "-m", "env"], repo)

    snap = collect_diff(base="HEAD~1", head="HEAD", cwd=repo)
    assert snap.changed_files == []
    assert snap.diff_lines == 0


def test_collect_diff_ignores_deleted_files(tmp_path: Path) -> None:
    repo = tmp_path / "repo_deleted"
    repo.mkdir()
    _run(["git", "init", "-b", "main"], repo)
    _run(["git", "config", "user.email", "test@example.com"], repo)
    _run(["git", "config", "user.name", "Test User"], repo)

    (repo / "old.py").write_text("x = 1\n", encoding="utf-8")
    _run(["git", "add", "."], repo)
    _run(["git", "commit", "-m", "init"], repo)

    (repo / "old.py").unlink()
    _run(["git", "add", "-A"], repo)
    _run(["git", "commit", "-m", "delete"], repo)

    snap = collect_diff(base="HEAD~1", head="HEAD", cwd=repo)
    assert snap.changed_files == []
    assert snap.diff_lines == 0


def test_collect_diff_keeps_renamed_file(tmp_path: Path) -> None:
    repo = tmp_path / "repo_rename"
    repo.mkdir()
    _run(["git", "init", "-b", "main"], repo)
    _run(["git", "config", "user.email", "test@example.com"], repo)
    _run(["git", "config", "user.name", "Test User"], repo)

    (repo / "before.py").write_text("x = 1\n", encoding="utf-8")
    _run(["git", "add", "."], repo)
    _run(["git", "commit", "-m", "init"], repo)

    _run(["git", "mv", "before.py", "after.py"], repo)
    _run(["git", "commit", "-m", "rename"], repo)

    snap = collect_diff(base="HEAD~1", head="HEAD", cwd=repo)
    assert len(snap.changed_files) == 1
    changed = snap.changed_files[0]
    assert changed.status.startswith("R")
    assert changed.old_path == Path("before.py")
    assert changed.path == Path("after.py")
