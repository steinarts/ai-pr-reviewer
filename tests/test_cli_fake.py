from __future__ import annotations

import json
import subprocess
from pathlib import Path

from reviewer.cli import main


def _run(cmd: list[str], cwd: Path) -> None:
    subprocess.run(cmd, cwd=str(cwd), check=True, capture_output=True, text=True)


def _init_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    _run(["git", "init", "-b", "main"], repo)
    _run(["git", "config", "user.email", "test@example.com"], repo)
    _run(["git", "config", "user.name", "Test User"], repo)

    (repo / "worker.py").write_text("x = 1\n", encoding="utf-8")
    _run(["git", "add", "."], repo)
    _run(["git", "commit", "-m", "init"], repo)

    (repo / "worker.py").write_text("x = 2\n", encoding="utf-8")
    _run(["git", "add", "."], repo)
    _run(["git", "commit", "-m", "change"], repo)
    return repo


def test_cli_fake_writes_json(tmp_path: Path, monkeypatch) -> None:
    repo = _init_repo(tmp_path)
    output = tmp_path / "review-result.json"

    monkeypatch.chdir(repo)
    exit_code = main(
        [
            "--base",
            "HEAD~1",
            "--head",
            "HEAD",
            "--fake-llm",
            "--output",
            str(output),
        ]
    )

    assert exit_code == 0
    assert output.exists()

    data = json.loads(output.read_text(encoding="utf-8"))
    assert "metadata" in data
    assert data["metadata"]["changed_files"] == 1
    assert "accepted_findings" in data
    assert "rejected_findings" in data


def test_cli_handles_invalid_json_from_llm(tmp_path: Path, monkeypatch) -> None:
    repo = _init_repo(tmp_path)
    output = tmp_path / "review-result-invalid.json"

    from reviewer import llm_client as llm_module

    monkeypatch.chdir(repo)
    monkeypatch.setattr(llm_module.FakeLLMClient, "review", lambda self, prompt: "{not-json")

    exit_code = main(
        [
            "--base",
            "HEAD~1",
            "--head",
            "HEAD",
            "--fake-llm",
            "--output",
            str(output),
        ]
    )

    assert exit_code == 0
    data = json.loads(output.read_text(encoding="utf-8"))
    assert data["accepted_findings"] == []
    assert data["rejected_findings"] == []


def test_cli_handles_empty_findings_list(tmp_path: Path, monkeypatch) -> None:
    repo = _init_repo(tmp_path)
    output = tmp_path / "review-result-empty.json"

    from reviewer import llm_client as llm_module

    monkeypatch.chdir(repo)
    monkeypatch.setattr(llm_module.FakeLLMClient, "review", lambda self, prompt: '{"findings": []}')

    exit_code = main(
        [
            "--base",
            "HEAD~1",
            "--head",
            "HEAD",
            "--fake-llm",
            "--output",
            str(output),
        ]
    )

    assert exit_code == 0
    data = json.loads(output.read_text(encoding="utf-8"))
    assert data["accepted_findings"] == []
    assert data["rejected_findings"] == []
