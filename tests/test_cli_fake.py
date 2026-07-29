from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

from reviewer.cli import main, parse_args


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
            "--dry-run",
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


def test_cli_verification_disabled_keeps_candidate_pipeline_unchanged(
    tmp_path: Path,
    monkeypatch,
) -> None:
    repo = _init_repo(tmp_path)
    output = tmp_path / "review-result-no-verify.json"

    monkeypatch.chdir(repo)
    exit_code = main(
        [
            "--base",
            "HEAD~1",
            "--head",
            "HEAD",
            "--dry-run",
            "--output",
            str(output),
        ]
    )

    assert exit_code == 0
    data = json.loads(output.read_text(encoding="utf-8"))
    assert data["metadata"]["verification_enabled"] is False
    assert data["metadata"]["verification_requests_planned"] == 0
    assert data["metadata"]["verification_requests_completed"] == 0
    assert data["metadata"]["verification_requests_failed"] == 0
    assert data["metadata"]["verification_requests_skipped"] == 0
    assert data["candidate_findings"] == data["verified_findings"]


def test_cli_handles_invalid_json_from_llm(tmp_path: Path, monkeypatch) -> None:
    repo = _init_repo(tmp_path)
    output = tmp_path / "review-result-invalid.json"

    from reviewer import llm_client as llm_module

    monkeypatch.chdir(repo)
    monkeypatch.setattr(
        llm_module.FakeLLMClient,
        "generate",
        lambda self, system_prompt, user_prompt, response_schema=None: "{not-json",
    )

    exit_code = main(
        [
            "--base",
            "HEAD~1",
            "--head",
            "HEAD",
            "--dry-run",
            "--output",
            str(output),
        ]
    )

    assert exit_code == 2
    assert output.exists()
    data = json.loads(output.read_text(encoding="utf-8"))
    assert data["metadata"]["completed_requests"] == 0
    assert data["metadata"]["failed_requests"] >= 1


def test_cli_handles_empty_findings_list(tmp_path: Path, monkeypatch) -> None:
    repo = _init_repo(tmp_path)
    output = tmp_path / "review-result-empty.json"

    from reviewer import llm_client as llm_module

    monkeypatch.chdir(repo)
    monkeypatch.setattr(
        llm_module.FakeLLMClient,
        "generate",
        lambda self, system_prompt, user_prompt, response_schema=None: '{"findings": []}',
    )

    exit_code = main(
        [
            "--base",
            "HEAD~1",
            "--head",
            "HEAD",
            "--dry-run",
            "--output",
            str(output),
        ]
    )

    assert exit_code == 0
    data = json.loads(output.read_text(encoding="utf-8"))
    assert data["accepted_findings"] == []
    assert data["rejected_findings"] == []


def test_cli_handles_timeout_from_llm(tmp_path: Path, monkeypatch) -> None:
    repo = _init_repo(tmp_path)
    output = tmp_path / "review-result-timeout.json"

    from reviewer import llm_client as llm_module

    monkeypatch.chdir(repo)
    monkeypatch.setattr(
        llm_module.FakeLLMClient,
        "generate",
        lambda self, system_prompt, user_prompt, response_schema=None: (_ for _ in ()).throw(
            TimeoutError("timed out")
        ),
    )

    exit_code = main(
        [
            "--base",
            "HEAD~1",
            "--head",
            "HEAD",
            "--dry-run",
            "--output",
            str(output),
        ]
    )

    assert exit_code == 2
    assert output.exists()
    data = json.loads(output.read_text(encoding="utf-8"))
    assert data["metadata"]["completed_requests"] == 0
    assert data["metadata"]["failed_requests"] >= 1


def test_cli_writes_llm_debug_log(tmp_path: Path, monkeypatch) -> None:
    repo = _init_repo(tmp_path)
    output = tmp_path / "review-result-debug.json"
    debug_log = tmp_path / "llm-debug.jsonl"

    monkeypatch.chdir(repo)
    exit_code = main(
        [
            "--base",
            "HEAD~1",
            "--head",
            "HEAD",
            "--dry-run",
            "--llm-debug",
            "--llm-debug-log",
            str(debug_log),
            "--output",
            str(output),
        ]
    )

    assert exit_code == 0
    assert output.exists()
    assert debug_log.exists()
    lines = debug_log.read_text(encoding="utf-8").strip().splitlines()
    assert any('"event": "reviewer_start"' in line for line in lines)
    assert any('"event": "reviewer_complete"' in line for line in lines)


def test_cli_rejects_invalid_prompt_budget(tmp_path: Path, monkeypatch) -> None:
    repo = _init_repo(tmp_path)
    output = tmp_path / "review-result-invalid-budget.json"

    monkeypatch.chdir(repo)
    exit_code = main(
        [
            "--base",
            "HEAD~1",
            "--head",
            "HEAD",
            "--dry-run",
            "--max-prompt-tokens",
            "0",
            "--output",
            str(output),
        ]
    )

    assert exit_code == 2
    assert not output.exists()


def test_cli_exit_zero_on_partial_success(tmp_path: Path, monkeypatch) -> None:
    repo = _init_repo(tmp_path)
    output = tmp_path / "review-result-partial.json"

    from reviewer import llm_client as llm_module

    monkeypatch.chdir(repo)
    calls = {"count": 0}

    def flaky_generate(self, system_prompt, user_prompt, response_schema=None):
        calls["count"] += 1
        if calls["count"] == 1:
            raise TimeoutError("timed out")
        return '{"findings": []}'

    monkeypatch.setattr(llm_module.FakeLLMClient, "generate", flaky_generate)

    exit_code = main(
        [
            "--base",
            "HEAD~1",
            "--head",
            "HEAD",
            "--dry-run",
            "--review-mode",
            "separate",
            "--max-prompt-tokens",
            "500",
            "--output",
            str(output),
        ]
    )

    assert exit_code == 0
    data = json.loads(output.read_text(encoding="utf-8"))
    assert data["metadata"]["failed_requests"] >= 1
    assert data["metadata"]["completed_requests"] >= 1
    assert data["metadata"]["reviewer_failures"]


def test_cli_exit_two_when_all_requests_fail(tmp_path: Path, monkeypatch) -> None:
    repo = _init_repo(tmp_path)
    output = tmp_path / "review-result-all-fail.json"

    from reviewer import llm_client as llm_module

    monkeypatch.chdir(repo)

    monkeypatch.setattr(
        llm_module.FakeLLMClient,
        "generate",
        lambda self, system_prompt, user_prompt, response_schema=None: (_ for _ in ()).throw(
            TimeoutError("timed out")
        ),
    )

    exit_code = main(
        [
            "--base",
            "HEAD~1",
            "--head",
            "HEAD",
            "--dry-run",
            "--max-prompt-tokens",
            "500",
            "--output",
            str(output),
        ]
    )

    assert exit_code == 2


def test_cli_default_prompt_budget_is_5000() -> None:
    args = parse_args(["--base", "main", "--head", "HEAD"])
    assert args.max_prompt_tokens == 3500
    assert args.review_mode == "consolidated"
    assert args.llm_timeout == 180.0
    assert args.max_review_seconds == 900.0
    assert args.test_max_prompt_tokens == 2500


def test_cli_rejects_invalid_output_token_budget(tmp_path: Path, monkeypatch) -> None:
    repo = _init_repo(tmp_path)
    output = tmp_path / "review-result-invalid-output-budget.json"

    monkeypatch.chdir(repo)
    exit_code = main(
        [
            "--base",
            "HEAD~1",
            "--head",
            "HEAD",
            "--dry-run",
            "--llm-max-output-tokens",
            "0",
            "--output",
            str(output),
        ]
    )

    assert exit_code == 2
    assert not output.exists()


def test_cli_allows_override_of_new_defaults(tmp_path: Path, monkeypatch) -> None:
    repo = _init_repo(tmp_path)
    output = tmp_path / "review-result-overrides.json"

    monkeypatch.chdir(repo)
    exit_code = main(
        [
            "--base",
            "HEAD~1",
            "--head",
            "HEAD",
            "--dry-run",
            "--review-mode",
            "separate",
            "--llm-timeout",
            "12",
            "--max-prompt-tokens",
            "777",
            "--test-max-prompt-tokens",
            "555",
            "--max-review-seconds",
            "30",
            "--output",
            str(output),
        ]
    )

    assert exit_code in {0, 2}
    data = json.loads(output.read_text(encoding="utf-8"))
    assert data["metadata"]["review_mode"] == "separate"
    assert data["metadata"]["total_time_budget_seconds"] == 30.0


def test_cli_writes_partial_output_when_total_budget_exhausted(tmp_path: Path, monkeypatch) -> None:
    repo = _init_repo(tmp_path)
    output = tmp_path / "review-result-budget.json"

    from reviewer import llm_client as llm_module

    monkeypatch.chdir(repo)
    calls = {"count": 0}

    def slow_then_fast(self, system_prompt, user_prompt, response_schema=None):
        calls["count"] += 1
        if calls["count"] == 1:
            time.sleep(0.02)
            return '{"findings": []}'
        raise TimeoutError("should be skipped by total budget")

    monkeypatch.setattr(llm_module.FakeLLMClient, "generate", slow_then_fast)

    (repo / "worker2.py").write_text("y = 3\n", encoding="utf-8")
    _run(["git", "add", "."], repo)
    _run(["git", "commit", "-m", "change 2"], repo)

    exit_code = main(
        [
            "--base",
            "HEAD~1",
            "--head",
            "HEAD",
            "--dry-run",
            "--review-mode",
            "separate",
            "--max-prompt-tokens",
            "300",
            "--max-review-seconds",
            "0.01",
            "--output",
            str(output),
        ]
    )

    assert output.exists()
    data = json.loads(output.read_text(encoding="utf-8"))
    assert data["metadata"]["planned_requests"] >= 1
    assert data["metadata"]["completed_requests"] >= 0
    assert data["metadata"]["skipped_requests"] >= 1
    assert exit_code in {0, 2}


def test_cli_writes_partial_output_when_total_budget_exhausted_single_chunk(
    tmp_path: Path,
    monkeypatch,
) -> None:
    repo = _init_repo(tmp_path)
    output = tmp_path / "review-result-budget-single.json"

    monkeypatch.chdir(repo)
    exit_code = main(
        [
            "--base",
            "HEAD~1",
            "--head",
            "HEAD",
            "--dry-run",
            "--max-review-seconds",
            "0.001",
            "--output",
            str(output),
        ]
    )

    assert output.exists()
    data = json.loads(output.read_text(encoding="utf-8"))
    assert data["metadata"]["planned_requests"] == 1
    assert exit_code in {0, 2}


def test_cli_summary_contains_planned_and_completed_counts(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    repo = _init_repo(tmp_path)
    output = tmp_path / "review-result-summary.json"

    monkeypatch.chdir(repo)
    main(
        [
            "--base",
            "HEAD~1",
            "--head",
            "HEAD",
            "--dry-run",
            "--output",
            str(output),
        ]
    )

    out = capsys.readouterr().out
    assert "Planned LLM requests:" in out
    assert "Completed LLM requests:" in out
