from __future__ import annotations

import multiprocessing
import time
from pathlib import Path

from reviewer.hard_timeout import run_with_hard_timeout
from reviewer.models import ReviewContext
from reviewer.scouts import REVIEW_MODE_CONSOLIDATED, REVIEW_MODE_SEPARATE, run_reviewers
from reviewer.token_utils import estimate_tokens


def _slow_worker(delay_seconds: float) -> str:
    time.sleep(delay_seconds)
    return '{"findings": []}'


class CountingLLMClient:
    def __init__(self, payload: str) -> None:
        self.payload = payload
        self.calls: list[dict[str, object]] = []
        self.model = "fake-model"
        self.timeout = 42.0

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        response_schema: dict[str, object] | None = None,
    ) -> str:
        self.calls.append(
            {
                "system": system_prompt,
                "user": user_prompt,
                "schema": response_schema,
            }
        )
        return self.payload


class SlowHardTimeoutClient:
    def __init__(self, slow_calls: int = 1) -> None:
        self.calls = 0
        self.slow_calls = slow_calls
        self.model = "fake-model"

    def generate_hard_timeout(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        response_schema: dict[str, object] | None = None,
        timeout_seconds: float,
    ) -> str:
        self.calls += 1
        if self.calls <= self.slow_calls:
            return run_with_hard_timeout(
                _slow_worker,
                10.0,
                timeout_seconds=timeout_seconds,
                timeout_error_message="simulated timeout",
            )
        return '{"findings": []}'


class SleepyClient:
    def __init__(self, sleep_seconds: float = 0.2) -> None:
        self.sleep_seconds = sleep_seconds
        self.calls = 0
        self.model = "fake-model"

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        response_schema: dict[str, object] | None = None,
    ) -> str:
        self.calls += 1
        time.sleep(self.sleep_seconds)
        return '{"findings": []}'


def _fake_context(file_count: int = 3, section_lines: int = 120) -> ReviewContext:
    sections: list[str] = []
    file_contexts: dict[str, str] = {}

    for idx in range(file_count):
        file_path = f"src/file_{idx}.py"
        header = (
            f"diff --git a/{file_path} b/{file_path}\n"
            f"--- a/{file_path}\n"
            f"+++ b/{file_path}\n"
            "@@ -1,1 +1,3 @@\n"
        )
        body = "".join([f"+line {idx}-{j}\n" for j in range(section_lines)])
        sections.append(header + body)
        file_contexts[file_path] = (
            f"FILE: {file_path}\nSTATUS: M\nCHANGED_LINES: [1, 2, 3]\n"
            "SYMBOLS: []\nSNIPPET:\n    1: x = 1\n"
        )

    return ReviewContext(
        base="main",
        head="HEAD",
        diff_text="".join(sections),
        file_contexts=file_contexts,
    )


def test_estimate_tokens_minimum_one() -> None:
    assert estimate_tokens("") == 1
    assert estimate_tokens("abcd") == 1
    assert estimate_tokens("a" * 40) == 10


def test_consolidated_mode_creates_one_request_per_chunk(tmp_path: Path) -> None:
    prompts_dir = tmp_path / "prompts"
    prompts_dir.mkdir()
    for reviewer in ("bug", "reliability", "security"):
        (prompts_dir / f"{reviewer}_reviewer.md").write_text("Template", encoding="utf-8")

    context = _fake_context(file_count=3, section_lines=200)
    client = CountingLLMClient('{"findings": []}')

    result = run_reviewers(
        reviewers=["bug", "reliability", "security"],
        llm_client=client,
        context=context,
        prompts_dir=prompts_dir,
        max_prompt_tokens=1000,
        review_mode=REVIEW_MODE_CONSOLIDATED,
    )

    assert result.planned_requests == result.reviewable_chunks
    assert result.completed_requests == result.planned_requests


def test_separate_mode_still_runs_all_reviewers(tmp_path: Path) -> None:
    prompts_dir = tmp_path / "prompts"
    prompts_dir.mkdir()
    for reviewer in ("bug", "reliability", "security"):
        (prompts_dir / f"{reviewer}_reviewer.md").write_text("Template", encoding="utf-8")

    context = _fake_context(file_count=2, section_lines=180)
    client = CountingLLMClient('{"findings": []}')

    result = run_reviewers(
        reviewers=["bug", "reliability", "security"],
        llm_client=client,
        context=context,
        prompts_dir=prompts_dir,
        max_prompt_tokens=1000,
        review_mode=REVIEW_MODE_SEPARATE,
    )

    assert result.planned_requests == result.reviewable_chunks * 3


def test_documentation_only_chunks_are_skipped(tmp_path: Path) -> None:
    prompts_dir = tmp_path / "prompts"
    prompts_dir.mkdir()
    for reviewer in ("bug", "reliability", "security"):
        (prompts_dir / f"{reviewer}_reviewer.md").write_text("Template", encoding="utf-8")

    diff_text = (
        "diff --git a/README.md b/README.md\n"
        "--- a/README.md\n"
        "+++ b/README.md\n"
        "@@ -1,1 +1,2 @@\n"
        "+docs update\n"
        "diff --git a/.gitignore b/.gitignore\n"
        "--- a/.gitignore\n"
        "+++ b/.gitignore\n"
        "@@ -1,1 +1,2 @@\n"
        "+.cache/\n"
    )
    context = ReviewContext(
        base="main",
        head="HEAD",
        diff_text=diff_text,
        file_contexts={
            "README.md": "FILE: README.md\nCHANGED_LINES: [1,2]\n",
            ".gitignore": "FILE: .gitignore\nCHANGED_LINES: [1,2]\n",
        },
    )

    client = CountingLLMClient('{"findings": []}')
    result = run_reviewers(
        reviewers=["bug", "reliability", "security"],
        llm_client=client,
        context=context,
        prompts_dir=prompts_dir,
        max_prompt_tokens=1000,
        review_mode=REVIEW_MODE_CONSOLIDATED,
    )

    assert result.reviewable_chunks == 0
    assert result.skipped_chunks >= 1
    assert result.planned_requests == 0
    assert any(skip.reason == "no_reviewable_code" for skip in result.reviewer_skips)


def test_production_chunks_run_before_tests(tmp_path: Path) -> None:
    prompts_dir = tmp_path / "prompts"
    prompts_dir.mkdir()
    for reviewer in ("bug", "reliability", "security"):
        (prompts_dir / f"{reviewer}_reviewer.md").write_text("Template", encoding="utf-8")

    diff_text = (
        "diff --git a/tests/test_app.py b/tests/test_app.py\n"
        "--- a/tests/test_app.py\n"
        "+++ b/tests/test_app.py\n"
        "@@ -1,1 +1,2 @@\n"
        "+assert True\n"
        "diff --git a/src/app.py b/src/app.py\n"
        "--- a/src/app.py\n"
        "+++ b/src/app.py\n"
        "@@ -1,1 +1,2 @@\n"
        "+return value\n"
    )
    context = ReviewContext(
        base="main",
        head="HEAD",
        diff_text=diff_text,
        file_contexts={
            "tests/test_app.py": "FILE: tests/test_app.py\nCHANGED_LINES: [1,2]\n",
            "src/app.py": "FILE: src/app.py\nCHANGED_LINES: [1,2]\n",
        },
    )

    client = CountingLLMClient('{"findings": []}')
    events: list[dict[str, object]] = []

    run_reviewers(
        reviewers=["bug", "reliability", "security"],
        llm_client=client,
        context=context,
        prompts_dir=prompts_dir,
        max_prompt_tokens=1000,
        review_mode=REVIEW_MODE_CONSOLIDATED,
        debug_sink=lambda event: events.append(event),
    )

    start_events = [event for event in events if event.get("event") == "reviewer_start"]
    assert start_events
    assert start_events[0]["file_class"] == "source"


def test_total_review_budget_stops_new_requests(tmp_path: Path) -> None:
    prompts_dir = tmp_path / "prompts"
    prompts_dir.mkdir()
    for reviewer in ("bug", "reliability", "security"):
        (prompts_dir / f"{reviewer}_reviewer.md").write_text("Template", encoding="utf-8")

    context = _fake_context(file_count=4, section_lines=120)
    client = SleepyClient(sleep_seconds=0.25)

    result = run_reviewers(
        reviewers=["bug", "reliability", "security"],
        llm_client=client,
        context=context,
        prompts_dir=prompts_dir,
        max_prompt_tokens=1000,
        review_mode=REVIEW_MODE_CONSOLIDATED,
        max_review_seconds=0.35,
    )

    assert result.completed_requests >= 1
    assert result.skipped_requests >= 1
    assert any(skip.reason == "total_time_budget_exceeded" for skip in result.reviewer_skips)


def test_hard_timeout_in_run_reviewers_isolated_worker_cleanup(tmp_path: Path) -> None:
    prompts_dir = tmp_path / "prompts"
    prompts_dir.mkdir()
    for reviewer in ("bug", "reliability", "security"):
        (prompts_dir / f"{reviewer}_reviewer.md").write_text("Template", encoding="utf-8")

    context = _fake_context(file_count=2, section_lines=120)
    client = SlowHardTimeoutClient(slow_calls=1)
    baseline_children = len(multiprocessing.active_children())

    result = run_reviewers(
        reviewers=["bug", "reliability", "security"],
        llm_client=client,
        context=context,
        prompts_dir=prompts_dir,
        max_prompt_tokens=1000,
        review_mode=REVIEW_MODE_CONSOLIDATED,
        llm_timeout_seconds=0.2,
    )

    assert result.failed_requests >= 1
    assert result.completed_requests >= 1
    time.sleep(0.05)
    assert len(multiprocessing.active_children()) == baseline_children
