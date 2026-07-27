from __future__ import annotations

import json
import multiprocessing
import time
from unittest.mock import MagicMock, patch

import pytest

from reviewer.hard_timeout import run_with_hard_timeout
from reviewer.llm_client import FakeLLMClient
from reviewer.llm_factory import create_llm_client
from reviewer.ollama_client import OllamaLLMClient, OllamaProviderError


def _slow_worker(delay_seconds: float) -> str:
    time.sleep(delay_seconds)
    return "done"


def _fast_worker(value: str) -> str:
    return value


class TestLLMFactory:
    def test_create_fake_client(self) -> None:
        client = create_llm_client(provider="fake")
        assert isinstance(client, FakeLLMClient)

    def test_create_ollama_client(self) -> None:
        client = create_llm_client(
            provider="ollama",
            model="qwen2.5-coder:7b",
        )
        assert isinstance(client, OllamaLLMClient)
        assert client.model == "qwen2.5-coder:7b"

    def test_create_ollama_with_timeout(self) -> None:
        client = create_llm_client(
            provider="ollama",
            model="qwen2.5-coder:7b",
            llm_timeout=123.5,
        )
        assert isinstance(client, OllamaLLMClient)
        assert client.timeout == 123.5

    def test_create_ollama_with_output_token_cap(self) -> None:
        client = create_llm_client(
            provider="ollama",
            model="qwen2.5-coder:7b",
            llm_max_output_tokens=321,
        )
        assert isinstance(client, OllamaLLMClient)
        assert client.max_output_tokens == 321


class TestFakeLLMClient:
    @pytest.mark.parametrize("reviewer", ["bug", "reliability", "security", "consolidated"])
    def test_generate_uses_explicit_reviewer_line(self, reviewer: str) -> None:
        client = FakeLLMClient()
        response = client.generate(
            system_prompt="You are reliability reviewer.",
            user_prompt=f"REVIEWER: {reviewer}\nDIFF:\n...",
        )
        payload = json.loads(response)

        assert payload["findings"][0]["reviewer"] == reviewer
        assert payload["findings"][0]["category"] == reviewer

    def test_generate_does_not_infer_reviewer_from_system_prompt(self) -> None:
        client = FakeLLMClient()
        response = client.generate(
            system_prompt="You are a security reviewer.",
            user_prompt="REVIEWER: bug\nDIFF:\n...",
        )
        payload = json.loads(response)
        assert payload["findings"][0]["reviewer"] == "bug"

    def test_generate_falls_back_to_bug_when_reviewer_line_missing(self) -> None:
        client = FakeLLMClient()
        response = client.generate(
            system_prompt="You are a security reviewer.",
            user_prompt="DIFF:\n...",
        )
        payload = json.loads(response)
        assert payload["findings"][0]["reviewer"] == "bug"

    def test_generate_ignores_unsupported_reviewer_value(self) -> None:
        client = FakeLLMClient()
        response = client.generate(
            system_prompt="You are a security reviewer.",
            user_prompt="REVIEWER: nonsense\nREVIEWER: security\nDIFF:\n...",
        )
        payload = json.loads(response)
        assert payload["findings"][0]["reviewer"] == "security"


class TestHardTimeout:
    def test_hard_timeout_returns_quickly_for_blocking_call(self) -> None:
        start = time.perf_counter()
        with pytest.raises(TimeoutError):
            run_with_hard_timeout(
                _slow_worker,
                10.0,
                timeout_seconds=0.2,
                timeout_error_message="simulated timeout",
            )
        elapsed = time.perf_counter() - start
        assert elapsed < 1.0

    def test_timed_out_worker_is_cleaned_up(self) -> None:
        baseline_children = len(multiprocessing.active_children())
        with pytest.raises(TimeoutError):
            run_with_hard_timeout(
                _slow_worker,
                10.0,
                timeout_seconds=0.2,
                timeout_error_message="simulated timeout",
            )

        time.sleep(0.05)
        assert len(multiprocessing.active_children()) == baseline_children

    def test_later_requests_still_run_after_timeout(self) -> None:
        with pytest.raises(TimeoutError):
            run_with_hard_timeout(
                _slow_worker,
                10.0,
                timeout_seconds=0.2,
                timeout_error_message="simulated timeout",
            )

        value = run_with_hard_timeout(
            _fast_worker,
            "ok",
            timeout_seconds=1.0,
            timeout_error_message="should not timeout",
        )
        assert value == "ok"


class TestOllamaLLMClient:
    @staticmethod
    def _mock_client_with_timeout(timeout: float = 300.0) -> MagicMock:
        client = MagicMock()
        client._client.timeout = timeout
        return client

    def test_init_fails_without_timeout_support(self) -> None:
        mock_client = MagicMock()
        mock_client._client.timeout = None

        with patch("reviewer.ollama_client.Client", return_value=mock_client):
            with pytest.raises(OllamaProviderError, match="timeout support"):
                OllamaLLMClient(model="test-model")

    def test_generate_uses_hard_timeout_runner(self) -> None:
        mock_client = self._mock_client_with_timeout()

        with patch("reviewer.ollama_client.Client", return_value=mock_client):
            client = OllamaLLMClient(model="test-model")

        with patch("reviewer.ollama_client.run_with_hard_timeout", return_value="ok") as patched:
            result = client.generate(
                system_prompt="You are a reviewer.",
                user_prompt="Review code.",
                response_schema={"type": "object"},
            )

        assert result == "ok"
        assert patched.called

    def test_generate_translates_timeout_error(self) -> None:
        mock_client = self._mock_client_with_timeout()

        with patch("reviewer.ollama_client.Client", return_value=mock_client):
            client = OllamaLLMClient(model="test-model")

        with patch(
            "reviewer.ollama_client.run_with_hard_timeout",
            side_effect=TimeoutError("elapsed=0.3s"),
        ):
            with pytest.raises(TimeoutError, match="Timed out while waiting for Ollama response"):
                client.generate(
                    system_prompt="You are a reviewer.",
                    user_prompt="Review code.",
                )
