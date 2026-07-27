from __future__ import annotations

from httpx import ReadTimeout, TimeoutException
from ollama import Client, ResponseError

from .hard_timeout import run_with_hard_timeout


class OllamaProviderError(ValueError):
    """Base class for expected Ollama provider errors."""


def _ollama_chat_worker(
    *,
    host: str,
    model: str,
    timeout: float,
    max_output_tokens: int,
    system_prompt: str,
    user_prompt: str,
    response_schema: dict[str, object] | None,
) -> str:
    client = Client(host=host, timeout=timeout)
    request: dict[str, object] = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "stream": False,
        "options": {"temperature": 0, "num_predict": max_output_tokens},
    }
    if response_schema is not None:
        request["format"] = response_schema

    response = client.chat(**request)
    content = getattr(getattr(response, "message", None), "content", None)
    if content is None and isinstance(response, dict):
        message = response.get("message")
        if isinstance(message, dict):
            content = message.get("content")

    if not isinstance(content, str) or not content.strip():
        raise OllamaProviderError("Ollama returned an empty assistant response.")
    return content


class OllamaLLMClient:
    """Ollama LLM client for code reviews using local models."""

    def __init__(
        self,
        model: str,
        host: str = "http://localhost:11434",
        timeout: float = 300.0,
        max_output_tokens: int = 700,
        worker_func=_ollama_chat_worker,
    ) -> None:
        """Initialize Ollama client.

        Args:
            model: Model name (e.g., "qwen2.5-coder:7b")
            host: Ollama server host. Default: http://localhost:11434
            timeout: Request timeout in seconds. Default: 300
        """
        self.model = model
        self.host = host
        self.timeout = timeout
        self.max_output_tokens = max_output_tokens
        self._timeout_margin_seconds = 0.25
        self.client = Client(host=host, timeout=timeout)
        self._worker_func = worker_func
        self._ensure_timeout_support()

    def _ensure_timeout_support(self) -> None:
        """Fail fast if the installed ollama client does not expose timeout configuration."""
        inner_client = getattr(self.client, "_client", None)
        inner_timeout = getattr(inner_client, "timeout", None)
        if inner_timeout is None:
            raise OllamaProviderError(
                "Installed ollama package does not expose timeout support "
                "on its internal HTTP client. "
                "Upgrade ollama package to a version with timeout support."
            )

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        response_schema: dict[str, object] | None = None,
    ) -> str:
        """Generate a response from Ollama.

        Args:
            system_prompt: System/role instructions
            user_prompt: User query or code review context

        Returns:
            Model's response text

        Raises:
            ConnectionError: If Ollama server is not reachable
            ValueError: If model is not available or response is invalid
        """
        return self.generate_hard_timeout(
            system_prompt,
            user_prompt,
            response_schema=response_schema,
            timeout_seconds=self.timeout,
        )

    def generate_hard_timeout(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        response_schema: dict[str, object] | None = None,
        timeout_seconds: float,
    ) -> str:
        try:
            return run_with_hard_timeout(
                self._worker_func,
                timeout_seconds=timeout_seconds + self._timeout_margin_seconds,
                timeout_error_message=(
                    "Timed out while waiting for Ollama response. "
                    f"Model: {self.model}, host: {self.host}. "
                    f"Configured timeout: {timeout_seconds}s."
                ),
                kwargs={
                    "host": self.host,
                    "model": self.model,
                    "timeout": timeout_seconds,
                    "max_output_tokens": self.max_output_tokens,
                    "system_prompt": system_prompt,
                    "user_prompt": user_prompt,
                    "response_schema": response_schema,
                },
            )
        except ConnectionError as e:
            raise ConnectionError(
                f"Could not connect to Ollama at {self.host}. Start Ollama and try again."
            ) from e
        except ResponseError as e:
            if "not found" in str(e).lower() or "does not exist" in str(e).lower():
                raise OllamaProviderError(
                    f"Model '{self.model}' not found in Ollama. Run: ollama pull {self.model}"
                ) from e
            raise OllamaProviderError(f"Ollama error: {e}") from e
        except (ReadTimeout, TimeoutException) as e:
            raise TimeoutError(
                "Timed out while waiting for Ollama response. "
                f"Model: {self.model}, host: {self.host}. "
                f"Configured timeout: {timeout_seconds}s. "
                "Try a smaller diff, a faster model, or run Ollama on a less busy machine."
            ) from e
        except TimeoutError as e:
            raise TimeoutError(
                "Timed out while waiting for Ollama response. "
                f"Model: {self.model}, host: {self.host}. "
                f"Configured timeout: {timeout_seconds}s. "
                "Try a smaller diff, a faster model, or run Ollama on a less busy machine."
            ) from e
        except ValueError as e:
            if "not found" in str(e).lower() or "does not exist" in str(e).lower():
                raise OllamaProviderError(
                    f"Model '{self.model}' not found in Ollama. Run: ollama pull {self.model}"
                ) from e
            raise OllamaProviderError(str(e)) from e
