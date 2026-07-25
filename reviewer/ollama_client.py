from __future__ import annotations

from ollama import Client, ResponseError


class OllamaLLMClient:
    """Ollama LLM client for code reviews using local models."""

    def __init__(
        self,
        model: str,
        host: str = "http://localhost:11434",
        timeout: float = 300.0,
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
        self.client = Client(host=host)

    def generate(self, system_prompt: str, user_prompt: str) -> str:
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
        try:
            response = self.client.chat(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                stream=False,
            )
            return response.message.content
        except ConnectionError as e:
            raise ConnectionError(
                f"Could not connect to Ollama at {self.host}. Start Ollama and try again."
            ) from e
        except ResponseError as e:
            if "not found" in str(e).lower() or "does not exist" in str(e).lower():
                raise ValueError(
                    f"Model '{self.model}' not found in Ollama. Run: ollama pull {self.model}"
                ) from e
            raise ValueError(f"Ollama error: {e}") from e
        except Exception as e:
            raise ValueError(f"Unexpected error from Ollama: {e}") from e
