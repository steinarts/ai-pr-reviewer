from unittest.mock import MagicMock, patch

import pytest

from reviewer.llm_client import FakeLLMClient
from reviewer.llm_factory import create_llm_client
from reviewer.ollama_client import OllamaLLMClient


class TestLLMFactory:
    """Tests for create_llm_client factory function."""

    def test_create_fake_client(self):
        """Verify factory creates FakeLLMClient."""
        client = create_llm_client(provider="fake")
        assert isinstance(client, FakeLLMClient)

    def test_create_ollama_client(self):
        """Verify factory creates OllamaLLMClient with model."""
        client = create_llm_client(
            provider="ollama",
            model="qwen2.5-coder:7b",
        )
        assert isinstance(client, OllamaLLMClient)
        assert client.model == "qwen2.5-coder:7b"

    def test_create_ollama_with_custom_host(self):
        """Verify factory accepts custom ollama host."""
        client = create_llm_client(
            provider="ollama",
            model="qwen2.5-coder:7b",
            ollama_host="http://192.168.1.100:11434",
        )
        assert client.host == "http://192.168.1.100:11434"

    def test_create_ollama_requires_model(self):
        """Verify factory raises error if model missing for ollama."""
        with pytest.raises(ValueError, match="--model is required"):
            create_llm_client(provider="ollama", model="")

    def test_unknown_provider(self):
        """Verify factory rejects unknown provider."""
        with pytest.raises(ValueError, match="Unknown provider"):
            create_llm_client(provider="unknown")


class TestFakeLLMClientGenerate:
    """Tests for FakeLLMClient.generate method."""

    def test_generate_returns_json(self):
        """Verify FakeLLMClient returns valid JSON."""
        client = FakeLLMClient()
        response = client.generate(
            system_prompt="You are a bug reviewer.",
            user_prompt="Review this code.",
        )
        assert isinstance(response, str)
        assert "findings" in response

    def test_generate_respects_system_prompt_bug(self):
        """Verify reviewer type is inferred from system prompt."""
        client = FakeLLMClient()
        response = client.generate(
            system_prompt="You are a bug reviewer.",
            user_prompt="Review this code.",
        )
        assert '"reviewer": "bug"' in response

    def test_generate_respects_system_prompt_reliability(self):
        """Verify reviewer type for reliability."""
        client = FakeLLMClient()
        response = client.generate(
            system_prompt="You are a reliability reviewer.",
            user_prompt="Review this code.",
        )
        assert '"reviewer": "reliability"' in response

    def test_generate_respects_system_prompt_security(self):
        """Verify reviewer type for security."""
        client = FakeLLMClient()
        response = client.generate(
            system_prompt="You are a security reviewer.",
            user_prompt="Review this code.",
        )
        assert '"reviewer": "security"' in response


class TestOllamaLLMClient:
    """Tests for OllamaLLMClient."""

    def test_init_defaults(self):
        """Verify OllamaLLMClient initializes with defaults."""
        with patch("reviewer.ollama_client.Client"):
            client = OllamaLLMClient(model="test-model")
            assert client.model == "test-model"
            assert client.host == "http://localhost:11434"
            assert client.timeout == 300.0

    def test_init_custom_host(self):
        """Verify OllamaLLMClient accepts custom host."""
        with patch("reviewer.ollama_client.Client"):
            client = OllamaLLMClient(
                model="test-model",
                host="http://192.168.1.100:11434",
            )
            assert client.host == "http://192.168.1.100:11434"

    def test_generate_calls_chat(self):
        """Verify generate calls client.chat with correct args."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.message.content = "Test response"
        mock_client.chat.return_value = mock_response

        with patch("reviewer.ollama_client.Client", return_value=mock_client):
            client = OllamaLLMClient(model="test-model")
            result = client.generate(
                system_prompt="You are a reviewer.",
                user_prompt="Review code.",
            )

        assert result == "Test response"
        mock_client.chat.assert_called_once()
        call_args = mock_client.chat.call_args
        assert call_args[1]["model"] == "test-model"
        assert call_args[1]["stream"] is False
        messages = call_args[1]["messages"]
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"

    def test_generate_connection_error(self):
        """Verify generate handles connection errors."""
        mock_client = MagicMock()
        mock_client.chat.side_effect = ConnectionError("Connection refused")

        with patch("reviewer.ollama_client.Client", return_value=mock_client):
            client = OllamaLLMClient(model="test-model")
            with pytest.raises(ConnectionError, match="Could not connect to Ollama"):
                client.generate(
                    system_prompt="You are a reviewer.",
                    user_prompt="Review code.",
                )

    def test_generate_model_not_found(self):
        """Verify generate handles model not found error."""
        from ollama import ResponseError

        mock_client = MagicMock()
        mock_client.chat.side_effect = ResponseError("model not found")

        with patch("reviewer.ollama_client.Client", return_value=mock_client):
            client = OllamaLLMClient(model="nonexistent-model")
            with pytest.raises(ValueError, match="ollama pull nonexistent-model"):
                client.generate(
                    system_prompt="You are a reviewer.",
                    user_prompt="Review code.",
                )
