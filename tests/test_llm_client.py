import pytest

from app.llm_client import LLMClient, LLMClientError


def test_generate_rejects_empty_prompt() -> None:
    client = LLMClient(provider="ollama", model_name="llama3")

    with pytest.raises(LLMClientError, match="Prompt cannot be empty"):
        client.generate(" ")


def test_generate_rejects_unknown_provider() -> None:
    client = LLMClient(provider="unknown", model_name="model")

    with pytest.raises(LLMClientError, match="Unsupported LLM provider"):
        client.generate("hello")


def test_gemini_requires_api_key() -> None:
    client = LLMClient(provider="gemini", model_name="gemini-1.5-flash")

    with pytest.raises(LLMClientError, match="GEMINI_API_KEY"):
        client.generate("hello")


def test_openai_requires_api_key() -> None:
    client = LLMClient(provider="openai", model_name="gpt-4o-mini")

    with pytest.raises(LLMClientError, match="OPENAI_API_KEY"):
        client.generate("hello")
