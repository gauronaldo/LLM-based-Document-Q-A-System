"""Provider-neutral LLM client."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Iterator


class LLMClientError(Exception):
    """Raised when an LLM request cannot be completed."""


class LLMClient:
    """Simple LLM interface for Gemini, OpenAI, or Ollama."""

    def __init__(
        self,
        provider: str,
        model_name: str,
        gemini_api_key: str | None = None,
        openai_api_key: str | None = None,
        ollama_base_url: str = "http://localhost:11434",
    ):
        self.provider = provider.lower().strip()
        self.model_name = model_name
        self.gemini_api_key = gemini_api_key
        self.openai_api_key = openai_api_key
        self.ollama_base_url = ollama_base_url.rstrip("/")

    def generate(self, prompt: str) -> str:
        """Generate a response from the configured LLM provider."""

        if not prompt.strip():
            raise LLMClientError("Prompt cannot be empty.")

        if self.provider == "gemini":
            return self._generate_gemini(prompt)
        if self.provider == "openai":
            return self._generate_openai(prompt)
        if self.provider == "ollama":
            return self._generate_ollama(prompt)

        raise LLMClientError(
            f"Unsupported LLM provider '{self.provider}'. "
            "Supported providers are: gemini, openai, ollama."
        )

    def generate_stream(self, prompt: str) -> Iterator[str]:
        """Stream response chunks when the provider supports it."""

        if not prompt.strip():
            raise LLMClientError("Prompt cannot be empty.")

        if self.provider == "ollama":
            yield from self._generate_ollama_stream(prompt)
            return
        if self.provider == "openai":
            yield from self._generate_openai_stream(prompt)
            return

        yield self.generate(prompt)

    def _generate_gemini(self, prompt: str) -> str:
        """Generate text with Gemini."""

        if not self.gemini_api_key:
            raise LLMClientError("GEMINI_API_KEY is required when LLM_PROVIDER=gemini.")

        try:
            import google.generativeai as genai
        except ImportError as exc:
            raise LLMClientError(
                "google-generativeai is required for Gemini. "
                "Install project dependencies with 'pip install -r requirements.txt'."
            ) from exc

        try:
            genai.configure(api_key=self.gemini_api_key)
            model = genai.GenerativeModel(self.model_name)
            response = model.generate_content(prompt)
        except Exception as exc:
            raise LLMClientError(
                f"Gemini request failed for model '{self.model_name}': {exc}. "
                "Try setting LLM_MODEL=gemini-2.5-flash or another model listed "
                "in Google AI Studio."
            ) from exc

        text = getattr(response, "text", None)
        if not text:
            raise LLMClientError("Gemini returned an empty response.")
        return text.strip()

    def _generate_openai(self, prompt: str) -> str:
        """Generate text with OpenAI."""

        if not self.openai_api_key:
            raise LLMClientError("OPENAI_API_KEY is required when LLM_PROVIDER=openai.")

        try:
            from openai import OpenAI
        except ImportError as exc:
            raise LLMClientError(
                "openai is required for OpenAI models. "
                "Install project dependencies with 'pip install -r requirements.txt'."
            ) from exc

        try:
            client = OpenAI(api_key=self.openai_api_key)
            response = client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "system",
                        "content": "You answer document questions using only supplied context.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
            )
        except Exception as exc:
            raise LLMClientError(f"OpenAI request failed: {exc}") from exc

        content = response.choices[0].message.content
        if not content:
            raise LLMClientError("OpenAI returned an empty response.")
        return content.strip()

    def _generate_openai_stream(self, prompt: str) -> Iterator[str]:
        """Stream text with OpenAI chat completions."""

        if not self.openai_api_key:
            raise LLMClientError("OPENAI_API_KEY is required when LLM_PROVIDER=openai.")

        try:
            from openai import OpenAI
        except ImportError as exc:
            raise LLMClientError(
                "openai is required for OpenAI models. "
                "Install project dependencies with 'pip install -r requirements.txt'."
            ) from exc

        try:
            client = OpenAI(api_key=self.openai_api_key)
            stream = client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "system",
                        "content": "You answer document questions using only supplied context.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
                stream=True,
            )
            for event in stream:
                delta = event.choices[0].delta.content
                if delta:
                    yield delta
        except Exception as exc:
            raise LLMClientError(f"OpenAI streaming request failed: {exc}") from exc

    def _generate_ollama(self, prompt: str) -> str:
        """Generate text with a local Ollama server."""

        payload = json.dumps(
            {
                "model": self.model_name,
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "You are a document-grounded assistant. Follow the user's "
                            "language instruction exactly. Use only compact numeric "
                            "citations like [1]."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                "stream": False,
                "options": {
                    "temperature": 0.1,
                    "num_predict": 700,
                },
            }
        ).encode("utf-8")
        request = urllib.request.Request(
            f"{self.ollama_base_url}/api/chat",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                data = json.loads(response.read().decode("utf-8"))
        except urllib.error.URLError as exc:
            raise LLMClientError(f"Ollama request failed: {exc}") from exc
        except Exception as exc:
            raise LLMClientError(f"Failed to parse Ollama response: {exc}") from exc

        message = data.get("message", {})
        text = message.get("content", "") if isinstance(message, dict) else ""
        if not text:
            raise LLMClientError("Ollama returned an empty response.")
        return text.strip()

    def _generate_ollama_stream(self, prompt: str) -> Iterator[str]:
        """Stream text with a local Ollama server."""

        payload = json.dumps(
            {
                "model": self.model_name,
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "You are a document-grounded assistant. Follow the user's "
                            "language instruction exactly. Use only compact numeric "
                            "citations like [1]."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                "stream": True,
                "options": {
                    "temperature": 0.1,
                    "num_predict": 700,
                },
            }
        ).encode("utf-8")
        request = urllib.request.Request(
            f"{self.ollama_base_url}/api/chat",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                for line in response:
                    if not line.strip():
                        continue
                    data = json.loads(line.decode("utf-8"))
                    message = data.get("message", {})
                    text = message.get("content", "") if isinstance(message, dict) else ""
                    if text:
                        yield text
                    if data.get("done"):
                        break
        except urllib.error.URLError as exc:
            raise LLMClientError(f"Ollama streaming request failed: {exc}") from exc
        except Exception as exc:
            raise LLMClientError(f"Failed to parse Ollama streaming response: {exc}") from exc
