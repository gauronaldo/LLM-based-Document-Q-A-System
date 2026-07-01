"""Embedding model wrapper for multilingual document retrieval."""

from __future__ import annotations

from typing import Any


class EmbeddingModelError(Exception):
    """Raised when embeddings cannot be generated."""


class EmbeddingModel:
    """Thin wrapper around sentence-transformers embedding models."""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self._model: Any | None = None

    def encode(self, texts: list[str]) -> list[list[float]]:
        """Encode text strings into embedding vectors."""

        if not texts:
            return []

        model = self._load_model()

        try:
            embeddings = model.encode(
                texts,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
        except TypeError:
            embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        except Exception as exc:
            raise EmbeddingModelError(f"Failed to generate embeddings: {exc}") from exc

        return [embedding.tolist() for embedding in embeddings]

    def _load_model(self) -> Any:
        """Load the sentence-transformers model on first use."""

        if self._model is not None:
            return self._model

        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise EmbeddingModelError(
                "sentence-transformers is required for embeddings. "
                "Install project dependencies with 'pip install -r requirements.txt'."
            ) from exc

        try:
            self._model = SentenceTransformer(self.model_name)
        except Exception as exc:
            raise EmbeddingModelError(
                f"Failed to load embedding model '{self.model_name}': {exc}"
            ) from exc

        return self._model
