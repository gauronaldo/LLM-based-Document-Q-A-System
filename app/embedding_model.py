"""Embedding model wrapper for multilingual document retrieval."""

from __future__ import annotations

import hashlib
import math
import re
import sys
import time
from typing import Any, Callable

from app.processing_debug import log_event, text_stats


class EmbeddingModelError(Exception):
    """Raised when embeddings cannot be generated."""


class EmbeddingModel:
    """Thin wrapper around sentence-transformers embedding models."""

    LOCAL_HASH_MODEL_NAMES = {"local-hash", "local_hash", "hash", "hashing"}
    LOCAL_HASH_DIMENSION = 384

    def __init__(self, model_name: str, batch_size: int = 32):
        if batch_size <= 0:
            raise ValueError("batch_size must be greater than 0.")

        self.model_name = model_name
        self.batch_size = batch_size
        self._model: Any | None = None

    def encode(
        self,
        texts: list[str],
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> list[list[float]]:
        """Encode text strings into embedding vectors."""

        if not texts:
            return []

        log_event(
            "embedding_encode_call_start",
            model_name=self.model_name,
            batch_size=self.batch_size,
            **text_stats(texts),
        )

        if self._is_local_hash_model():
            embeddings: list[list[float]] = []
            for batch in _batches(texts, self.batch_size):
                log_event("local_hash_batch_start", **text_stats(batch))
                embeddings.extend(
                    _local_hash_embedding(text, self.LOCAL_HASH_DIMENSION) for text in batch
                )
                log_event("local_hash_batch_done", completed=len(embeddings), total=len(texts))
                _notify_embedding_progress(progress_callback, len(embeddings), len(texts))
            log_event("embedding_encode_call_done", embeddings=len(embeddings))
            return embeddings

        model = self._load_model()
        embeddings: list[list[float]] = []

        for batch in _batches(texts, self.batch_size):
            batch_started = time.perf_counter()
            log_event("sentence_transformer_batch_start", **text_stats(batch))
            try:
                batch_embeddings = model.encode(
                    batch,
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                )
            except TypeError:
                batch_embeddings = model.encode(
                    batch,
                    convert_to_numpy=True,
                    show_progress_bar=False,
                )
            except Exception as exc:
                log_event("sentence_transformer_batch_error", error=repr(exc))
                raise EmbeddingModelError(f"Failed to generate embeddings: {exc}") from exc

            embeddings.extend(_as_python_list(embedding) for embedding in batch_embeddings)
            log_event(
                "sentence_transformer_batch_done",
                elapsed_seconds=round(time.perf_counter() - batch_started, 3),
                completed=len(embeddings),
                total=len(texts),
            )
            _notify_embedding_progress(progress_callback, len(embeddings), len(texts))

        log_event("embedding_encode_call_done", embeddings=len(embeddings))
        return embeddings

    def _is_local_hash_model(self) -> bool:
        return self.model_name.strip().lower() in self.LOCAL_HASH_MODEL_NAMES

    def _load_model(self) -> Any:
        """Load the sentence-transformers model on first use."""

        if self._model is not None:
            return self._model

        try:
            log_event(
                "embedding_model_import_start",
                model_name=self.model_name,
                already_loaded="sentence_transformers" in sys.modules,
            )
            from sentence_transformers import SentenceTransformer
            log_event(
                "embedding_model_import_done",
                model_name=self.model_name,
                already_loaded="sentence_transformers" in sys.modules,
            )
        except ImportError as exc:
            log_event("embedding_model_import_error", model_name=self.model_name, error=repr(exc))
            raise EmbeddingModelError(
                "sentence-transformers is required for embeddings. "
                "Install project dependencies with 'pip install -r requirements.txt'."
            ) from exc
        except Exception as exc:
            log_event("embedding_model_import_error", model_name=self.model_name, error=repr(exc))
            raise EmbeddingModelError(
                f"Failed to import sentence-transformers for '{self.model_name}': {exc}"
            ) from exc

        try:
            started = time.perf_counter()
            log_event("embedding_model_load_start", model_name=self.model_name)
            self._model = SentenceTransformer(self.model_name)
            log_event(
                "embedding_model_load_done",
                model_name=self.model_name,
                elapsed_seconds=round(time.perf_counter() - started, 3),
            )
        except Exception as exc:
            log_event("embedding_model_load_error", model_name=self.model_name, error=repr(exc))
            raise EmbeddingModelError(
                f"Failed to load embedding model '{self.model_name}': {exc}"
            ) from exc

        return self._model


def _local_hash_embedding(text: str, dimension: int) -> list[float]:
    """Create a deterministic no-download lexical embedding for offline demos."""

    vector = [0.0] * dimension
    features = _hash_features(text)
    if not features:
        return vector

    for feature, weight in features:
        digest = hashlib.sha256(feature.encode("utf-8")).digest()
        index = int.from_bytes(digest[:4], byteorder="big") % dimension
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        vector[index] += sign * weight

    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0:
        return vector
    return [value / norm for value in vector]


def _batches(items: list[str], batch_size: int) -> list[list[str]]:
    return [items[index : index + batch_size] for index in range(0, len(items), batch_size)]


def _as_python_list(embedding: Any) -> list[float]:
    if hasattr(embedding, "tolist"):
        return embedding.tolist()
    return list(embedding)


def _notify_embedding_progress(
    progress_callback: Callable[[int, int], None] | None,
    completed: int,
    total: int,
) -> None:
    if progress_callback:
        progress_callback(completed, total)


def _hash_features(text: str) -> list[tuple[str, float]]:
    normalized = " ".join(text.lower().split())
    tokens = re.findall(r"\w+", normalized, flags=re.UNICODE)
    features: list[tuple[str, float]] = []

    for token in tokens:
        if len(token) > 1:
            features.append((f"w:{token}", 1.0))
        for ngram in _character_ngrams(token, size=3):
            features.append((f"c:{ngram}", 0.35))

    for left, right in zip(tokens, tokens[1:]):
        features.append((f"b:{left}_{right}", 0.75))

    return features


def _character_ngrams(token: str, size: int) -> list[str]:
    if len(token) <= size:
        return [token]
    return [token[index : index + size] for index in range(len(token) - size + 1)]
