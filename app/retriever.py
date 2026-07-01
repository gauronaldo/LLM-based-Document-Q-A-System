"""Retrieve relevant chunks for user questions."""

from __future__ import annotations

from typing import Any


class RetrieverError(Exception):
    """Raised when retrieval cannot be completed."""


class Retriever:
    """Encode questions, search the vector store, and filter by relevance."""

    def __init__(
        self,
        vector_store: Any,
        embedder: Any,
        top_k: int = 5,
        similarity_threshold: float = 0.3,
    ):
        if top_k <= 0:
            raise ValueError("top_k must be greater than 0.")
        if not 0 <= similarity_threshold <= 1:
            raise ValueError("similarity_threshold must be between 0 and 1.")

        self.vector_store = vector_store
        self.embedder = embedder
        self.top_k = top_k
        self.similarity_threshold = similarity_threshold

    def retrieve(
        self,
        question: str,
        top_k: int | None = None,
        similarity_threshold: float | None = None,
    ) -> list[dict[str, Any]]:
        """Return chunks relevant enough to answer the question."""

        clean_question = question.strip()
        if not clean_question:
            return []

        try:
            query_embeddings = self.embedder.encode([clean_question])
        except Exception as exc:
            raise RetrieverError(f"Failed to encode question: {exc}") from exc

        if not query_embeddings:
            return []

        search_top_k = self.top_k if top_k is None else top_k
        if search_top_k <= 0:
            raise ValueError("top_k must be greater than 0.")

        try:
            results = self.vector_store.search(query_embeddings[0], top_k=search_top_k)
        except Exception as exc:
            raise RetrieverError(f"Failed to retrieve chunks: {exc}") from exc

        threshold = self.similarity_threshold if similarity_threshold is None else similarity_threshold
        filtered = [
            result
            for result in results
            if float(result.get("score", 0.0)) >= threshold
        ]

        return sorted(filtered, key=lambda result: float(result.get("score", 0.0)), reverse=True)
