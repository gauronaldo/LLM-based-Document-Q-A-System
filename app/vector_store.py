"""ChromaDB vector store wrapper."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any


class VectorStoreError(Exception):
    """Raised when vector database operations fail."""


class VectorStore:
    """Store and search document chunks in a persistent ChromaDB collection."""

    def __init__(self, persist_directory: str | Path, collection_name: str):
        self.persist_directory = Path(persist_directory)
        self.collection_name = collection_name
        self._client: Any | None = None
        self._collection: Any | None = None

    def add_chunks(self, chunks: list[dict[str, Any]], embeddings: list[list[float]]) -> None:
        """Add chunks and their embeddings to the ChromaDB collection."""

        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings must have the same length.")
        if not chunks:
            return

        ids = [chunk["chunk_id"] for chunk in chunks]
        documents = [chunk["text"] for chunk in chunks]
        metadatas = [self._clean_metadata(chunk.get("metadata", {})) for chunk in chunks]

        try:
            self.collection.upsert(
                ids=ids,
                documents=documents,
                embeddings=embeddings,
                metadatas=metadatas,
            )
        except Exception as exc:
            raise VectorStoreError(f"Failed to add chunks to vector store: {exc}") from exc

    def search(self, query_embedding: list[float], top_k: int = 5) -> list[dict[str, Any]]:
        """Search for the most relevant chunks for a query embedding."""

        if top_k <= 0:
            raise ValueError("top_k must be greater than 0.")
        if not query_embedding:
            return []

        try:
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                include=["documents", "metadatas", "distances"],
            )
        except Exception as exc:
            raise VectorStoreError(f"Failed to search vector store: {exc}") from exc

        ids = results.get("ids", [[]])[0]
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        retrieved: list[dict[str, Any]] = []
        for chunk_id, text, metadata, distance in zip(ids, documents, metadatas, distances):
            retrieved.append(
                {
                    "chunk_id": chunk_id,
                    "text": text,
                    "metadata": metadata or {},
                    "score": self._distance_to_score(distance),
                    "distance": distance,
                }
            )

        return retrieved

    def reset(self) -> None:
        """Delete and recreate the collection."""

        try:
            client = self.client
            existing = [collection.name for collection in client.list_collections()]
            if self.collection_name in existing:
                client.delete_collection(self.collection_name)
            self._collection = client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},
            )
        except Exception as exc:
            raise VectorStoreError(f"Failed to reset vector store: {exc}") from exc

    @property
    def client(self) -> Any:
        """Return a persistent ChromaDB client."""

        if self._client is None:
            try:
                import chromadb
            except ImportError as exc:
                raise VectorStoreError(
                    "chromadb is required for vector storage. "
                    "Install project dependencies with 'pip install -r requirements.txt'."
                ) from exc

            try:
                self.persist_directory.mkdir(parents=True, exist_ok=True)
                self._client = chromadb.PersistentClient(path=str(self.persist_directory))
            except Exception as exc:
                raise VectorStoreError(f"Failed to initialize ChromaDB: {exc}") from exc

        return self._client

    @property
    def collection(self) -> Any:
        """Return the configured ChromaDB collection."""

        if self._collection is None:
            try:
                self._collection = self.client.get_or_create_collection(
                    name=self.collection_name,
                    metadata={"hnsw:space": "cosine"},
                )
            except Exception as exc:
                raise VectorStoreError(f"Failed to open collection '{self.collection_name}': {exc}") from exc

        return self._collection

    @staticmethod
    def _clean_metadata(metadata: dict[str, Any]) -> dict[str, str | int | float | bool]:
        """Convert metadata to Chroma-compatible scalar values."""

        cleaned: dict[str, str | int | float | bool] = {}
        for key, value in metadata.items():
            if value is None:
                continue
            if isinstance(value, str | int | float | bool):
                cleaned[key] = value
            else:
                cleaned[key] = str(value)
        return cleaned

    @staticmethod
    def _distance_to_score(distance: float | int | None) -> float:
        """Convert Chroma distance to a bounded relevance score."""

        if distance is None:
            return 0.0

        distance_value = float(distance)
        if math.isnan(distance_value):
            return 0.0

        return max(0.0, min(1.0, 1.0 - distance_value))
