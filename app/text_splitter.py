"""Split cleaned documents into source-preserving chunks."""

from __future__ import annotations

import re
from typing import Any


SENTENCE_PATTERN = re.compile(r"(?<=[.!?;:])\s+")


class TextSplitter:
    """Split documents with Vietnamese-friendly paragraph and sentence boundaries."""

    def __init__(self, chunk_size: int = 900, chunk_overlap: int = 150):
        if chunk_size <= 0:
            raise ValueError("chunk_size must be greater than 0.")
        if chunk_overlap < 0:
            raise ValueError("chunk_overlap must be greater than or equal to 0.")
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size.")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split(self, documents: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Split document objects into chunks and preserve source metadata."""

        chunks: list[dict[str, Any]] = []

        for document in documents:
            text = document.get("text", "").strip()
            metadata = document.get("metadata", {})
            if not text:
                continue

            parts = self._split_text(text)
            for chunk_index, chunk_text in enumerate(parts):
                chunk_metadata = dict(metadata)
                chunk_metadata["chunk_index"] = chunk_index
                chunk_id = self._build_chunk_id(chunk_metadata)

                chunks.append(
                    {
                        "chunk_id": chunk_id,
                        "text": chunk_text,
                        "metadata": chunk_metadata,
                    }
                )

        return chunks

    def _split_text(self, text: str) -> list[str]:
        """Split text by paragraph, sentence, then character windows if needed."""

        paragraphs = [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]
        chunks: list[str] = []
        current = ""

        for paragraph in paragraphs:
            units = self._split_long_paragraph(paragraph)
            for unit in units:
                if not current:
                    current = unit
                    continue

                candidate = f"{current}\n\n{unit}"
                if len(candidate) <= self.chunk_size:
                    current = candidate
                else:
                    chunks.append(current)
                    current = self._with_overlap(current, unit)

        if current:
            chunks.append(current)

        return chunks

    def _split_long_paragraph(self, paragraph: str) -> list[str]:
        """Split long paragraphs on sentence boundaries before falling back."""

        if len(paragraph) <= self.chunk_size:
            return [paragraph]

        sentences = [sentence.strip() for sentence in SENTENCE_PATTERN.split(paragraph) if sentence.strip()]
        units: list[str] = []
        current = ""

        for sentence in sentences:
            if len(sentence) > self.chunk_size:
                if current:
                    units.append(current)
                    current = ""
                units.extend(self._split_by_window(sentence))
                continue

            candidate = f"{current} {sentence}".strip()
            if len(candidate) <= self.chunk_size:
                current = candidate
            else:
                if current:
                    units.append(current)
                current = sentence

        if current:
            units.append(current)

        return units

    def _split_by_window(self, text: str) -> list[str]:
        """Fallback chunking for text with no usable sentence boundaries."""

        chunks: list[str] = []
        start = 0
        step = self.chunk_size - self.chunk_overlap

        while start < len(text):
            end = start + self.chunk_size
            chunks.append(text[start:end].strip())
            if end >= len(text):
                break
            start += step

        return [chunk for chunk in chunks if chunk]

    def _with_overlap(self, previous: str, next_text: str) -> str:
        """Start a new chunk with a small tail from the previous chunk."""

        if self.chunk_overlap == 0:
            return next_text

        overlap = previous[-self.chunk_overlap :].strip()
        if not overlap:
            return next_text
        return f"{overlap}\n\n{next_text}"

    @staticmethod
    def _build_chunk_id(metadata: dict[str, Any]) -> str:
        """Build a stable chunk id from source metadata."""

        file_id = str(metadata.get("file_id", "unknown"))
        page = metadata.get("page")
        chunk_index = metadata.get("chunk_index", 0)

        if page is None:
            paragraph_index = metadata.get("paragraph_index")
            if paragraph_index is not None:
                return f"{file_id}_paragraph_{paragraph_index}_chunk_{chunk_index}"
            return f"{file_id}_document_chunk_{chunk_index}"

        return f"{file_id}_page_{page}_chunk_{chunk_index}"
