"""Split cleaned documents into source-preserving chunks."""

from __future__ import annotations

import re
from typing import Any

from app.heading_detector import line_looks_like_heading


SENTENCE_PATTERN = re.compile(r"(?<=[.!?;:])\s+")


class TextSplitter:
    """Split documents with Vietnamese/English-friendly section boundaries."""

    def __init__(
        self,
        chunk_size: int = 900,
        chunk_overlap: int = 150,
        parent_context_max_chars: int = 3000,
    ):
        if chunk_size <= 0:
            raise ValueError("chunk_size must be greater than 0.")
        if chunk_overlap < 0:
            raise ValueError("chunk_overlap must be greater than or equal to 0.")
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size.")
        if parent_context_max_chars <= 0:
            raise ValueError("parent_context_max_chars must be greater than 0.")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.parent_context_max_chars = parent_context_max_chars

    def split(self, documents: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Split document objects into chunks and preserve source metadata."""

        chunks: list[dict[str, Any]] = []

        for document in documents:
            text = document.get("text", "").strip()
            metadata = document.get("metadata", {})
            if not text:
                continue

            parts = self._split_document_sections(text)
            for chunk_index, part in enumerate(parts):
                chunk_metadata = dict(metadata)
                chunk_metadata["chunk_index"] = chunk_index
                chunk_metadata["parent_id"] = part["parent_id"]
                chunk_metadata["section_title"] = part["section_title"]
                chunk_metadata["parent_text"] = self._trim_parent_context(part["parent_text"])
                chunk_id = self._build_chunk_id(chunk_metadata)

                chunks.append(
                    {
                        "chunk_id": chunk_id,
                        "text": part["text"],
                        "metadata": chunk_metadata,
                    }
                )

        return chunks

    def _split_document_sections(self, text: str) -> list[dict[str, str]]:
        """Split a document into sections, then chunks with parent section context."""

        sections = self._split_sections(text)
        parts: list[dict[str, str]] = []

        for section_index, section in enumerate(sections):
            section_title = section["section_title"]
            parent_text = section["text"]
            parent_id = f"section_{section_index}"
            for chunk_text in self._split_text(parent_text):
                parts.append(
                    {
                        "text": chunk_text,
                        "parent_text": parent_text,
                        "parent_id": parent_id,
                        "section_title": section_title,
                    }
                )

        return parts

    def _split_sections(self, text: str) -> list[dict[str, str]]:
        """Split text into heading/section blocks when headings are visible."""

        line_sections = self._split_sections_by_lines(text)
        if line_sections:
            return line_sections

        paragraphs = [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]
        if not paragraphs:
            return []

        sections: list[dict[str, str]] = []
        current_title = "Document"
        current_parts: list[str] = []

        for paragraph in paragraphs:
            if self._looks_like_heading(paragraph):
                if current_parts:
                    sections.append(
                        {
                            "section_title": current_title,
                            "text": "\n\n".join(current_parts),
                        }
                    )
                current_title = self._extract_section_title(paragraph)
                current_parts = [paragraph]
            else:
                current_parts.append(paragraph)

        if current_parts:
            sections.append(
                {
                    "section_title": current_title,
                    "text": "\n\n".join(current_parts),
                }
            )

        return sections

    def _split_sections_by_lines(self, text: str) -> list[dict[str, str]]:
        """Split sections when PDF extraction puts headings on their own lines."""

        lines = [line.rstrip() for line in text.splitlines()]
        sections: list[dict[str, str]] = []
        current_title = "Document"
        current_lines: list[str] = []
        found_heading = False
        index = 0

        while index < len(lines):
            line = lines[index].strip()
            if not line:
                current_lines.append("")
                index += 1
                continue

            candidate = line
            next_line = lines[index + 1].strip() if index + 1 < len(lines) else ""
            consumed_next = False

            if re.fullmatch(r"\d+(?:\.\d+)*[.)]?", line) and next_line:
                candidate = f"{line} {next_line}"
                consumed_next = True

            if line_looks_like_heading(candidate):
                found_heading = True
                if current_lines:
                    sections.append(
                        {
                            "section_title": current_title,
                            "text": "\n".join(current_lines).strip(),
                        }
                    )
                current_title = candidate
                current_lines = [candidate]
                index += 2 if consumed_next else 1
                continue

            current_lines.append(line)
            index += 1

        if current_lines:
            sections.append(
                {
                    "section_title": current_title,
                    "text": "\n".join(current_lines).strip(),
                }
            )

        return [section for section in sections if section["text"]] if found_heading else []

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
        """Start a new chunk with a word-aware tail from the previous chunk."""

        if self.chunk_overlap == 0:
            return next_text

        overlap = self._word_aware_tail(previous)
        if not overlap:
            return next_text
        return f"{overlap}\n\n{next_text}"

    def _word_aware_tail(self, text: str) -> str:
        """Return an overlap tail that does not begin in the middle of a word."""

        tail = text[-self.chunk_overlap :].strip()
        if not tail:
            return ""

        original_start = len(text) - self.chunk_overlap
        starts_at_boundary = original_start <= 0 or text[original_start - 1].isspace()
        if starts_at_boundary:
            return tail

        first_space = tail.find(" ")
        if first_space == -1:
            return ""
        return tail[first_space + 1 :].strip()

    def _trim_parent_context(self, text: str) -> str:
        """Keep parent context useful without storing very large metadata strings."""

        if len(text) <= self.parent_context_max_chars:
            return text
        return text[: self.parent_context_max_chars].rsplit(" ", 1)[0].strip()

    @staticmethod
    def _looks_like_heading(paragraph: str) -> bool:
        """Return whether a paragraph looks like a section heading."""

        lines = [line.strip() for line in paragraph.strip().splitlines() if line.strip()]
        first_line = lines[0]
        heading_candidates = [first_line]
        if len(lines) > 1 and re.fullmatch(r"\d+(?:\.\d+)*[.)]?", first_line):
            heading_candidates.append(f"{first_line} {lines[1]}")

        for candidate in heading_candidates:
            if line_looks_like_heading(candidate):
                return True
        return False

    @staticmethod
    def _extract_section_title(paragraph: str) -> str:
        lines = [line.strip() for line in paragraph.strip().splitlines() if line.strip()]
        if len(lines) > 1 and re.fullmatch(r"\d+(?:\.\d+)*[.)]?", lines[0]):
            return f"{lines[0]} {lines[1]}"
        return lines[0]

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
