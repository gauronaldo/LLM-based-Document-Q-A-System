"""Vietnamese text normalization and cleanup."""

from __future__ import annotations

import re
import unicodedata


class VietnameseTextPreprocessor:
    """Clean document and query text while preserving Vietnamese accents."""

    def clean(self, text: str) -> str:
        """Normalize document text without removing Vietnamese accents."""

        if not text:
            return ""

        normalized = unicodedata.normalize("NFC", text)
        normalized = normalized.replace("\ufeff", "")
        normalized = normalized.replace("\u00a0", " ")
        normalized = normalized.replace("\r\n", "\n").replace("\r", "\n")
        normalized = self._fix_line_breaks(normalized)
        normalized = self._normalize_punctuation_spacing(normalized)
        normalized = self._collapse_spaces(normalized)
        normalized = self._collapse_blank_lines(normalized)
        return normalized.strip()

    def clean_query(self, query: str) -> str:
        """Normalize and clean a user query while preserving Vietnamese accents."""

        return self.clean(query)

    @staticmethod
    def _fix_line_breaks(text: str) -> str:
        """Join wrapped lines while keeping paragraph boundaries."""

        paragraphs = re.split(r"\n\s*\n", text)
        fixed_paragraphs = []

        for paragraph in paragraphs:
            lines = [line.strip() for line in paragraph.splitlines() if line.strip()]
            fixed_paragraphs.append(" ".join(lines))

        return "\n\n".join(paragraph for paragraph in fixed_paragraphs if paragraph)

    @staticmethod
    def _collapse_spaces(text: str) -> str:
        """Collapse repeated horizontal whitespace."""

        return re.sub(r"[ \t\f\v]+", " ", text)

    @staticmethod
    def _collapse_blank_lines(text: str) -> str:
        """Limit blank lines to a single paragraph break."""

        return re.sub(r"\n{3,}", "\n\n", text)

    @staticmethod
    def _normalize_punctuation_spacing(text: str) -> str:
        """Clean common spacing issues around punctuation."""

        text = re.sub(r"\s+([,.!?;:%])", r"\1", text)
        text = re.sub(r"([,.!?;:])(?=\S)", r"\1 ", text)
        text = re.sub(r"(\d)\s+%", r"\1%", text)
        return text
