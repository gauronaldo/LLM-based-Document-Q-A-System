"""Reusable heading detection helpers for document structure recovery."""

from __future__ import annotations

import re


EXPLICIT_HEADING_PATTERN = re.compile(
    r"^(?:"
    r"(?:chapter|section|part|article|heading)\s+\d+"
    r"|(?:ch\u01b0\u01a1ng|m\u1ee5c|ph\u1ea7n|\u0111i\u1ec1u)\s+\d+"
    r")",
    re.IGNORECASE,
)

COMMON_SECTION_HEADINGS = {
    "abstract",
    "acknowledgements",
    "acknowledgments",
    "appendix",
    "background",
    "conclusion",
    "conclusions",
    "concluding remarks",
    "data",
    "data source",
    "data sources",
    "discussion",
    "empirical strategy",
    "introduction",
    "literature review",
    "methodology",
    "methods",
    "references",
    "results",
}

TITLE_CONNECTORS = {"and", "or", "of", "the", "for", "to", "in", "on", "with", "a", "an"}


def line_looks_like_heading(line: str) -> bool:
    """Return whether a line is likely a structural heading, not a sentence."""

    line = line.strip()
    if not line or len(line) > 120:
        return False

    if _numbered_heading_looks_like_title(line):
        return True

    normalized = strip_numbering(line).lower()
    if normalized in COMMON_SECTION_HEADINGS:
        return True
    if line.isupper() and len(line.split()) <= 12:
        return True
    return bool(EXPLICIT_HEADING_PATTERN.match(line))


def strip_numbering(line: str) -> str:
    """Remove a leading section number from a possible heading."""

    return re.sub(r"^\d+(?:\.\d+)*[.)]?\s*", "", line).strip()


def _numbered_heading_looks_like_title(line: str) -> bool:
    match = re.match(r"^\d+(?:\.\d+)*[.)]?\s+(.+)$", line)
    if not match:
        return False
    return _title_fragment_looks_like_heading(match.group(1))


def _title_fragment_looks_like_heading(title: str) -> bool:
    title = title.strip()
    normalized = title.lower()
    if normalized in COMMON_SECTION_HEADINGS:
        return True
    if len(title) > 80 or re.search(r"[,;:!?]", title):
        return False

    words = re.findall(r"[\w\u00c0-\u1ef9'-]+", title)
    if not 1 <= len(words) <= 8:
        return False

    title_like_words = 0
    for word in words:
        if word.lower() in TITLE_CONNECTORS:
            title_like_words += 1
        elif word[:1].isupper() or word.isupper():
            title_like_words += 1

    return title_like_words / len(words) >= 0.65
