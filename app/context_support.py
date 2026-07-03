"""Lightweight context support estimation for generation control."""

from __future__ import annotations

from typing import Any, Literal

from app.keyword_search import tokenize


SupportLevel = Literal["STRONG_SUPPORT", "PARTIAL_SUPPORT", "NO_SUPPORT"]

STRONG_SUPPORT: SupportLevel = "STRONG_SUPPORT"
PARTIAL_SUPPORT: SupportLevel = "PARTIAL_SUPPORT"
NO_SUPPORT: SupportLevel = "NO_SUPPORT"


def estimate_context_support(question: str, retrieved_chunks: list[dict[str, Any]]) -> SupportLevel:
    """Estimate whether retrieved chunks contain enough evidence to answer cautiously."""

    if not retrieved_chunks:
        return NO_SUPPORT

    query_terms = _content_terms(question)
    if not query_terms:
        return PARTIAL_SUPPORT if _top_confidence(retrieved_chunks) >= 0.7 else NO_SUPPORT

    context = " ".join(
        str(part)
        for chunk in retrieved_chunks
        for part in (
            chunk.get("metadata", {}).get("section_title", ""),
            chunk.get("matched_text", ""),
            chunk.get("text", ""),
        )
        if part
    )
    context_terms = set(tokenize(context))
    term_coverage = _weighted_term_coverage(query_terms, context_terms)
    phrase_coverage = _phrase_coverage(query_terms, context)
    top_score = _top_confidence(retrieved_chunks)
    profile_boost = max(float(chunk.get("profile_evidence_boost", 0.0) or 0.0) for chunk in retrieved_chunks)

    if term_coverage >= 0.45:
        return STRONG_SUPPORT
    if phrase_coverage >= 0.25:
        return STRONG_SUPPORT
    if top_score >= 0.75 and term_coverage >= 0.2:
        return STRONG_SUPPORT
    if profile_boost > 0 and term_coverage >= 0.25:
        return STRONG_SUPPORT

    if term_coverage >= 0.22:
        return PARTIAL_SUPPORT
    if phrase_coverage > 0:
        return PARTIAL_SUPPORT
    if top_score >= 0.85:
        return PARTIAL_SUPPORT
    if top_score >= 0.55 and term_coverage >= 0.12:
        return PARTIAL_SUPPORT
    if profile_boost > 0:
        return PARTIAL_SUPPORT
    return NO_SUPPORT


def has_context_support(question: str, retrieved_chunks: list[dict[str, Any]]) -> bool:
    return estimate_context_support(question, retrieved_chunks) != NO_SUPPORT


def _top_confidence(chunks: list[dict[str, Any]]) -> float:
    scores = []
    for chunk in chunks:
        if "base_score" in chunk:
            scores.append(float(chunk.get("base_score", 0.0) or 0.0))
        else:
            scores.append(float(chunk.get("score", 0.0) or 0.0))
    return max(scores, default=0.0)


def _weighted_term_coverage(query_terms: list[str], context_terms: set[str]) -> float:
    if not query_terms or not context_terms:
        return 0.0

    total = 0.0
    matched = 0.0
    for term in query_terms:
        weight = _term_weight(term)
        total += weight
        if term in context_terms:
            matched += weight
    return matched / total if total else 0.0


def _phrase_coverage(query_terms: list[str], context: str) -> float:
    phrases = _query_phrases(query_terms)
    if not phrases:
        return 0.0

    normalized_context = " ".join(tokenize(context))
    hits = sum(1 for phrase in phrases if phrase in normalized_context)
    return hits / len(phrases)


def _query_phrases(query_terms: list[str]) -> list[str]:
    phrases = []
    for index in range(len(query_terms) - 1):
        left, right = query_terms[index], query_terms[index + 1]
        if len(left) >= 5 or len(right) >= 5:
            phrases.append(f"{left} {right}")
    return phrases


def _content_terms(text: str) -> list[str]:
    return [
        token
        for token in tokenize(text)
        if len(token) > 2 and token not in _STOPWORDS
    ]


def _term_weight(term: str) -> float:
    if any(character.isdigit() for character in term):
        return 2.0
    if len(term) >= 10:
        return 1.8
    if len(term) >= 7:
        return 1.4
    return 1.0


_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "by",
    "can",
    "could",
    "did",
    "do",
    "does",
    "document",
    "for",
    "from",
    "how",
    "in",
    "information",
    "is",
    "main",
    "of",
    "on",
    "or",
    "paper",
    "provided",
    "question",
    "report",
    "study",
    "the",
    "this",
    "to",
    "what",
    "when",
    "where",
    "whether",
    "why",
    "with",
}
