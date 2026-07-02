"""Optional reranking for retrieved chunks."""

from __future__ import annotations

from difflib import SequenceMatcher
from typing import Any

from app.keyword_search import lexical_similarity, tokenize


CONTENT_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "by",
    "does",
    "do",
    "for",
    "from",
    "how",
    "in",
    "is",
    "it",
    "key",
    "main",
    "of",
    "on",
    "or",
    "overall",
    "paper",
    "study",
    "the",
    "this",
    "to",
    "what",
    "where",
    "why",
    "with",
    "cho",
    "cua",
    "của",
    "dau",
    "den",
    "đâu",
    "đến",
    "la",
    "là",
    "o",
    "ở",
    "toi",
    "tới",
    "trong",
    "tu",
    "từ",
    "ve",
    "về",
}


class RerankerError(Exception):
    """Raised when reranking fails."""


class Reranker:
    """Rerank chunks with an optional cross-encoder and a lexical fallback."""

    def __init__(self, model_name: str | None = None):
        self.model_name = model_name
        self._model: Any | None = None

    def rerank(
        self,
        query: str,
        chunks: list[dict[str, Any]],
        top_k: int,
    ) -> list[dict[str, Any]]:
        """Rerank chunks and return the top results."""

        if top_k <= 0 or not chunks:
            return []

        if self.model_name:
            try:
                return self._cross_encoder_rerank(query, chunks, top_k)
            except Exception:
                # Keep the app usable if the optional reranker is not installed.
                pass

        return self._lexical_rerank(query, chunks, top_k)

    def _cross_encoder_rerank(
        self,
        query: str,
        chunks: list[dict[str, Any]],
        top_k: int,
    ) -> list[dict[str, Any]]:
        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self.model_name)

        pairs = [(query, _rerank_text(chunk)) for chunk in chunks]
        scores = self._model.predict(pairs)
        scored = []
        for chunk, score in zip(chunks, scores):
            result = dict(chunk)
            cross_encoder_score = float(score)
            result["cross_encoder_score"] = cross_encoder_score
            result["rerank_score"] = _combined_rerank_score(
                query=query,
                chunk=chunk,
                model_score=_squash_score(cross_encoder_score),
                model_weight=0.55,
            )
            scored.append(result)
        return sorted(scored, key=lambda result: result["rerank_score"], reverse=True)[:top_k]

    @staticmethod
    def _lexical_rerank(
        query: str,
        chunks: list[dict[str, Any]],
        top_k: int,
    ) -> list[dict[str, Any]]:
        scored = []
        for chunk in chunks:
            result = dict(chunk)
            result["rerank_score"] = _combined_rerank_score(
                query=query,
                chunk=chunk,
                model_score=0.0,
                model_weight=0.0,
            )
            scored.append(result)

        return sorted(
            scored,
            key=lambda result: (
                result.get("rerank_score", 0.0),
                result.get("score", 0.0),
            ),
            reverse=True,
        )[:top_k]


def _query_term_coverage(query: str, text: str) -> float:
    query_terms = _content_terms(query)
    if not query_terms:
        return 0.0

    text_terms = set(tokenize(text))
    if not text_terms:
        return 0.0

    matches = sum(1 for term in query_terms if _has_matching_term(term, text_terms))
    return matches / len(query_terms)


def _combined_rerank_score(
    query: str,
    chunk: dict[str, Any],
    model_score: float,
    model_weight: float,
) -> float:
    rerank_text = _rerank_text(chunk)
    lexical_score = lexical_similarity(query, rerank_text)
    query_coverage = _query_term_coverage(query, rerank_text)
    retrieval_score = _bounded_score(chunk.get("score", 0.0))
    keyword_score = _bounded_score(chunk.get("keyword_score", 0.0))
    vector_score = _bounded_score(chunk.get("vector_score", 0.0))
    section_score = _bounded_score(chunk.get("section_score", 0.0))

    heuristic_score = (
        0.28 * query_coverage
        + 0.22 * lexical_score
        + 0.20 * section_score
        + 0.15 * retrieval_score
        + 0.10 * keyword_score
        + 0.05 * vector_score
    )
    return model_weight * model_score + (1 - model_weight) * heuristic_score


def _bounded_score(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, number))


def _squash_score(value: float) -> float:
    if 0.0 <= value <= 1.0:
        return value
    if value < 0:
        return 1 / (1 + abs(value))
    return value / (1 + value)


def _rerank_text(chunk: dict[str, Any]) -> str:
    metadata = chunk.get("metadata", {})
    section_title = metadata.get("section_title", "")
    return f"{section_title}\n{chunk.get('text', '')}".strip()


def _content_terms(text: str) -> list[str]:
    return [
        token
        for token in tokenize(text)
        if len(token) > 2 and token not in CONTENT_STOPWORDS
    ]


def _has_matching_term(query_term: str, text_terms: set[str]) -> bool:
    return any(_term_similarity(query_term, text_term) >= 0.8 for text_term in text_terms)


def _term_similarity(left: str, right: str) -> float:
    if left == right:
        return 1.0
    if left in right or right in left:
        return 0.85
    if _common_prefix_length(left, right) >= 5:
        return 0.8

    score = SequenceMatcher(None, left, right).ratio()
    return score if score >= 0.8 else 0.0


def _common_prefix_length(left: str, right: str) -> int:
    count = 0
    for left_char, right_char in zip(left, right):
        if left_char != right_char:
            break
        count += 1
    return count
