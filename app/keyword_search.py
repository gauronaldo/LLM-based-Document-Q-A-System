"""Lightweight keyword/BM25 search utilities for hybrid retrieval."""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any


TOKEN_PATTERN = re.compile(r"[\w\u00c0-\u1ef9]+", re.UNICODE)


def tokenize(text: str) -> list[str]:
    """Tokenize English/Vietnamese text for lexical retrieval."""

    return [
        normalized
        for token in TOKEN_PATTERN.findall(text)
        if (normalized := normalize_token(token))
    ]


def normalize_token(token: str) -> str:
    """Normalize light English morphology without changing Vietnamese tokens."""

    normalized = token.lower()
    if not normalized.isascii() or not normalized.isalpha():
        return normalized

    if len(normalized) > 4 and normalized.endswith("ies"):
        return f"{normalized[:-3]}y"

    if len(normalized) > 5 and normalized.endswith(("ches", "shes", "sses", "xes", "zes")):
        return normalized[:-2]

    if (
        len(normalized) > 4
        and normalized.endswith("s")
        and not normalized.endswith(("ss", "sis", "us"))
    ):
        return normalized[:-1]

    return normalized


def bm25_search(
    query: str,
    chunks: list[dict[str, Any]],
    top_k: int = 10,
    k1: float = 1.5,
    b: float = 0.75,
) -> list[dict[str, Any]]:
    """Return chunks ranked with a small BM25 implementation."""

    if top_k <= 0 or not query.strip() or not chunks:
        return []

    query_terms = tokenize(query)
    if not query_terms:
        return []

    tokenized_docs = [tokenize(chunk.get("text", "")) for chunk in chunks]
    doc_lengths = [len(tokens) for tokens in tokenized_docs]
    avg_doc_length = sum(doc_lengths) / len(doc_lengths) if doc_lengths else 0.0
    if avg_doc_length == 0:
        return []

    document_frequency = Counter()
    for tokens in tokenized_docs:
        document_frequency.update(set(tokens))

    scored = []
    total_docs = len(chunks)
    for chunk, tokens, doc_length in zip(chunks, tokenized_docs, doc_lengths):
        term_frequency = Counter(tokens)
        score = 0.0
        for term in query_terms:
            if term not in term_frequency:
                continue

            df = document_frequency[term]
            idf = math.log(1 + (total_docs - df + 0.5) / (df + 0.5))
            numerator = term_frequency[term] * (k1 + 1)
            denominator = term_frequency[term] + k1 * (1 - b + b * doc_length / avg_doc_length)
            score += idf * numerator / denominator

        if score > 0:
            result = dict(chunk)
            result["keyword_score_raw"] = score
            scored.append(result)

    if not scored:
        return []

    max_score = max(result["keyword_score_raw"] for result in scored)
    for result in scored:
        result["keyword_score"] = result["keyword_score_raw"] / max_score if max_score else 0.0

    return sorted(scored, key=lambda result: result["keyword_score"], reverse=True)[:top_k]


def lexical_similarity(left: str, right: str) -> float:
    """Return Jaccard similarity over lexical tokens."""

    left_tokens = set(tokenize(left))
    right_tokens = set(tokenize(right))
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)
