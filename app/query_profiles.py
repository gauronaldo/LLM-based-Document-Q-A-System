"""Configurable query profiles for domain-agnostic retrieval behavior."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from app.keyword_search import tokenize


DEFAULT_PROFILE_NAME = "general"
DEFAULT_PROFILE_PATH = Path(__file__).resolve().parents[1] / "config" / "query_profiles.json"


@dataclass(frozen=True)
class QueryProfile:
    """Retrieval and prompt hints loaded from configuration."""

    name: str
    description: str = ""
    prompt_extension: str = ""
    heading_terms: tuple[str, ...] = ()
    query_expansions: dict[str, tuple[str, ...]] | None = None
    section_intents: dict[str, dict[str, tuple[str, ...]]] | None = None
    retrieval_intents: dict[str, dict[str, tuple[str, ...]]] | None = None
    scoring_weights: dict[str, float] | None = None

    def expand_query(self, query: str, max_terms: int = 12) -> str:
        """Append configured related terms without replacing the user's wording."""

        expansions = self.query_expansions or {}
        query_terms = set(tokenize(query))
        added: list[str] = []

        for trigger, related_terms in expansions.items():
            trigger_terms = set(tokenize(trigger))
            if not trigger_terms or not trigger_terms <= query_terms:
                continue
            for term in related_terms:
                if term.lower() not in query.lower() and term not in added:
                    added.append(term)
                if len(added) >= max_terms:
                    break
            if len(added) >= max_terms:
                break

        if not added:
            return query
        return f"{query}\nRelated retrieval terms: {', '.join(added)}"

    def section_boost(
        self,
        query: str,
        section_title: str,
        chunk_text: str,
        focus_terms: list[str],
    ) -> float:
        """Return a configurable section boost for matching section-like questions."""

        query_tokens = set(tokenize(query))
        title_tokens = set(tokenize(section_title))
        if not query_tokens or not title_tokens:
            return 0.0

        for intent in (self.section_intents or {}).values():
            query_terms = set(_tokenize_many(intent.get("query_terms", ())))
            section_terms = set(_tokenize_many(intent.get("section_terms", ())))
            if not query_terms or not section_terms:
                continue
            if not (query_tokens & query_terms and title_tokens & section_terms):
                continue
            if not _chunk_supports_focus_terms(chunk_text, focus_terms, query_terms):
                continue
            return 0.72

        return 0.0

    def evidence_boost(
        self,
        query: str,
        section_title: str,
        chunk_text: str,
    ) -> float:
        """Return a small profile-configured boost for section and phrase evidence."""

        policy = self.retrieval_policy(query)
        if not policy:
            return 0.0

        weights = {
            "section_boost_weight": 0.08,
            "phrase_match_boost_weight": 0.15,
            "exact_phrase_boost_weight": 0.2,
        }
        weights.update(self.scoring_weights or {})

        query_tokens = set(tokenize(query))
        section_tokens = set(tokenize(section_title))
        text_tokens = set(tokenize(chunk_text))
        searchable_text = " ".join(tokenize(f"{section_title} {chunk_text}"))

        section_score = _overlap_ratio(
            section_tokens,
            set(_tokenize_many(policy.get("section_terms", ()))),
        )
        phrase_score = _overlap_ratio(
            text_tokens | section_tokens,
            set(_tokenize_many(policy.get("phrase_terms", ()))),
        )
        query_focus_score = _query_focus_score(query_tokens, text_tokens | section_tokens)
        exact_phrase_score = _exact_phrase_score(
            searchable_text,
            tuple(policy.get("phrase_terms", ())),
            query,
        )

        return min(
            0.35,
            weights["section_boost_weight"] * section_score
            + weights["phrase_match_boost_weight"] * max(phrase_score, query_focus_score)
            + weights["exact_phrase_boost_weight"] * exact_phrase_score,
        )

    def retrieval_policy(self, query: str) -> dict[str, tuple[str, ...]] | None:
        """Return the first configured retrieval policy that matches the query."""

        query_tokens = set(tokenize(query))
        if not query_tokens:
            return None
        for policy in (self.retrieval_intents or {}).values():
            query_terms = set(_tokenize_many(policy.get("query_terms", ())))
            if query_terms and query_tokens & query_terms:
                return policy
        return None


def active_profile_name() -> str:
    """Return the configured document/query profile name."""

    return (
        os.getenv("QUERY_PROFILE")
        or os.getenv("DOCUMENT_PROFILE")
        or DEFAULT_PROFILE_NAME
    ).strip().lower()


def load_query_profile(name: str | None = None) -> QueryProfile:
    """Load a query profile, falling back to the general profile."""

    profile_name = (name or active_profile_name() or DEFAULT_PROFILE_NAME).strip().lower()
    profiles = _load_profiles(DEFAULT_PROFILE_PATH)
    if profile_name not in profiles:
        profile_name = DEFAULT_PROFILE_NAME
    data = _resolve_profile(profile_name, profiles)
    return QueryProfile(
        name=profile_name,
        description=str(data.get("description", "")),
        prompt_extension=str(data.get("prompt_extension", "")),
        heading_terms=tuple(data.get("heading_terms", ())),
        query_expansions={
            key: tuple(value)
            for key, value in dict(data.get("query_expansions", {})).items()
        },
        section_intents={
            key: {
                "query_terms": tuple(value.get("query_terms", ())),
                "section_terms": tuple(value.get("section_terms", ())),
            }
            for key, value in dict(data.get("section_intents", {})).items()
        },
        retrieval_intents={
            key: {
                "query_terms": tuple(value.get("query_terms", ())),
                "section_terms": tuple(value.get("section_terms", ())),
                "phrase_terms": tuple(value.get("phrase_terms", ())),
            }
            for key, value in dict(data.get("retrieval_intents", {})).items()
        },
        scoring_weights={
            key: float(value)
            for key, value in dict(data.get("scoring_weights", {})).items()
        },
    )


@lru_cache(maxsize=8)
def _load_profiles(path: Path) -> dict[str, dict[str, Any]]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _resolve_profile(name: str, profiles: dict[str, dict[str, Any]]) -> dict[str, Any]:
    current = dict(profiles.get(name, {}))
    parent_name = current.pop("extends", None)
    if not parent_name:
        return current

    parent = _resolve_profile(str(parent_name), profiles)
    merged = dict(parent)
    for key, value in current.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            nested = dict(merged[key])
            for nested_key, nested_value in value.items():
                if (
                    isinstance(nested_value, dict)
                    and isinstance(nested.get(nested_key), dict)
                ):
                    inner = dict(nested[nested_key])
                    inner.update(nested_value)
                    nested[nested_key] = inner
                else:
                    nested[nested_key] = nested_value
            merged[key] = nested
        else:
            if isinstance(value, list) and isinstance(merged.get(key), list):
                merged[key] = [*merged[key], *[item for item in value if item not in merged[key]]]
            else:
                merged[key] = value
    return merged


def _tokenize_many(values: tuple[str, ...] | list[str]) -> list[str]:
    tokens: list[str] = []
    for value in values:
        tokens.extend(tokenize(value))
    return tokens


def _chunk_supports_focus_terms(
    chunk_text: str,
    focus_terms: list[str],
    intent_query_terms: set[str],
) -> bool:
    distinctive_terms = [
        term for term in focus_terms if term not in intent_query_terms and len(term) > 2
    ]
    if not distinctive_terms:
        return True

    chunk_terms = set(tokenize(chunk_text))
    matched = sum(1 for term in distinctive_terms if term in chunk_terms)
    return matched / len(distinctive_terms) >= 0.5


def _overlap_ratio(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(right)


def _query_focus_score(query_tokens: set[str], text_tokens: set[str]) -> float:
    focus_terms = [
        term
        for term in query_tokens
        if len(term) > 3 and term not in _PROFILE_STOPWORDS
    ]
    if not focus_terms:
        return 0.0
    matched = sum(1 for term in focus_terms if term in text_tokens)
    return matched / len(focus_terms)


def _exact_phrase_score(searchable_text: str, phrase_terms: tuple[str, ...], query: str) -> float:
    phrases = set()
    for phrase in phrase_terms:
        normalized = " ".join(tokenize(phrase))
        if " " in normalized:
            phrases.add(normalized)
    query_tokens = [
        token for token in tokenize(query)
        if len(token) > 3 and token not in _PROFILE_STOPWORDS
    ]
    for index in range(len(query_tokens) - 1):
        phrases.add(f"{query_tokens[index]} {query_tokens[index + 1]}")

    if not phrases:
        return 0.0
    hits = sum(1 for phrase in phrases if phrase in searchable_text)
    return min(1.0, hits / max(1, min(len(phrases), 4)))


_PROFILE_STOPWORDS = {
    "about",
    "does",
    "from",
    "have",
    "paper",
    "study",
    "that",
    "what",
    "when",
    "where",
    "which",
    "with",
}
