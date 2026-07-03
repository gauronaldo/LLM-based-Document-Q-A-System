"""Hybrid retrieval for user questions."""

from __future__ import annotations

import re
from difflib import SequenceMatcher
from dataclasses import dataclass
from typing import Any

from app.keyword_search import bm25_search, lexical_similarity, tokenize
from app.query_profiles import QueryProfile, load_query_profile
from app.reranker import Reranker


class RetrieverError(Exception):
    """Raised when retrieval cannot be completed."""


@dataclass(frozen=True)
class AutoRetrievalPlan:
    top_k: int
    similarity_threshold: float
    candidate_k: int
    force_parent_context: bool = False


CONTENT_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "by",
    "can",
    "could",
    "did",
    "do",
    "does",
    "finding",
    "findings",
    "find",
    "finds",
    "for",
    "from",
    "give",
    "how",
    "in",
    "is",
    "it",
    "key",
    "main",
    "me",
    "of",
    "on",
    "or",
    "overall",
    "outcome",
    "outcomes",
    "paper",
    "please",
    "provide",
    "related",
    "retrieval",
    "result",
    "results",
    "show",
    "studied",
    "study",
    "tell",
    "terms",
    "the",
    "this",
    "to",
    "what",
    "where",
    "why",
    "with",
    "cac",
    "các",
    "cua",
    "của",
    "gi",
    "gì",
    "goi",
    "gọi",
    "khac",
    "khác",
    "la",
    "là",
    "nao",
    "nào",
    "nguoi",
    "người",
    "nhung",
    "những",
    "ten",
    "tên",
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

YES_NO_CLAIM_STOPWORDS = CONTENT_STOPWORDS | {
    "analyze",
    "analysis",
    "claim",
    "compare",
    "comparison",
    "conclude",
    "discuss",
    "estimate",
    "estimates",
    "evaluate",
    "evidence",
    "effect",
    "effects",
    "find",
    "finding",
    "findings",
    "impact",
    "impacts",
    "provide",
    "show",
    "shows",
    "use",
    "uses",
}

YES_NO_STARTERS = {
    "are",
    "can",
    "could",
    "did",
    "do",
    "does",
    "is",
    "should",
    "was",
    "were",
}

ENTITY_TOKEN_PATTERN = re.compile(r"[\w\u00c0-\u1ef9]+", re.UNICODE)
ENTITY_START_STOPWORDS = {
    "Ai",
    "Các",
    "Cái",
    "Cho",
    "Có",
    "Gì",
    "Hãy",
    "Khi",
    "Nêu",
    "Những",
    "Ở",
    "Tại",
    "Theo",
    "Trong",
    "Vì",
}


ENGLISH_ENTITY_START_STOPWORDS = {
    "Are",
    "Can",
    "Could",
    "Did",
    "Do",
    "Does",
    "How",
    "Is",
    "Should",
    "Was",
    "Were",
    "What",
    "When",
    "Where",
    "Which",
    "Who",
    "Why",
}
ENTITY_START_STOPWORDS = ENTITY_START_STOPWORDS | ENGLISH_ENTITY_START_STOPWORDS


class Retriever:
    """Encode questions, run hybrid search, rerank, and filter evidence."""

    def __init__(
        self,
        vector_store: Any,
        embedder: Any,
        top_k: int = 5,
        similarity_threshold: float = 0.3,
        hybrid_alpha: float = 0.7,
        use_hybrid_search: bool = True,
        use_mmr: bool = True,
        reranker: Reranker | None = None,
        query_profile: QueryProfile | None = None,
        use_multi_query: bool = False,
        multi_query_count: int = 4,
    ):
        if top_k <= 0:
            raise ValueError("top_k must be greater than 0.")
        if not 0 <= similarity_threshold <= 1:
            raise ValueError("similarity_threshold must be between 0 and 1.")
        if not 0 <= hybrid_alpha <= 1:
            raise ValueError("hybrid_alpha must be between 0 and 1.")

        self.vector_store = vector_store
        self.embedder = embedder
        self.top_k = top_k
        self.similarity_threshold = similarity_threshold
        self.hybrid_alpha = hybrid_alpha
        self.use_hybrid_search = use_hybrid_search
        self.use_mmr = use_mmr
        self.reranker = reranker or Reranker()
        self.query_profile = query_profile or load_query_profile()
        self.use_multi_query = use_multi_query
        self.multi_query_count = max(1, multi_query_count)
        self._all_chunks_cache: list[dict[str, Any]] | None = None
        self._searchable_chunks_cache: list[dict[str, Any]] | None = None

    def retrieve(
        self,
        question: str,
        top_k: int | None = None,
        similarity_threshold: float | None = None,
        candidate_k: int | None = None,
        auto: bool = False,
        intent: str | None = None,
        original_question: str | None = None,
    ) -> list[dict[str, Any]]:
        """Return relevant, diverse chunks for a question."""

        clean_question = question.strip()
        if not clean_question:
            return []
        answer_question = (original_question or clean_question).strip() or clean_question
        guard_question = clean_question

        if auto:
            plan = self._auto_retrieval_plan(
                answer_question,
                intent=intent,
                top_k=top_k,
                similarity_threshold=similarity_threshold,
                candidate_k=candidate_k,
            )
            search_top_k = plan.top_k
            threshold = plan.similarity_threshold
            candidate_count = plan.candidate_k
            force_parent_context = plan.force_parent_context
        else:
            search_top_k = self.top_k if top_k is None else top_k
            threshold = self.similarity_threshold if similarity_threshold is None else similarity_threshold
            candidate_count = candidate_k or max(search_top_k * 8, 40)
            force_parent_context = False

        if search_top_k <= 0:
            raise ValueError("top_k must be greater than 0.")

        if self.use_multi_query:
            candidates = self._multi_query_candidates(clean_question, candidate_count)
        else:
            vector_results = self._vector_search(clean_question, candidate_count)
            keyword_results = self._keyword_search(clean_question, candidate_count)
            section_results = self._section_search(clean_question, candidate_count)
            candidates = self._merge_results(
                vector_results,
                keyword_results,
                section_results,
                query=clean_question,
            )
        candidates = _filter_or_penalize_entity_mismatches(guard_question, candidates)
        candidates = [
            result for result in candidates if float(result.get("score", 0.0)) >= threshold
        ]
        if not candidates:
            return []

        candidates = sorted(candidates, key=lambda result: result.get("score", 0.0), reverse=True)
        reranked = self._maybe_rerank(guard_question, candidates, search_top_k, auto=auto)

        if self._should_use_mmr(answer_question, intent, search_top_k):
            selected = self._select_with_preserved_top_candidate(
                guard_question,
                reranked,
                top_k=search_top_k,
            )
        else:
            selected = reranked[:search_top_k]

        expanded = [
            self._expand_parent_context(
                result,
                query=guard_question,
                force=force_parent_context,
            )
            for result in selected
        ]
        if not self._has_sufficient_evidence(guard_question, expanded):
            return []

        return expanded

    def _multi_query_candidates(self, question: str, candidate_k: int) -> list[dict[str, Any]]:
        query_variants = _query_variants(question, self.multi_query_count)
        if len(query_variants) <= 1:
            vector_results = self._vector_search(question, candidate_k)
            keyword_results = self._keyword_search(question, candidate_k)
            section_results = self._section_search(question, candidate_k)
            return self._merge_results(vector_results, keyword_results, section_results)

        ranked_lists = []
        for variant in query_variants:
            vector_results = self._vector_search(variant, candidate_k)
            keyword_results = self._keyword_search(variant, candidate_k)
            section_results = self._section_search(variant, candidate_k)
            ranked = sorted(
                self._merge_results(
                    vector_results,
                    keyword_results,
                    section_results,
                    query=variant,
                ),
                key=lambda result: result.get("score", 0.0),
                reverse=True,
            )
            ranked_lists.append(ranked[:candidate_k])
        return _reciprocal_rank_fusion(ranked_lists, top_k=candidate_k)

    def clear_cache(self) -> None:
        """Clear cached chunk/index data after vector store changes."""

        self._all_chunks_cache = None
        self._searchable_chunks_cache = None

    def _vector_search(self, question: str, candidate_k: int) -> list[dict[str, Any]]:
        try:
            query_embeddings = self.embedder.encode([question])
        except Exception as exc:
            raise RetrieverError(f"Failed to encode question: {exc}") from exc

        if not query_embeddings:
            return []

        try:
            results = self.vector_store.search(query_embeddings[0], top_k=candidate_k)
        except Exception as exc:
            raise RetrieverError(f"Failed to retrieve chunks: {exc}") from exc

        vector_results = []
        for result in results:
            item = dict(result)
            item["vector_score"] = float(result.get("score", 0.0))
            vector_results.append(item)
        return vector_results

    def _keyword_search(self, question: str, candidate_k: int) -> list[dict[str, Any]]:
        if not self.use_hybrid_search or not hasattr(self.vector_store, "get_all"):
            return []

        chunks = self._all_chunks()

        keyword_query = " ".join(_content_terms(question)) or question
        results = bm25_search(keyword_query, self._searchable_chunks(), top_k=candidate_k)
        for result in results:
            if "original_text" in result:
                result["text"] = result.pop("original_text")
        return results

    def _section_search(self, question: str, candidate_k: int) -> list[dict[str, Any]]:
        if not hasattr(self.vector_store, "get_all"):
            return []

        chunks = self._all_chunks()

        scored = []
        for chunk in chunks:
            metadata = chunk.get("metadata", {})
            section_title = str(metadata.get("section_title", "")).strip()
            if not section_title:
                continue

            section_score = _section_title_similarity(question, section_title)
            profile_boost = self.query_profile.section_boost(
                question,
                section_title,
                chunk.get("text", ""),
                _content_terms(question),
            )
            section_score = max(section_score, profile_boost)
            if section_score <= 0:
                continue

            result = dict(chunk)
            result["section_score"] = section_score
            scored.append(result)

        return sorted(scored, key=lambda result: result["section_score"], reverse=True)[:candidate_k]

    def _all_chunks(self) -> list[dict[str, Any]]:
        if self._all_chunks_cache is not None:
            return self._all_chunks_cache

        try:
            self._all_chunks_cache = self.vector_store.get_all()
        except Exception:
            self._all_chunks_cache = []
        return self._all_chunks_cache

    def _searchable_chunks(self) -> list[dict[str, Any]]:
        if self._searchable_chunks_cache is not None:
            return self._searchable_chunks_cache

        searchable_chunks = []
        for chunk in self._all_chunks():
            metadata = chunk.get("metadata", {})
            section_title = metadata.get("section_title", "")
            item = dict(chunk)
            item["original_text"] = chunk.get("text", "")
            item["text"] = f"{section_title}\n{section_title}\n{chunk.get('text', '')}".strip()
            searchable_chunks.append(item)

        self._searchable_chunks_cache = searchable_chunks
        return searchable_chunks

    def _merge_results(
        self,
        vector_results: list[dict[str, Any]],
        keyword_results: list[dict[str, Any]],
        section_results: list[dict[str, Any]],
        query: str = "",
    ) -> list[dict[str, Any]]:
        merged: dict[str, dict[str, Any]] = {}

        for result in vector_results:
            chunk_id = result["chunk_id"]
            item = dict(result)
            item["keyword_score"] = 0.0
            item["section_score"] = 0.0
            merged[chunk_id] = item

        for result in keyword_results:
            chunk_id = result["chunk_id"]
            if chunk_id not in merged:
                item = dict(result)
                item["vector_score"] = 0.0
                item["section_score"] = 0.0
                merged[chunk_id] = item
            merged[chunk_id]["keyword_score"] = float(result.get("keyword_score", 0.0))

        for result in section_results:
            chunk_id = result["chunk_id"]
            if chunk_id not in merged:
                item = dict(result)
                item["vector_score"] = 0.0
                item["keyword_score"] = 0.0
                merged[chunk_id] = item
            merged[chunk_id]["section_score"] = float(result.get("section_score", 0.0))

        has_vector_results = bool(vector_results)
        has_keyword_results = bool(keyword_results)
        has_section_results = bool(section_results)
        for item in merged.values():
            vector_score = float(item.get("vector_score", item.get("score", 0.0)))
            keyword_score = float(item.get("keyword_score", 0.0))
            section_score = float(item.get("section_score", 0.0))
            if has_vector_results and has_keyword_results and has_section_results:
                semantic_score = self.hybrid_alpha * vector_score + (1 - self.hybrid_alpha) * keyword_score
                item["score"] = max(semantic_score, 0.65 * section_score + 0.35 * semantic_score)
            elif has_vector_results and has_keyword_results:
                item["score"] = self.hybrid_alpha * vector_score + (1 - self.hybrid_alpha) * keyword_score
            elif has_section_results:
                item["score"] = max(vector_score, keyword_score, section_score)
            elif has_vector_results:
                item["score"] = vector_score
            else:
                item["score"] = keyword_score
            entity_score = _entity_match_score(item)
            if entity_score:
                item["entity_score"] = entity_score
                item["score"] = max(float(item.get("score", 0.0)), entity_score)
            evidence_boost = self._profile_evidence_boost(query, item)
            if evidence_boost:
                item["profile_evidence_boost"] = evidence_boost
                item["score"] = min(1.0, float(item.get("score", 0.0)) + evidence_boost)

        return list(merged.values())

    def _profile_evidence_boost(self, query: str, item: dict[str, Any]) -> float:
        if not query:
            return 0.0
        metadata = item.get("metadata", {})
        return self.query_profile.evidence_boost(
            query=query,
            section_title=str(metadata.get("section_title", "")),
            chunk_text=" ".join(
                str(part)
                for part in (
                    item.get("matched_text", ""),
                    item.get("text", ""),
                )
                if part
            ),
        )

    def _auto_retrieval_plan(
        self,
        query: str,
        intent: str | None,
        top_k: int | None,
        similarity_threshold: float | None,
        candidate_k: int | None,
    ) -> AutoRetrievalPlan:
        query_terms = _content_terms(query)
        token_count = len(tokenize(query))
        intent = intent or "qa"

        if intent == "summary":
            auto_top_k = max(top_k or self.top_k, 10)
            threshold = 0.0 if similarity_threshold is None else similarity_threshold
            return AutoRetrievalPlan(
                top_k=auto_top_k,
                similarity_threshold=threshold,
                candidate_k=candidate_k or max(auto_top_k * 6, 60),
                force_parent_context=True,
            )

        if intent in {"comparison", "explanation", "extraction"}:
            auto_top_k = max(top_k or self.top_k, 7)
            threshold = min(self.similarity_threshold, 0.2) if similarity_threshold is None else similarity_threshold
            return AutoRetrievalPlan(
                top_k=auto_top_k,
                similarity_threshold=threshold,
                candidate_k=candidate_k or max(auto_top_k * 6, 50),
                force_parent_context=True,
            )

        if token_count <= 7 and len(query_terms) <= 3:
            auto_top_k = min(top_k or self.top_k, 4)
            threshold = max(self.similarity_threshold, 0.35) if similarity_threshold is None else similarity_threshold
            candidate_count = candidate_k or max(auto_top_k * 6, 30)
        else:
            auto_top_k = top_k or self.top_k
            threshold = self.similarity_threshold if similarity_threshold is None else similarity_threshold
            candidate_count = candidate_k or max(auto_top_k * 8, 40)

        return AutoRetrievalPlan(
            top_k=auto_top_k,
            similarity_threshold=threshold,
            candidate_k=candidate_count,
            force_parent_context=False,
        )

    def _maybe_rerank(
        self,
        query: str,
        candidates: list[dict[str, Any]],
        top_k: int,
        auto: bool,
    ) -> list[dict[str, Any]]:
        rerank_top_k = max(top_k * 2, top_k)
        if not auto or self._should_rerank(query, candidates):
            return self.reranker.rerank(query, candidates, top_k=rerank_top_k)
        return candidates[:rerank_top_k]

    @staticmethod
    def _should_rerank(query: str, candidates: list[dict[str, Any]]) -> bool:
        if len(candidates) <= 1:
            return False
        if len(_content_terms(query)) >= 5:
            return True

        top_score = float(candidates[0].get("score", 0.0))
        comparison_window = candidates[1: min(len(candidates), 6)]
        return any(top_score - float(candidate.get("score", 0.0)) <= 0.08 for candidate in comparison_window)

    def _should_use_mmr(self, query: str, intent: str | None, top_k: int) -> bool:
        if not self.use_mmr:
            return False
        if intent in {"summary", "comparison"}:
            return True
        if top_k >= 8 and len(_content_terms(query)) >= 5:
            return True
        return False

    @staticmethod
    def _expand_parent_context(
        result: dict[str, Any],
        query: str = "",
        force: bool = False,
    ) -> dict[str, Any]:
        metadata = result.get("metadata", {})
        parent_text = metadata.get("parent_text")
        if not parent_text or len(parent_text) <= len(result.get("text", "")):
            return result
        if not force and not _should_expand_parent_context(query, result):
            return result

        expanded = dict(result)
        expanded["matched_text"] = result.get("text", "")
        expanded["text"] = parent_text
        return expanded

    @staticmethod
    def _has_sufficient_evidence(query: str, chunks: list[dict[str, Any]]) -> bool:
        if not chunks:
            return False

        context = " ".join(
            f"{chunk.get('metadata', {}).get('section_title', '')} {chunk.get('text', '')}"
            for chunk in chunks
        )
        if not context.strip():
            return True

        query_terms = _content_terms(query)
        if not query_terms:
            return True

        entity_phrases = _query_entity_phrases(query)
        if entity_phrases and not _chunks_support_entity_phrases(entity_phrases, chunks):
            return False

        context_terms = set(tokenize(context))
        matched_terms = [term for term in query_terms if _has_matching_term(term, context_terms)]
        coverage = len(matched_terms) / len(query_terms)

        if _is_yes_no_topic_question(query):
            claim_terms = _claim_terms(query)
            if claim_terms and not _chunks_support_claim_terms(claim_terms, chunks):
                return False

        if coverage >= 0.45:
            return True

        if any(
            _section_title_similarity(query, str(chunk.get("metadata", {}).get("section_title", ""))) >= 0.45
            for chunk in chunks
        ):
            return True

        return max(lexical_similarity(query, chunk.get("text", "")) for chunk in chunks) >= 0.08

    @staticmethod
    def _select_with_preserved_top_candidate(
        query: str,
        candidates: list[dict[str, Any]],
        top_k: int,
    ) -> list[dict[str, Any]]:
        if top_k <= 0 or not candidates:
            return []

        preserved = candidates[0]
        if top_k == 1:
            return [preserved]

        remaining = candidates[1:]
        diverse_tail = Retriever._mmr_select(query, remaining, top_k=top_k - 1)
        return [preserved, *diverse_tail]

    @staticmethod
    def _mmr_select(
        query: str,
        candidates: list[dict[str, Any]],
        top_k: int,
        lambda_mult: float = 0.7,
    ) -> list[dict[str, Any]]:
        if top_k <= 0:
            return []

        selected: list[dict[str, Any]] = []
        remaining = list(candidates)

        while remaining and len(selected) < top_k:
            best_result = None
            best_score = float("-inf")

            for candidate in remaining:
                relevance = float(candidate.get("score", 0.0))
                diversity_penalty = 0.0
                if selected:
                    diversity_penalty = max(
                        lexical_similarity(candidate.get("text", ""), item.get("text", ""))
                        for item in selected
                    )
                query_overlap = lexical_similarity(query, candidate.get("text", ""))
                mmr_score = lambda_mult * max(relevance, query_overlap) - (1 - lambda_mult) * diversity_penalty

                if mmr_score > best_score:
                    best_score = mmr_score
                    best_result = candidate

            selected.append(best_result)
            remaining.remove(best_result)

        return selected


def _content_terms(text: str) -> list[str]:
    return [
        token
        for token in tokenize(text)
        if (len(token) > 2 or token == "ai") and token not in CONTENT_STOPWORDS
    ]


def _query_variants(question: str, max_variants: int) -> list[str]:
    clean = question.strip()
    if max_variants <= 1:
        return [clean]

    parts = [part.strip() for part in clean.splitlines() if part.strip()]
    base = parts[0] if parts else clean
    variants = [clean]
    if base != clean:
        variants.append(base)

    related_terms = _related_terms_from_query(clean)
    for term in related_terms:
        variants.append(f"{base} {term}")
        if len(variants) >= max_variants:
            break

    deduped = []
    seen = set()
    for variant in variants:
        key = variant.lower()
        if key not in seen:
            seen.add(key)
            deduped.append(variant)
    return deduped[:max_variants] or [clean]


def _related_terms_from_query(query: str) -> list[str]:
    marker = "related retrieval terms:"
    lowered = query.lower()
    if marker not in lowered:
        return []
    related = query[lowered.index(marker) + len(marker):]
    terms = []
    for part in related.replace("\n", ",").split(","):
        term = part.strip()
        if term:
            terms.append(term)
    return terms


def _reciprocal_rank_fusion(
    ranked_lists: list[list[dict[str, Any]]],
    top_k: int,
    k: int = 60,
) -> list[dict[str, Any]]:
    fused: dict[str, dict[str, Any]] = {}
    scores: dict[str, float] = {}

    for ranked in ranked_lists:
        for rank, item in enumerate(ranked, start=1):
            chunk_id = item.get("chunk_id")
            if not chunk_id:
                continue
            if chunk_id not in fused or float(item.get("score", 0.0)) > float(fused[chunk_id].get("score", 0.0)):
                fused[chunk_id] = dict(item)
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)

    results = []
    max_score = max(scores.values()) if scores else 0.0
    for chunk_id, item in fused.items():
        rrf_score = scores.get(chunk_id, 0.0)
        result = dict(item)
        result["rrf_score"] = rrf_score
        normalized_rrf = rrf_score / max_score if max_score else 0.0
        base_score = float(result.get("score", 0.0))
        result["base_score"] = base_score
        result["rank_score"] = normalized_rrf
        result["score"] = min(1.0, 0.85 * base_score + 0.15 * normalized_rrf)
        results.append(result)

    return sorted(
        results,
        key=lambda result: (result.get("rrf_score", 0.0), result.get("score", 0.0)),
        reverse=True,
    )[:top_k]


def _claim_terms(text: str) -> list[str]:
    return [
        token
        for token in tokenize(text)
        if len(token) > 2 and token not in YES_NO_CLAIM_STOPWORDS
    ]


def _has_supported_claim_terms(claim_terms: list[str], context_terms: set[str]) -> bool:
    matched_terms = [term for term in claim_terms if _has_matching_term(term, context_terms)]
    coverage = len(matched_terms) / len(claim_terms)
    missing_terms = [term for term in claim_terms if term not in matched_terms]
    missing_distinctive_terms = [term for term in missing_terms if len(term) >= 7]

    if missing_distinctive_terms:
        return False
    if len(missing_terms) >= 2:
        return False
    return coverage >= 0.7


def _chunks_support_claim_terms(claim_terms: list[str], chunks: list[dict[str, Any]]) -> bool:
    for chunk in chunks:
        metadata = chunk.get("metadata", {})
        chunk_context = " ".join(
            str(part)
            for part in (
                metadata.get("section_title", ""),
                chunk.get("matched_text", ""),
                chunk.get("text", ""),
            )
            if part
        )
        if _has_supported_claim_terms(claim_terms, set(tokenize(chunk_context))):
            return True
    return False


def _is_yes_no_topic_question(query: str) -> bool:
    tokens = tokenize(query)
    return bool(tokens and tokens[0] in YES_NO_STARTERS)


def _should_expand_parent_context(query: str, result: dict[str, Any]) -> bool:
    if not query:
        return False

    query_terms = _content_terms(query)
    if len(query_terms) >= 5:
        return True

    metadata = result.get("metadata", {})
    section_title = str(metadata.get("section_title", ""))
    if _section_title_similarity(query, section_title) >= 0.45:
        return True

    matched_text = result.get("text", "")
    parent_text = metadata.get("parent_text", "")
    if not parent_text:
        return False

    return lexical_similarity(query, matched_text) < lexical_similarity(query, parent_text)


def _section_title_similarity(query: str, section_title: str) -> float:
    query_terms = _content_terms(query)
    title_terms = _content_terms(section_title)
    if not query_terms or not title_terms:
        return 0.0

    token_overlap = len(set(query_terms) & set(title_terms)) / len(set(query_terms) | set(title_terms))
    fuzzy_matches = []
    for query_term in query_terms:
        best_match = max(
            _term_similarity(query_term, title_term)
            for title_term in title_terms
        )
        fuzzy_matches.append(best_match)

    fuzzy_score = sum(fuzzy_matches) / len(fuzzy_matches)
    return max(token_overlap, fuzzy_score)


def _term_similarity(left: str, right: str) -> float:
    if left == right:
        return 1.0
    if left in right or right in left:
        return 0.85
    if _common_prefix_length(left, right) >= 5:
        return 0.8

    sequence_score = SequenceMatcher(None, left, right).ratio()
    return sequence_score if sequence_score >= 0.8 else 0.0


def _has_matching_term(query_term: str, text_terms: set[str]) -> bool:
    return any(_term_similarity(query_term, text_term) >= 0.8 for text_term in text_terms)


def _common_prefix_length(left: str, right: str) -> int:
    count = 0
    for left_char, right_char in zip(left, right):
        if left_char != right_char:
            break
        count += 1
    return count


def _query_entity_phrases(query: str) -> list[str]:
    """Extract multi-token proper-name phrases from the user's original query."""

    phrases: list[str] = []
    current: list[str] = []

    for token in ENTITY_TOKEN_PATTERN.findall(query):
        if _looks_like_entity_token(token):
            current.append(token)
            continue

        if len(current) >= 2:
            phrases.append(" ".join(current))
        current = []

    if len(current) >= 2:
        phrases.append(" ".join(current))

    return [
        phrase
        for phrase in phrases
        if phrase.split()[0] not in ENTITY_START_STOPWORDS
    ]


def _looks_like_entity_token(token: str) -> bool:
    if not token:
        return False
    if token in ENTITY_START_STOPWORDS:
        return False
    return token[0].isupper() and any(character.isalpha() for character in token)


def _filter_or_penalize_entity_mismatches(
    query: str,
    candidates: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    entity_phrases = _query_entity_phrases(query)
    if not entity_phrases:
        return candidates

    exact_matches = [
        _with_entity_score(candidate, query, entity_phrases)
        for candidate in candidates
        if _chunk_supports_all_entity_phrases(candidate, entity_phrases)
    ]
    if exact_matches:
        return exact_matches

    penalized = []
    for candidate in candidates:
        item = dict(candidate)
        if _chunk_has_conflicting_entity_phrase(candidate, entity_phrases):
            item["entity_mismatch"] = True
            item["score"] = float(item.get("score", 0.0)) * 0.05
        penalized.append(item)
    return penalized


def _with_entity_score(
    candidate: dict[str, Any],
    query: str,
    entity_phrases: list[str],
) -> dict[str, Any]:
    item = dict(candidate)
    entity_score = _entity_phrase_coverage(entity_phrases, _chunk_entity_text(candidate))
    lexical_score = lexical_similarity(query, _chunk_entity_text(candidate))
    item["entity_score"] = max(entity_score, lexical_score)
    item["score"] = max(float(item.get("score", 0.0)), 0.75 + 0.2 * entity_score)
    return item


def _entity_match_score(candidate: dict[str, Any]) -> float:
    return float(candidate.get("entity_score", 0.0) or 0.0)


def _chunks_support_entity_phrases(
    entity_phrases: list[str],
    chunks: list[dict[str, Any]],
) -> bool:
    return any(_chunk_supports_all_entity_phrases(chunk, entity_phrases) for chunk in chunks)


def _chunk_supports_all_entity_phrases(
    chunk: dict[str, Any],
    entity_phrases: list[str],
) -> bool:
    text = _chunk_entity_text(chunk)
    return all(_contains_token_phrase(text, phrase) for phrase in entity_phrases)


def _chunk_has_conflicting_entity_phrase(
    chunk: dict[str, Any],
    entity_phrases: list[str],
) -> bool:
    text_terms = tokenize(_chunk_entity_text(chunk))
    for phrase in entity_phrases:
        phrase_terms = tokenize(phrase)
        if len(phrase_terms) < 2 or _contains_token_phrase_terms(text_terms, phrase_terms):
            continue
        first_term = phrase_terms[0]
        for left, right in zip(text_terms, text_terms[1:]):
            if left == first_term and right != phrase_terms[1]:
                return True
    return False


def _entity_phrase_coverage(entity_phrases: list[str], text: str) -> float:
    if not entity_phrases:
        return 0.0
    matches = sum(1 for phrase in entity_phrases if _contains_token_phrase(text, phrase))
    return matches / len(entity_phrases)


def _contains_token_phrase(text: str, phrase: str) -> bool:
    return _contains_token_phrase_terms(tokenize(text), tokenize(phrase))


def _contains_token_phrase_terms(text_terms: list[str], phrase_terms: list[str]) -> bool:
    if not phrase_terms or len(phrase_terms) > len(text_terms):
        return False
    phrase_length = len(phrase_terms)
    return any(
        text_terms[index : index + phrase_length] == phrase_terms
        for index in range(len(text_terms) - phrase_length + 1)
    )


def _chunk_entity_text(chunk: dict[str, Any]) -> str:
    metadata = chunk.get("metadata", {})
    return " ".join(
        str(part)
        for part in (
            metadata.get("section_title", ""),
            chunk.get("matched_text", ""),
            chunk.get("text", ""),
        )
        if part
    )
