"""Shared helpers for fast, deterministic RAG debugging."""

from __future__ import annotations

import csv
import hashlib
import json
import sys
import time
import unicodedata
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config import get_config
from app.answer_postprocessor import looks_like_refusal, postprocess_answer_behavior
from app.citation_validator import post_check_citations
from app.context_support import (
    NO_SUPPORT,
    PARTIAL_SUPPORT,
    STRONG_SUPPORT,
    estimate_context_support,
)
from app.prompt_template import (
    answer_needs_repair,
    answer_language_mismatch,
    build_answer_required_prompt,
    build_answer_repair_prompt,
    build_language_repair_prompt,
    build_prompt,
    normalize_answer,
    refusal_for_question,
)
from app.query_intent import detect_query_intent, retrieval_plan_for_intent
from app.query_rewriter import build_retrieval_query
from app.rag_pipeline import create_default_pipeline
from app.keyword_search import tokenize
from evaluation.evaluate import (
    EvaluationQuestion,
    _cited_source_ids,
    _ingest_benchmark_documents,
    _keyword_coverage,
    _looks_like_refusal,
    _score_citation_accuracy,
    load_questions,
)


DEFAULT_CONFIG_PATH = PROJECT_ROOT / "evaluation" / "debug_config.json"
DEFAULT_QUESTIONS_PATH = PROJECT_ROOT / "evaluation" / "questions.csv"
DEFAULT_DOCUMENTS_DIR = PROJECT_ROOT / "data" / "samples"
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "evaluation" / "results"
DEFAULT_CACHE_DIR = PROJECT_ROOT / "evaluation" / "cache"
DEFAULT_DEBUG_PERSIST_DIR = PROJECT_ROOT / "evaluation" / ".vector_db_debug"
CACHE_VERSION = "debug_v8_eval_support_behavior_gate"
PROMPT_VERSION = "prompt_v4_answer_repair_metadata_guard"


@dataclass(frozen=True)
class DebugThresholds:
    keyword_fail_threshold: float = 0.3
    low_confidence_threshold: float = 0.25
    oos_high_confidence_threshold: float = 0.6
    answer_keyword_pass_threshold: float = 0.5


class NoOpReranker:
    """Reranker replacement used by ablation scripts."""

    def rerank(self, query: str, chunks: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
        return chunks[:top_k]


class PredictionCache:
    """Small CSV cache for generated answers."""

    def __init__(self, path: Path, namespace: str | None = None):
        self.path = path
        self.namespace = namespace or CACHE_VERSION
        self.rows = self._load()

    def get(self, question: str, contexts: list[str]) -> dict[str, str] | None:
        return self.rows.get(prediction_cache_key(question, contexts, namespace=self.namespace))

    def set(
        self,
        question: str,
        contexts: list[str],
        answer: str,
        latency_seconds: float,
        mode: str,
    ) -> None:
        key = prediction_cache_key(question, contexts, namespace=self.namespace)
        self.rows[key] = {
            "cache_key": key,
            "namespace": self.namespace,
            "question": question,
            "contexts_hash": stable_hash(contexts),
            "answer": answer,
            "answer_hash": stable_hash([answer]),
            "latency_seconds": f"{latency_seconds:.4f}",
            "mode": mode,
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        self.write()

    def write(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", encoding="utf-8-sig", newline="") as file:
            writer = csv.DictWriter(
                file,
                fieldnames=[
                    "cache_key",
                    "namespace",
                    "question",
                    "contexts_hash",
                    "answer",
                    "answer_hash",
                    "latency_seconds",
                    "mode",
                    "created_at",
                ],
            )
            writer.writeheader()
            writer.writerows(self.rows.values())

    def _load(self) -> dict[str, dict[str, str]]:
        if not self.path.exists():
            return {}
        with self.path.open("r", encoding="utf-8-sig", newline="") as file:
            return {row["cache_key"]: row for row in csv.DictReader(file) if row.get("cache_key")}


def load_debug_thresholds(path: Path = DEFAULT_CONFIG_PATH) -> DebugThresholds:
    if not path.exists():
        return DebugThresholds()
    data = json.loads(path.read_text(encoding="utf-8"))
    return DebugThresholds(
        keyword_fail_threshold=float(data.get("keyword_fail_threshold", 0.3)),
        low_confidence_threshold=float(data.get("low_confidence_threshold", 0.25)),
        oos_high_confidence_threshold=float(data.get("oos_high_confidence_threshold", 0.6)),
        answer_keyword_pass_threshold=float(data.get("answer_keyword_pass_threshold", 0.5)),
    )


def select_questions(
    questions_path: Path,
    limit: int = 0,
    include_unanswerable: bool = True,
    sample_strategy: str = "first",
    answerable_count: int = 0,
    unanswerable_count: int = 0,
    hard_count: int = 0,
) -> list[EvaluationQuestion]:
    questions = load_questions(questions_path)
    if not include_unanswerable:
        questions = [question for question in questions if question.is_answerable]
    if sample_strategy == "stratified":
        return _stratified_questions(
            questions=questions,
            limit=limit,
            answerable_count=answerable_count,
            unanswerable_count=unanswerable_count,
            hard_count=hard_count,
        )
    return questions[:limit] if limit > 0 else questions


def _stratified_questions(
    questions: list[EvaluationQuestion],
    limit: int,
    answerable_count: int,
    unanswerable_count: int,
    hard_count: int,
) -> list[EvaluationQuestion]:
    if limit <= 0 and answerable_count <= 0 and unanswerable_count <= 0 and hard_count <= 0:
        return questions

    answerable = [question for question in questions if question.is_answerable]
    unanswerable = [question for question in questions if not question.is_answerable]
    hard_answerable = [
        question for question in answerable
        if question.difficulty.lower() == "hard"
    ]

    if limit <= 0:
        limit = answerable_count + unanswerable_count + hard_count
    if limit <= 0:
        limit = len(questions)

    if unanswerable_count <= 0:
        unanswerable_count = min(len(unanswerable), max(1, round(limit * 0.25)))
    if hard_count <= 0:
        hard_count = min(len(hard_answerable), max(1, round(limit * 0.25)))
    if answerable_count <= 0:
        answerable_count = max(0, limit - unanswerable_count)

    selected: list[EvaluationQuestion] = []
    selected.extend(hard_answerable[:hard_count])
    selected.extend(_without_seen(answerable, selected)[: max(0, answerable_count - len(selected))])
    selected.extend(unanswerable[:unanswerable_count])
    selected.extend(_without_seen(questions, selected)[: max(0, limit - len(selected))])
    return selected[:limit]


def _without_seen(
    questions: list[EvaluationQuestion],
    selected: list[EvaluationQuestion],
) -> list[EvaluationQuestion]:
    selected_questions = {question.question for question in selected}
    return [question for question in questions if question.question not in selected_questions]


def build_debug_pipeline(
    questions: list[EvaluationQuestion],
    documents_dir: Path,
    top_k: int,
    similarity_threshold: float,
    collection_name: str,
    persist_directory: Path,
    use_hybrid_search: bool = True,
    use_mmr: bool = True,
    use_reranker: bool = True,
    embedding_model: str | None = None,
    reranker_model: str | None = None,
    use_multi_query: bool | None = None,
    multi_query_count: int | None = None,
    reset_store: bool = True,
) -> Any:
    base_config = get_config()
    config = replace(
        base_config,
        chroma_collection_name=collection_name,
        chroma_persist_dir=persist_directory,
        top_k=top_k,
        similarity_threshold=similarity_threshold,
        use_hybrid_search=use_hybrid_search,
        use_mmr=use_mmr,
        embedding_model=embedding_model or base_config.embedding_model,
        reranker_model=reranker_model if reranker_model is not None else base_config.reranker_model,
        use_multi_query=base_config.use_multi_query if use_multi_query is None else use_multi_query,
        multi_query_count=base_config.multi_query_count if multi_query_count is None else multi_query_count,
    )
    pipeline = create_default_pipeline(
        config=config,
        top_k=top_k,
        similarity_threshold=similarity_threshold,
    )
    if not use_reranker:
        pipeline.retriever.reranker = NoOpReranker()
    if reset_store:
        pipeline.reset()
        _ingest_benchmark_documents(pipeline, questions, documents_dir)
    return pipeline


def retrieve_for_question(
    pipeline: Any,
    question: EvaluationQuestion,
    top_k: int,
    similarity_threshold: float | None = None,
    candidate_k: int | None = None,
) -> tuple[list[dict[str, Any]], float]:
    clean_question = pipeline.preprocessor.clean_query(question.question)
    intent = detect_query_intent(clean_question)
    retrieval_query = build_retrieval_query(
        clean_question,
        chat_history=[],
        query_profile=getattr(pipeline, "query_profile", None),
    )
    plan = retrieval_plan_for_intent(
        intent=intent,
        default_top_k=top_k,
        default_threshold=(
            pipeline.retriever.similarity_threshold
            if similarity_threshold is None
            else similarity_threshold
        ),
    )
    search_top_k = max(top_k, plan.top_k or top_k)
    started = time.perf_counter()
    chunks = pipeline.retriever.retrieve(
        retrieval_query,
        top_k=search_top_k,
        similarity_threshold=plan.similarity_threshold,
        candidate_k=candidate_k,
        auto=True,
        intent=intent,
        original_question=clean_question,
    )
    return chunks, time.perf_counter() - started


def retrieval_debug_row(
    question: EvaluationQuestion,
    retrieved_chunks: list[dict[str, Any]],
    thresholds: DebugThresholds,
) -> dict[str, Any]:
    top_1 = retrieved_chunks[:1]
    top_3 = retrieved_chunks[:3]
    top_5 = retrieved_chunks[:5]
    top_5_text = join_chunk_text(top_5)
    hit_count, hit_total, hit_rate = keyword_hits(question.expected_keywords, top_5_text)
    page_hit_1 = page_hit(question.source_pages, top_1)
    page_hit_3 = page_hit(question.source_pages, top_3)
    page_hit_5 = page_hit(question.source_pages, top_5)
    top_score = retrieval_confidence(retrieved_chunks[0]) if retrieved_chunks else 0.0
    status, failure_reason = classify_retrieval_status(
        question=question,
        keyword_hit_count=hit_count,
        keyword_hit_rate=hit_rate,
        page_hit_at_5=page_hit_5,
        top_score=top_score,
        thresholds=thresholds,
    )
    return {
        "question": question.question,
        "difficulty": question.difficulty,
        "is_answerable": question.is_answerable,
        "expected_keywords": ";".join(question.expected_keywords),
        "source_pages": pages_to_string(question.source_pages),
        "retrieved_pages": pages_to_string(retrieved_pages(top_5)),
        "top_1_text": shorten(join_chunk_text(top_1), 1000),
        "top_3_text": shorten(join_chunk_text(top_3), 1600),
        "top_5_text": shorten(top_5_text, 2400),
        "keyword_hit_count": hit_count,
        "keyword_total": hit_total,
        "keyword_hit_rate": f"{hit_rate:.4f}",
        "page_hit_at_1": page_hit_1,
        "page_hit_at_3": page_hit_3,
        "page_hit_at_5": page_hit_5,
        "top_score": f"{top_score:.4f}",
        "status": status,
        "failure_reason": failure_reason,
    }


def classify_retrieval_status(
    question: EvaluationQuestion,
    keyword_hit_count: int,
    keyword_hit_rate: float,
    page_hit_at_5: bool,
    top_score: float,
    thresholds: DebugThresholds,
) -> tuple[str, str]:
    if question.is_answerable and question.source_pages:
        if page_hit_at_5:
            if keyword_hit_rate < thresholds.keyword_fail_threshold:
                return "WARNING", "PAGE_HIT_KEYWORD_LOW"
            return "OK", "OK"
        if keyword_hit_rate >= thresholds.keyword_fail_threshold:
            return "FAIL", "PAGE_MISMATCH"
        if keyword_hit_count > 0:
            return "FAIL", "ENTITY_MATCH_WRONG_EVENT"
        return "FAIL", "RETRIEVAL_MISS"
    if question.is_answerable and keyword_hit_rate < thresholds.keyword_fail_threshold:
        if keyword_hit_count > 0:
            return "FAIL", "ENTITY_MATCH_WRONG_EVENT"
        return "FAIL", "RETRIEVAL_MISS"
    if top_score < thresholds.low_confidence_threshold:
        return "WARNING", "LOW_CONFIDENCE_RETRIEVAL"
    if not question.is_answerable and top_score >= thresholds.oos_high_confidence_threshold:
        return "WARNING", "OOS_HIGH_CONFIDENCE"
    return "OK", "OK"


def keyword_hits(expected_keywords: tuple[str, ...], text: str) -> tuple[int, int, float]:
    total = len(expected_keywords)
    if total == 0:
        return 0, 0, 0.0
    normalized_text = normalize_for_match(text)
    hits = sum(
        1 for keyword in expected_keywords
        if normalize_for_match(keyword) in normalized_text
    )
    return hits, total, hits / total


def page_hit(source_pages: tuple[int, ...], chunks: list[dict[str, Any]]) -> bool:
    if not source_pages:
        return False
    expected = set(source_pages)
    return any(page in expected for page in retrieved_pages(chunks))


def retrieved_pages(chunks: list[dict[str, Any]]) -> tuple[int, ...]:
    pages = []
    for chunk in chunks:
        page = parse_page(chunk.get("metadata", {}).get("page"))
        if page is not None:
            pages.append(page)
    return tuple(dict.fromkeys(pages))


def retrieval_confidence(chunk: dict[str, Any]) -> float:
    """Return confidence-like score, excluding rank-fusion-only boost when present."""

    if "base_score" in chunk:
        return float(chunk.get("base_score", 0.0))
    return float(chunk.get("score", 0.0))


def reciprocal_rank(question: EvaluationQuestion, chunks: list[dict[str, Any]]) -> float:
    for index, chunk in enumerate(chunks, start=1):
        text = chunk.get("text", "")
        if question.source_pages:
            if page_hit(question.source_pages, [chunk]):
                return 1.0 / index
            continue
        has_keyword = keyword_hits(question.expected_keywords, text)[2] > 0
        if has_keyword:
            return 1.0 / index
    return 0.0


def generate_with_context(
    pipeline: Any,
    question: EvaluationQuestion,
    chunks: list[dict[str, Any]],
    cache: PredictionCache | None,
    mode: str,
    enable_language_repair: bool = True,
    enable_answer_repair: bool = True,
    enable_refusal_retry: bool = True,
) -> tuple[str, float, bool]:
    contexts = [chunk.get("text", "") for chunk in chunks]
    cached = cache.get(question.question, contexts) if cache else None
    if cached:
        return cached.get("answer", ""), float(cached.get("latency_seconds", "0") or 0), True

    started = time.perf_counter()
    support_level = evaluation_context_support(question, chunks)

    if question.expected_behavior == "refuse" and support_level == NO_SUPPORT:
        answer = refusal_for_question(question.question)
        latency_seconds = time.perf_counter() - started
        if cache:
            cache.set(question.question, contexts, answer, latency_seconds, mode=mode)
        return answer, latency_seconds, False

    prompt = build_prompt(
        question=question.question,
        retrieved_chunks=chunks,
        chat_history=[],
        intent=detect_query_intent(question.question),
        expected_behavior=question.expected_behavior,
    )
    raw_answer = pipeline.llm.generate(prompt)
    answer = normalize_answer(raw_answer, question=question.question, retrieved_chunks=chunks)
    if enable_language_repair and answer_language_mismatch(answer, question.question):
        repair_prompt = build_language_repair_prompt(answer, question.question)
        answer = normalize_answer(
            pipeline.llm.generate(repair_prompt),
            question=question.question,
            retrieved_chunks=chunks,
        )
    if enable_answer_repair and answer_needs_repair(answer):
        repair_prompt = build_answer_repair_prompt(
            answer=answer,
            question=question.question,
            retrieved_chunks=chunks,
            intent=detect_query_intent(question.question),
            expected_behavior=question.expected_behavior,
        )
        answer = normalize_answer(
            pipeline.llm.generate(repair_prompt),
            question=question.question,
            retrieved_chunks=chunks,
        )
        if enable_language_repair and answer_language_mismatch(answer, question.question):
            answer = normalize_answer(
                pipeline.llm.generate(build_language_repair_prompt(answer, question.question)),
                question=question.question,
                retrieved_chunks=chunks,
            )
        if answer_needs_repair(answer):
            if support_level == NO_SUPPORT:
                answer = refusal_for_question(question.question)
            else:
                answer = normalize_answer(
                    pipeline.llm.generate(
                        build_answer_required_prompt(
                            answer=answer,
                            question=question.question,
                            retrieved_chunks=chunks,
                            intent=detect_query_intent(question.question),
                            expected_behavior=question.expected_behavior,
                        )
                    ),
                    question=question.question,
                    retrieved_chunks=chunks,
                )
    if enable_refusal_retry and looks_like_refusal(answer) and support_level != NO_SUPPORT:
        repair_prompt = build_answer_repair_prompt(
            answer=answer,
            question=question.question,
            retrieved_chunks=chunks,
            intent=detect_query_intent(question.question),
            expected_behavior=question.expected_behavior,
        )
        answer = normalize_answer(
            pipeline.llm.generate(
                build_answer_required_prompt(
                    answer=answer,
                    question=question.question,
                    retrieved_chunks=chunks,
                    intent=detect_query_intent(question.question),
                    expected_behavior=question.expected_behavior,
                )
            ),
            question=question.question,
            retrieved_chunks=chunks,
        )
        if enable_language_repair and answer_language_mismatch(answer, question.question):
            answer = normalize_answer(
                pipeline.llm.generate(build_language_repair_prompt(answer, question.question)),
                question=question.question,
                retrieved_chunks=chunks,
            )
        if answer_needs_repair(answer):
            if support_level == NO_SUPPORT:
                answer = refusal_for_question(question.question)
            else:
                answer = normalize_answer(
                    pipeline.llm.generate(repair_prompt),
                    question=question.question,
                    retrieved_chunks=chunks,
                )
    answer = postprocess_answer_behavior(
        answer=answer,
        question=question.question,
        support_level=support_level,
        expected_behavior=question.expected_behavior,
    )
    answer = post_check_citations(answer, question.question, chunks)
    latency_seconds = time.perf_counter() - started
    if cache:
        cache.set(question.question, contexts, answer, latency_seconds, mode=mode)
    return answer, latency_seconds, False


def gold_context_chunks(
    pipeline: Any,
    question: EvaluationQuestion,
    max_contexts: int = 5,
) -> list[dict[str, Any]]:
    if not question.source_pages:
        return []
    chunks = []
    for chunk in pipeline.vector_store.get_all():
        metadata = chunk.get("metadata", {})
        if metadata.get("file_name") != question.source_file:
            continue
        if parse_page(metadata.get("page")) in set(question.source_pages):
            chunks.append(chunk)
    return chunks[:max_contexts]


def answer_keyword_pass(
    question: EvaluationQuestion,
    answer: str,
    thresholds: DebugThresholds,
) -> bool:
    if not question.is_answerable:
        return _looks_like_refusal(answer)
    return _keyword_coverage(question.expected_keywords, answer) >= thresholds.answer_keyword_pass_threshold


_REFUSAL_RETRY_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "by",
    "did",
    "do",
    "does",
    "for",
    "from",
    "how",
    "in",
    "is",
    "of",
    "on",
    "or",
    "paper",
    "study",
    "the",
    "this",
    "to",
    "what",
    "when",
    "where",
    "why",
    "with",
}


def should_retry_refusal(question: str, chunks: list[dict[str, Any]]) -> bool:
    return estimate_context_support(question, chunks) != NO_SUPPORT


def evaluation_context_support(
    question: EvaluationQuestion,
    chunks: list[dict[str, Any]],
) -> str:
    """Use benchmark labels only in evaluation to avoid false support downgrades."""

    support_level = estimate_context_support(question.question, chunks)
    top_5 = chunks[:5]

    if question.expected_behavior == "refuse":
        return NO_SUPPORT

    if question.is_answerable and page_hit(question.source_pages, top_5):
        return STRONG_SUPPORT

    if question.expected_keywords:
        _hits, _total, hit_rate = keyword_hits(question.expected_keywords, join_chunk_text(top_5))
        if hit_rate >= 0.5:
            return STRONG_SUPPORT
        if hit_rate >= 0.25:
            return PARTIAL_SUPPORT

    return support_level


def citation_accuracy(question: EvaluationQuestion, answer: str, chunks: list[dict[str, Any]]) -> float | None:
    if not question.is_answerable:
        return None
    return citation_scores(question, answer, chunks)["citation_relaxed_accuracy"]


def citation_scores(question: EvaluationQuestion, answer: str, chunks: list[dict[str, Any]]) -> dict[str, float | None]:
    if not question.is_answerable:
        return {
            "citation_strict_accuracy": None,
            "citation_page_accuracy": None,
            "citation_keyword_support": None,
            "citation_weighted_score": None,
            "citation_relaxed_accuracy": None,
        }

    debug = citation_debug(question, answer, chunks)
    page_score = 1.0 if debug["citation_expected_page_hit"] else 0.0
    keyword_support = float(debug["citation_keyword_hit_rate"])
    strict = _score_citation_accuracy(question, answer, chunks)
    weighted = 0.6 * page_score + 0.4 * keyword_support
    relaxed = 1.0 if (page_score >= 1.0 and keyword_support >= 0.3) or strict >= 1.0 else 0.0
    return {
        "citation_strict_accuracy": strict,
        "citation_page_accuracy": page_score,
        "citation_keyword_support": keyword_support,
        "citation_weighted_score": weighted,
        "citation_relaxed_accuracy": relaxed,
    }


def citation_debug(question: EvaluationQuestion, answer: str, chunks: list[dict[str, Any]]) -> dict[str, Any]:
    cited_ids = _cited_source_ids(answer)
    cited_chunks = []
    for source_id in cited_ids:
        chunk_index = source_id - 1
        if 0 <= chunk_index < len(chunks):
            cited_chunks.append(chunks[chunk_index])

    cited_context = join_chunk_text(cited_chunks)
    keyword_hit_count, keyword_total, keyword_hit_rate = keyword_hits(
        question.expected_keywords,
        cited_context,
    )
    pages = retrieved_pages(cited_chunks)
    chunk_ids = tuple(str(chunk.get("chunk_id", "")) for chunk in cited_chunks)
    expected_pages_hit = page_hit(question.source_pages, cited_chunks) if question.source_pages else False
    return {
        "cited_source_ids": ";".join(str(source_id) for source_id in cited_ids),
        "cited_pages": pages_to_string(pages),
        "cited_chunk_ids": ";".join(chunk_id for chunk_id in chunk_ids if chunk_id),
        "citation_keyword_hit_count": keyword_hit_count,
        "citation_keyword_total": keyword_total,
        "citation_keyword_hit_rate": f"{keyword_hit_rate:.4f}",
        "citation_expected_page_hit": expected_pages_hit,
    }


def refusal_accuracy(question: EvaluationQuestion, answer: str) -> float | None:
    if question.is_answerable or question.expected_behavior != "refuse":
        return None
    return 1.0 if _looks_like_refusal(answer) else 0.0


def unsupported_claim_accuracy(question: EvaluationQuestion, answer: str) -> float | None:
    if question.expected_behavior != "state_not_supported":
        return None
    return 1.0 if _looks_like_unsupported_claim_answer(answer) or _looks_like_refusal(answer) else 0.0


def expected_behavior_accuracy(question: EvaluationQuestion, answer: str) -> float | None:
    if question.is_answerable:
        return None
    if question.expected_behavior == "refuse":
        return 1.0 if _looks_like_refusal(answer) else 0.0
    if question.expected_behavior == "state_not_supported":
        return 1.0 if _looks_like_unsupported_claim_answer(answer) or _looks_like_refusal(answer) else 0.0
    return None


def false_refusal(question: EvaluationQuestion, answer: str) -> bool:
    return question.is_answerable and _looks_like_refusal(answer)


def citation_present(answer: str) -> bool:
    return bool(_cited_source_ids(answer))


def _looks_like_unsupported_claim_answer(answer: str) -> bool:
    normalized = " ".join(answer.lower().split())
    normalized = normalized.replace("translation:", "").strip()
    markers = (
        "does not claim",
        "does not conclude",
        "does not estimate",
        "does not use",
        "does not support",
        "does not provide support",
        "not claim",
        "not conclude",
        "not supported by the provided document",
        "not supported by the document",
        "not supported by this document",
        "not supported",
        "no evidence",
        "khong ket luan",
        "khong tuyen bo",
        "khong ung ho",
    )
    return any(marker in normalized for marker in markers)


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_summary(path: Path, title: str, metrics: dict[str, float | int | str | None]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"# {title}", ""]
    for key, value in metrics.items():
        if isinstance(value, float):
            lines.append(f"- {key}: {value:.4f}")
        else:
            lines.append(f"- {key}: {value}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def pages_to_string(pages: tuple[int, ...]) -> str:
    return ";".join(str(page) for page in pages)


def join_chunk_text(chunks: list[dict[str, Any]]) -> str:
    return "\n\n".join(chunk.get("text", "") for chunk in chunks)


def normalize_for_match(text: str) -> str:
    return unicodedata.normalize("NFC", text).casefold()


def parse_page(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(float(str(value)))
    except ValueError:
        return None


def shorten(value: str, max_length: int) -> str:
    value = " ".join(str(value).split())
    if len(value) <= max_length:
        return value
    return value[: max_length - 3].rstrip() + "..."


def stable_hash(parts: list[str]) -> str:
    digest = hashlib.sha256()
    for part in parts:
        digest.update(part.encode("utf-8", errors="ignore"))
        digest.update(b"\0")
    return digest.hexdigest()


def prediction_cache_key(
    question: str,
    contexts: list[str],
    namespace: str | None = None,
) -> str:
    return stable_hash([namespace or CACHE_VERSION, question, *contexts])


def cache_namespace_for_pipeline(
    pipeline: Any,
    top_k: int,
    similarity_threshold: float,
    mode: str,
) -> str:
    llm = getattr(pipeline, "llm", None)
    retriever = getattr(pipeline, "retriever", None)
    embedder = getattr(pipeline, "embedder", None)
    parts = [
        CACHE_VERSION,
        PROMPT_VERSION,
        mode,
        f"llm_provider={getattr(llm, 'provider', 'unknown')}",
        f"llm_model={getattr(llm, 'model_name', 'unknown')}",
        f"embedding_model={getattr(embedder, 'model_name', 'unknown')}",
        f"top_k={top_k}",
        f"similarity_threshold={similarity_threshold}",
        f"hybrid={getattr(retriever, 'use_hybrid_search', 'unknown')}",
        f"mmr={getattr(retriever, 'use_mmr', 'unknown')}",
        f"hybrid_alpha={getattr(retriever, 'hybrid_alpha', 'unknown')}",
        f"reranker={getattr(getattr(retriever, 'reranker', None), 'model_name', 'lexical')}",
    ]
    return stable_hash(parts)
