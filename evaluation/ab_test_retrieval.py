"""A/B test retrieval backends without hardcoding benchmark questions."""

from __future__ import annotations

import argparse
import time
import sys
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config import get_config
from evaluation.debug_utils import (
    DEFAULT_CONFIG_PATH,
    DEFAULT_DEBUG_PERSIST_DIR,
    DEFAULT_DOCUMENTS_DIR,
    DEFAULT_QUESTIONS_PATH,
    DEFAULT_RESULTS_DIR,
    build_debug_pipeline,
    keyword_hits,
    load_debug_thresholds,
    mean,
    page_hit,
    reciprocal_rank,
    retrieve_for_question,
    select_questions,
    write_csv,
)


CURRENT_MODEL = "__current__"
BGE_M3_MODEL = "BAAI/bge-m3"
BGE_RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"


@dataclass(frozen=True)
class RetrievalVariant:
    name: str
    embedding_model: str
    reranker_model: str | None
    use_reranker: bool = True
    use_multi_query: bool | None = None


FIELDNAMES = [
    "variant",
    "embedding_model",
    "reranker_model",
    "use_multi_query",
    "questions",
    "page_recall_at_5",
    "evidence_hit_at_5",
    "mrr",
    "citation_page_accuracy",
    "keyword_hit_at_5",
    "keyword_recall_at_5",
    "average_retrieval_latency_seconds",
    "total_latency_seconds",
    "status",
    "error",
]


VARIANTS = [
    RetrievalVariant("current_env", CURRENT_MODEL, CURRENT_MODEL, use_reranker=True, use_multi_query=None),
    RetrievalVariant("current_no_multi_query", CURRENT_MODEL, CURRENT_MODEL, use_reranker=True, use_multi_query=False),
    RetrievalVariant("current_no_reranker", CURRENT_MODEL, None, use_reranker=True, use_multi_query=None),
    RetrievalVariant("bge_m3_embedding", BGE_M3_MODEL, None, use_reranker=True, use_multi_query=False),
    RetrievalVariant("bge_reranker", CURRENT_MODEL, BGE_RERANKER_MODEL, use_reranker=True, use_multi_query=False),
    RetrievalVariant("full_bge", BGE_M3_MODEL, BGE_RERANKER_MODEL, use_reranker=True, use_multi_query=True),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run retrieval backend A/B tests.")
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS_PATH)
    parser.add_argument("--documents-dir", type=Path, default=DEFAULT_DOCUMENTS_DIR)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_RESULTS_DIR / "retrieval_ablation_report.csv")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--sample-strategy", choices=["first", "stratified"], default="first")
    parser.add_argument("--answerable-count", type=int, default=0)
    parser.add_argument("--unanswerable-count", type=int, default=0)
    parser.add_argument("--hard-count", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--similarity-threshold", type=float, default=0.3)
    parser.add_argument("--collection-name", default="rag_retrieval_abtest")
    parser.add_argument("--persist-dir", type=Path, default=DEFAULT_DEBUG_PERSIST_DIR)
    parser.add_argument("--multi-query-count", type=int, default=4)
    parser.add_argument("--reuse-index", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    questions = select_questions(
        args.questions,
        limit=args.limit,
        sample_strategy=args.sample_strategy,
        answerable_count=args.answerable_count,
        unanswerable_count=args.unanswerable_count,
        hard_count=args.hard_count,
    )
    thresholds = load_debug_thresholds(args.config)
    app_config = get_config()
    print(f"Preparing retrieval A/B test: questions={len(questions)}")

    rows = []
    for variant in VARIANTS:
        print(f"Running variant {variant.name}")
        started = time.perf_counter()
        try:
            row = run_variant(args, thresholds, questions, app_config, variant)
            row["status"] = "success"
            row["error"] = ""
        except Exception as exc:
            row = error_row(args, app_config, variant, exc)
        row["total_latency_seconds"] = f"{time.perf_counter() - started:.4f}"
        rows.append(row)

    write_csv(args.output, rows, FIELDNAMES)
    print(f"Wrote retrieval A/B report to {args.output}")


def run_variant(args, thresholds, questions, app_config, variant: RetrievalVariant) -> dict[str, object]:
    embedding_model = (
        app_config.embedding_model if variant.embedding_model == CURRENT_MODEL else variant.embedding_model
    )
    reranker_model = (
        app_config.reranker_model
        if variant.reranker_model == CURRENT_MODEL
        else variant.reranker_model
    )
    collection_name = f"{args.collection_name}_{_safe_variant_name(variant.name)}"
    pipeline = build_debug_pipeline(
        questions=questions,
        documents_dir=args.documents_dir,
        top_k=args.top_k,
        similarity_threshold=args.similarity_threshold,
        collection_name=collection_name,
        persist_directory=args.persist_dir,
        use_hybrid_search=True,
        use_mmr=True,
        use_reranker=variant.use_reranker,
        embedding_model=embedding_model,
        reranker_model=reranker_model,
        use_multi_query=variant.use_multi_query,
        multi_query_count=args.multi_query_count,
        reset_store=not args.reuse_index,
    )

    keyword_rates = []
    keyword_recalls = []
    page_hits = []
    evidence_hits = []
    reciprocal_ranks = []
    citation_page_hits = []
    latencies = []

    for question in questions:
        chunks, latency = retrieve_for_question(
            pipeline,
            question,
            top_k=args.top_k,
            similarity_threshold=args.similarity_threshold,
        )
        top_5 = chunks[:5]
        if question.is_answerable:
            hit_rate = keyword_hits(
                question.expected_keywords,
                "\n\n".join(chunk.get("text", "") for chunk in top_5),
            )[2]
            has_page = page_hit(question.source_pages, top_5)
            keyword_rates.append(hit_rate)
            keyword_recall = hit_rate >= thresholds.keyword_fail_threshold
            keyword_recalls.append(1.0 if keyword_recall else 0.0)
            reciprocal_ranks.append(reciprocal_rank(question, top_5))
            evidence_hits.append(1.0 if (has_page or keyword_recall) else 0.0)
            if question.source_pages:
                page_hits.append(1.0 if has_page else 0.0)
                citation_page_hits.append(1.0 if _top_source_page_hit(question.source_pages, top_5) else 0.0)
        latencies.append(latency)

    return {
        "variant": variant.name,
        "embedding_model": embedding_model,
        "reranker_model": reranker_model or "",
        "use_multi_query": (
            app_config.use_multi_query if variant.use_multi_query is None else variant.use_multi_query
        ),
        "questions": len(questions),
        "page_recall_at_5": f"{mean(page_hits):.4f}",
        "evidence_hit_at_5": f"{mean(evidence_hits):.4f}",
        "mrr": f"{mean(reciprocal_ranks):.4f}",
        "citation_page_accuracy": f"{mean(citation_page_hits):.4f}",
        "keyword_hit_at_5": f"{mean(keyword_rates):.4f}",
        "keyword_recall_at_5": f"{mean(keyword_recalls):.4f}",
        "average_retrieval_latency_seconds": f"{mean(latencies):.4f}",
    }


def error_row(args, app_config, variant: RetrievalVariant, exc: Exception) -> dict[str, object]:
    embedding_model = (
        app_config.embedding_model if variant.embedding_model == CURRENT_MODEL else variant.embedding_model
    )
    return {
        "variant": variant.name,
        "embedding_model": embedding_model,
        "reranker_model": variant.reranker_model or "",
        "use_multi_query": (
            app_config.use_multi_query if variant.use_multi_query is None else variant.use_multi_query
        ),
        "questions": 0,
        "page_recall_at_5": "",
        "evidence_hit_at_5": "",
        "mrr": "",
        "citation_page_accuracy": "",
        "keyword_hit_at_5": "",
        "keyword_recall_at_5": "",
        "average_retrieval_latency_seconds": "",
        "total_latency_seconds": "",
        "status": "error",
        "error": f"{type(exc).__name__}: {exc}",
    }


def _top_source_page_hit(source_pages: tuple[int, ...], chunks: list[dict[str, object]]) -> bool:
    if not source_pages or not chunks:
        return False
    page = chunks[0].get("metadata", {}).get("page")
    return page in set(source_pages)


def _safe_variant_name(name: str) -> str:
    return "".join(character if character.isalnum() else "_" for character in name.lower())


if __name__ == "__main__":
    main()
