"""Custom deterministic evaluation before running slow RAGAS scoring."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.prompt_template import refusal_for_question
from evaluation.debug_utils import (
    DEFAULT_CACHE_DIR,
    DEFAULT_CONFIG_PATH,
    DEFAULT_DEBUG_PERSIST_DIR,
    DEFAULT_DOCUMENTS_DIR,
    DEFAULT_QUESTIONS_PATH,
    DEFAULT_RESULTS_DIR,
    PredictionCache,
    build_debug_pipeline,
    cache_namespace_for_pipeline,
    citation_accuracy,
    citation_debug,
    citation_present,
    citation_scores,
    expected_behavior_accuracy,
    false_refusal,
    generate_with_context,
    keyword_hits,
    load_debug_thresholds,
    mean,
    page_hit,
    pages_to_string,
    reciprocal_rank,
    refusal_accuracy,
    retrieve_for_question,
    retrieved_pages,
    select_questions,
    unsupported_claim_accuracy,
    write_csv,
    write_summary,
)


FIELDNAMES = [
    "question",
    "difficulty",
    "question_type",
    "expected_behavior",
    "is_answerable",
    "answer",
    "status",
    "source_pages",
    "keyword_hit_rate_at_5",
    "keyword_recall_at_5",
    "page_hit_at_5",
    "page_recall_at_5",
    "evidence_hit_at_5",
    "mrr",
    "citation_accuracy",
    "citation_strict_accuracy",
    "citation_page_accuracy",
    "citation_keyword_support",
    "citation_weighted_score",
    "citation_present",
    "cited_source_ids",
    "cited_pages",
    "cited_chunk_ids",
    "citation_keyword_hit_count",
    "citation_keyword_total",
    "citation_keyword_hit_rate",
    "citation_expected_page_hit",
    "refusal_accuracy",
    "unsupported_claim_accuracy",
    "expected_behavior_accuracy",
    "false_refusal",
    "retrieved_pages",
    "top_score",
    "retrieval_latency_seconds",
    "generation_latency_seconds",
    "latency_seconds",
    "cache_hit",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run custom deterministic RAG evaluation.")
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS_PATH)
    parser.add_argument("--documents-dir", type=Path, default=DEFAULT_DOCUMENTS_DIR)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_RESULTS_DIR / "custom_eval_results.csv")
    parser.add_argument("--summary", type=Path, default=DEFAULT_RESULTS_DIR / "custom_eval_summary.md")
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE_DIR / "predictions.csv")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--sample-strategy", choices=["first", "stratified"], default="first")
    parser.add_argument("--answerable-count", type=int, default=0)
    parser.add_argument("--unanswerable-count", type=int, default=0)
    parser.add_argument("--hard-count", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--similarity-threshold", type=float, default=0.3)
    parser.add_argument("--collection-name", default="rag_debug_custom_eval")
    parser.add_argument("--persist-dir", type=Path, default=DEFAULT_DEBUG_PERSIST_DIR)
    parser.add_argument("--reuse-index", action="store_true")
    parser.add_argument("--refresh-cache", action="store_true")
    parser.add_argument(
        "--generation-top-k",
        type=int,
        default=5,
        help="Number of retrieved chunks passed to the LLM after retrieval metrics are computed.",
    )
    parser.add_argument(
        "--skip-repairs",
        action="store_true",
        help="Disable language/answer/refusal repair passes for faster base-generation debugging.",
    )
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
    print(f"Preparing custom eval: questions={len(questions)}")
    pipeline = build_debug_pipeline(
        questions=questions,
        documents_dir=args.documents_dir,
        top_k=args.top_k,
        similarity_threshold=args.similarity_threshold,
        collection_name=args.collection_name,
        persist_directory=args.persist_dir,
        reset_store=not args.reuse_index,
    )
    if args.refresh_cache and args.cache.exists():
        args.cache.unlink()
    cache = PredictionCache(
        args.cache,
        namespace=cache_namespace_for_pipeline(
            pipeline,
            top_k=args.generation_top_k,
            similarity_threshold=args.similarity_threshold,
            mode=f"custom_eval_repair={not args.skip_repairs}",
        ),
    )

    rows = []
    for index, question in enumerate(questions, start=1):
        print(f"[{index}/{len(questions)}] Evaluating: {question.question}")
        chunks, retrieval_latency = retrieve_for_question(
            pipeline,
            question,
            top_k=args.top_k,
            similarity_threshold=args.similarity_threshold,
        )
        generation_chunks = chunks[: args.generation_top_k]
        if generation_chunks:
            answer, generation_latency, cache_hit = generate_with_context(
                pipeline,
                question,
                generation_chunks,
                cache=cache,
                mode="retrieved_context",
                enable_language_repair=not args.skip_repairs,
                enable_answer_repair=not args.skip_repairs,
                enable_refusal_retry=not args.skip_repairs,
            )
            status = "success"
        else:
            answer = refusal_for_question(question.question)
            generation_latency = 0.0
            cache_hit = False
            status = "no_context"

        top_5 = chunks[:5]
        keyword_rate = keyword_hits(
            question.expected_keywords,
            "\n\n".join(chunk.get("text", "") for chunk in top_5),
        )[2]
        page_hit_at_5 = page_hit(question.source_pages, top_5)
        keyword_recall_at_5 = keyword_rate >= thresholds.keyword_fail_threshold
        page_recall_at_5 = page_hit_at_5 if question.source_pages else False
        evidence_hit_at_5 = page_recall_at_5 or keyword_recall_at_5
        top_score = float(chunks[0].get("score", 0.0)) if chunks else 0.0
        citation_info = citation_debug(question, answer, generation_chunks)
        citation_score_info = citation_scores(question, answer, generation_chunks)
        rows.append(
            {
                "question": question.question,
                "difficulty": question.difficulty,
                "question_type": question.question_type,
                "expected_behavior": question.expected_behavior,
                "is_answerable": question.is_answerable,
                "answer": answer,
                "status": status,
                "source_pages": pages_to_string(question.source_pages),
                "keyword_hit_rate_at_5": f"{keyword_rate:.4f}",
                "keyword_recall_at_5": keyword_recall_at_5,
                "page_hit_at_5": page_hit_at_5,
                "page_recall_at_5": page_recall_at_5,
                "evidence_hit_at_5": evidence_hit_at_5,
                "mrr": f"{reciprocal_rank(question, top_5):.4f}",
                "citation_accuracy": _optional_score(citation_accuracy(question, answer, generation_chunks)),
                "citation_strict_accuracy": _optional_score(citation_score_info["citation_strict_accuracy"]),
                "citation_page_accuracy": _optional_score(citation_score_info["citation_page_accuracy"]),
                "citation_keyword_support": _optional_score(citation_score_info["citation_keyword_support"]),
                "citation_weighted_score": _optional_score(citation_score_info["citation_weighted_score"]),
                "citation_present": citation_present(answer),
                **citation_info,
                "refusal_accuracy": _optional_score(refusal_accuracy(question, answer)),
                "unsupported_claim_accuracy": _optional_score(unsupported_claim_accuracy(question, answer)),
                "expected_behavior_accuracy": _optional_score(expected_behavior_accuracy(question, answer)),
                "false_refusal": false_refusal(question, answer),
                "retrieved_pages": pages_to_string(retrieved_pages(top_5)),
                "top_score": f"{top_score:.4f}",
                "retrieval_latency_seconds": f"{retrieval_latency:.4f}",
                "generation_latency_seconds": f"{generation_latency:.4f}",
                "latency_seconds": f"{retrieval_latency + generation_latency:.4f}",
                "cache_hit": cache_hit,
            }
        )

    write_csv(args.output, rows, FIELDNAMES)
    write_summary(args.summary, "Custom RAG Evaluation Summary", summarize_rows(rows, thresholds))
    print(f"Wrote custom eval rows to {args.output}")
    print(f"Wrote custom eval summary to {args.summary}")


def summarize_rows(rows: list[dict[str, object]], thresholds: object) -> dict[str, float | int | str | None]:
    answerable = [row for row in rows if _as_bool(row["is_answerable"])]
    unanswerable = [row for row in rows if not _as_bool(row["is_answerable"])]
    answerable_with_pages = [row for row in answerable if row["source_pages"]]
    citation_scores = [
        float(row["citation_accuracy"])
        for row in rows
        if row["citation_accuracy"] != ""
    ]
    citation_strict_scores = [
        float(row["citation_strict_accuracy"])
        for row in rows
        if row.get("citation_strict_accuracy", "") != ""
    ]
    citation_page_scores = [
        float(row["citation_page_accuracy"])
        for row in rows
        if row.get("citation_page_accuracy", "") != ""
    ]
    citation_keyword_scores = [
        float(row["citation_keyword_support"])
        for row in rows
        if row.get("citation_keyword_support", "") != ""
    ]
    citation_weighted_scores = [
        float(row["citation_weighted_score"])
        for row in rows
        if row.get("citation_weighted_score", "") != ""
    ]
    refusal_scores = [
        float(row["refusal_accuracy"])
        for row in rows
        if row["refusal_accuracy"] != ""
    ]
    unsupported_claim_scores = [
        float(row["unsupported_claim_accuracy"])
        for row in rows
        if row.get("unsupported_claim_accuracy", "") != ""
    ]
    behavior_scores = [
        float(row["expected_behavior_accuracy"])
        for row in rows
        if row.get("expected_behavior_accuracy", "") != ""
    ]
    return {
        "total_questions": len(rows),
        "answerable_questions": len(answerable),
        "unanswerable_questions": len(unanswerable),
        "mean_keyword_hit_rate_at_5": mean([float(row["keyword_hit_rate_at_5"]) for row in answerable]),
        "keyword_recall_at_5": mean([1.0 if _as_bool(row["keyword_recall_at_5"]) else 0.0 for row in answerable]),
        "page_recall_at_5": mean([1.0 if _as_bool(row["page_recall_at_5"]) else 0.0 for row in answerable_with_pages]),
        "evidence_hit_at_5": mean([1.0 if _as_bool(row["evidence_hit_at_5"]) else 0.0 for row in answerable]),
        "mrr": mean([float(row["mrr"]) for row in answerable]),
        "citation_accuracy": mean(citation_scores) if citation_scores else None,
        "citation_strict_accuracy": mean(citation_strict_scores) if citation_strict_scores else None,
        "citation_page_accuracy": mean(citation_page_scores) if citation_page_scores else None,
        "citation_keyword_support": mean(citation_keyword_scores) if citation_keyword_scores else None,
        "citation_weighted_score": mean(citation_weighted_scores) if citation_weighted_scores else None,
        "refusal_accuracy": mean(refusal_scores) if refusal_scores else None,
        "unsupported_claim_accuracy": mean(unsupported_claim_scores) if unsupported_claim_scores else None,
        "expected_behavior_accuracy": mean(behavior_scores) if behavior_scores else None,
        "false_refusal_rate": mean([1.0 if _as_bool(row["false_refusal"]) else 0.0 for row in answerable]),
        "average_latency_seconds": mean([float(row["latency_seconds"]) for row in rows]),
    }


def _optional_score(value: float | None) -> str:
    return "" if value is None else f"{value:.4f}"


def _as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


if __name__ == "__main__":
    main()
