"""Debug generation with gold context, independent from retrieval quality."""

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
    answer_keyword_pass,
    build_debug_pipeline,
    cache_namespace_for_pipeline,
    gold_context_chunks,
    join_chunk_text,
    load_debug_thresholds,
    select_questions,
    generate_with_context,
    shorten,
    write_csv,
)


FIELDNAMES = [
    "question",
    "difficulty",
    "gold_context",
    "expected_answer",
    "generated_answer",
    "is_answer_correct_keyword_based",
    "latency_seconds",
    "cache_hit",
    "notes",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run generation debug with gold context.")
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS_PATH)
    parser.add_argument("--documents-dir", type=Path, default=DEFAULT_DOCUMENTS_DIR)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_RESULTS_DIR / "generation_debug_report.csv")
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE_DIR / "predictions.csv")
    parser.add_argument("--sample-size", type=int, default=10)
    parser.add_argument("--sample-strategy", choices=["first", "stratified"], default="first")
    parser.add_argument("--answerable-count", type=int, default=0)
    parser.add_argument("--unanswerable-count", type=int, default=0)
    parser.add_argument("--hard-count", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--similarity-threshold", type=float, default=0.3)
    parser.add_argument("--collection-name", default="rag_debug_generation")
    parser.add_argument("--persist-dir", type=Path, default=DEFAULT_DEBUG_PERSIST_DIR)
    parser.add_argument("--include-unanswerable", action="store_true")
    parser.add_argument("--reuse-index", action="store_true")
    parser.add_argument("--refresh-cache", action="store_true")
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
        limit=args.sample_size,
        include_unanswerable=args.include_unanswerable,
        sample_strategy=args.sample_strategy,
        answerable_count=args.answerable_count,
        unanswerable_count=args.unanswerable_count,
        hard_count=args.hard_count,
    )
    thresholds = load_debug_thresholds(args.config)
    print(f"Preparing generation debug: questions={len(questions)}")
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
            top_k=args.top_k,
            similarity_threshold=args.similarity_threshold,
            mode=f"gold_context_repair={not args.skip_repairs}",
        ),
    )

    rows = []
    for index, question in enumerate(questions, start=1):
        print(f"[{index}/{len(questions)}] Generating with gold context: {question.question}")
        chunks = gold_context_chunks(pipeline, question, max_contexts=args.top_k)
        if chunks:
            answer, latency, cache_hit = generate_with_context(
                pipeline,
                question,
                chunks,
                cache=cache,
                mode="gold_context",
                enable_language_repair=not args.skip_repairs,
                enable_answer_repair=not args.skip_repairs,
                enable_refusal_retry=not args.skip_repairs,
            )
        else:
            answer = refusal_for_question(question.question)
            latency = 0.0
            cache_hit = False
        rows.append(
            {
                "question": question.question,
                "difficulty": question.difficulty,
                "gold_context": shorten(join_chunk_text(chunks), 2400),
                "expected_answer": question.expected_answer,
                "generated_answer": answer,
                "is_answer_correct_keyword_based": answer_keyword_pass(question, answer, thresholds),
                "latency_seconds": f"{latency:.4f}",
                "cache_hit": cache_hit,
                "notes": "NO_GOLD_CONTEXT" if not chunks else "",
            }
        )

    write_csv(args.output, rows, FIELDNAMES)
    print(f"Wrote generation debug report to {args.output}")


if __name__ == "__main__":
    main()
