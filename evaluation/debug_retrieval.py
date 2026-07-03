"""Fast retrieval-only debug report for the RAG pipeline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.debug_utils import (
    DEFAULT_CONFIG_PATH,
    DEFAULT_DEBUG_PERSIST_DIR,
    DEFAULT_DOCUMENTS_DIR,
    DEFAULT_QUESTIONS_PATH,
    DEFAULT_RESULTS_DIR,
    build_debug_pipeline,
    load_debug_thresholds,
    retrieval_debug_row,
    retrieve_for_question,
    select_questions,
    write_csv,
)


FIELDNAMES = [
    "question",
    "difficulty",
    "is_answerable",
    "expected_keywords",
    "source_pages",
    "retrieved_pages",
    "top_1_text",
    "top_3_text",
    "top_5_text",
    "keyword_hit_count",
    "keyword_total",
    "keyword_hit_rate",
    "page_hit_at_1",
    "page_hit_at_3",
    "page_hit_at_5",
    "top_score",
    "status",
    "failure_reason",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run retrieval-only RAG debugging.")
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS_PATH)
    parser.add_argument("--documents-dir", type=Path, default=DEFAULT_DOCUMENTS_DIR)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_RESULTS_DIR / "retrieval_debug_report.csv")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--sample-strategy", choices=["first", "stratified"], default="first")
    parser.add_argument("--answerable-count", type=int, default=0)
    parser.add_argument("--unanswerable-count", type=int, default=0)
    parser.add_argument("--hard-count", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--candidate-k", type=int, default=None)
    parser.add_argument("--similarity-threshold", type=float, default=0.3)
    parser.add_argument("--collection-name", default="rag_debug_retrieval")
    parser.add_argument("--persist-dir", type=Path, default=DEFAULT_DEBUG_PERSIST_DIR)
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
    print(f"Preparing retrieval debug: questions={len(questions)}")
    pipeline = build_debug_pipeline(
        questions=questions,
        documents_dir=args.documents_dir,
        top_k=args.top_k,
        similarity_threshold=args.similarity_threshold,
        collection_name=args.collection_name,
        persist_directory=args.persist_dir,
        reset_store=not args.reuse_index,
    )

    rows = []
    for index, question in enumerate(questions, start=1):
        print(f"[{index}/{len(questions)}] Retrieving: {question.question}")
        chunks, _latency = retrieve_for_question(
            pipeline,
            question,
            top_k=args.top_k,
            similarity_threshold=args.similarity_threshold,
            candidate_k=args.candidate_k,
        )
        rows.append(retrieval_debug_row(question, chunks, thresholds))

    write_csv(args.output, rows, FIELDNAMES)
    print(f"Wrote retrieval debug report to {args.output}")


if __name__ == "__main__":
    main()
