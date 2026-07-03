"""Inspect retrieval misses with chunk-level evidence before generation/RAGAS."""

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
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect retrieval misses with full top-k chunks.")
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS_PATH)
    parser.add_argument("--documents-dir", type=Path, default=DEFAULT_DOCUMENTS_DIR)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_RESULTS_DIR / "retrieval_miss_inspection.md")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--candidate-k", type=int, default=None)
    parser.add_argument("--similarity-threshold", type=float, default=0.3)
    parser.add_argument("--collection-name", default="rag_debug_retrieval_inspect")
    parser.add_argument("--persist-dir", type=Path, default=DEFAULT_DEBUG_PERSIST_DIR)
    parser.add_argument("--reuse-index", action="store_true")
    parser.add_argument(
        "--include-ok",
        action="store_true",
        help="Include successful retrieval rows as well as failures/warnings.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    questions = select_questions(args.questions, limit=args.limit)
    thresholds = load_debug_thresholds(args.config)
    print(f"Preparing retrieval miss inspection: questions={len(questions)}")
    pipeline = build_debug_pipeline(
        questions=questions,
        documents_dir=args.documents_dir,
        top_k=args.top_k,
        similarity_threshold=args.similarity_threshold,
        collection_name=args.collection_name,
        persist_directory=args.persist_dir,
        reset_store=not args.reuse_index,
    )

    lines = ["# Retrieval Miss Inspection", ""]
    inspected = 0
    for index, question in enumerate(questions, start=1):
        chunks, _latency = retrieve_for_question(
            pipeline,
            question,
            top_k=args.top_k,
            similarity_threshold=args.similarity_threshold,
            candidate_k=args.candidate_k,
        )
        row = retrieval_debug_row(question, chunks, thresholds)
        if not args.include_ok and row["status"] == "OK":
            continue

        inspected += 1
        lines.extend(
            [
                f"## {inspected}. {question.question}",
                "",
                f"- original_row: {index}",
                f"- difficulty: {question.difficulty}",
                f"- question_type: {question.question_type}",
                f"- expected_behavior: {question.expected_behavior}",
                f"- is_answerable: {question.is_answerable}",
                f"- expected_pages: {_pages_to_string(question.source_pages)}",
                f"- expected_keywords: {'; '.join(question.expected_keywords)}",
                f"- status: {row['status']}",
                f"- failure_reason: {row['failure_reason']}",
                f"- keyword_hit_rate_at_{args.top_k}: {row['keyword_hit_rate']}",
                f"- retrieved_pages: {row['retrieved_pages']}",
                "",
                "### Expected Answer",
                "",
                question.expected_answer or "(none)",
                "",
                "### Retrieved Chunks",
                "",
            ]
        )
        if not chunks:
            lines.extend(["No chunks retrieved.", ""])
            continue
        for chunk_index, chunk in enumerate(chunks[: args.top_k], start=1):
            metadata = chunk.get("metadata", {})
            lines.extend(
                [
                    f"#### Chunk {chunk_index}",
                    "",
                    f"- score: {float(chunk.get('score', 0.0)):.4f}",
                    f"- page: {metadata.get('page', '')}",
                    f"- section: {metadata.get('section_title', '')}",
                    f"- chunk_id: {chunk.get('chunk_id', '')}",
                    "",
                    "```text",
                    _shorten(chunk.get("text", ""), 1400),
                    "```",
                    "",
                ]
            )

    if inspected == 0:
        lines.append("No retrieval failures or warnings found.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote retrieval miss inspection to {args.output}")


def _pages_to_string(pages: tuple[int, ...]) -> str:
    return ";".join(str(page) for page in pages)


def _shorten(text: str, max_chars: int) -> str:
    cleaned = " ".join(text.split())
    if len(cleaned) <= max_chars:
        return cleaned
    return cleaned[: max_chars - 3].rstrip() + "..."


if __name__ == "__main__":
    main()
