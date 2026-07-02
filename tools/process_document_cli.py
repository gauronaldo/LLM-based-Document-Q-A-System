"""Run the same ingestion pipeline as Streamlit, outside Streamlit.

This is intended to separate core pipeline failures from Streamlit UI/session issues.

Example:
    python tools/process_document_cli.py "C:\\Users\\ADMIN\\Downloads\\file.pdf"
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import get_config
from app.processing_debug import log_event
from app.rag_pipeline import create_default_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Process a document with the production RAG pipeline.")
    parser.add_argument("file", help="Path to PDF/TXT/DOCX file.")
    parser.add_argument(
        "--collection",
        default="debug_full_ingestion_probe",
        help="Chroma collection to reset and ingest into.",
    )
    parser.add_argument(
        "--use-app-collection",
        action="store_true",
        help="Use and reset the configured app collection instead of a debug collection.",
    )
    args = parser.parse_args()

    file_path = Path(args.file)
    base_config = get_config()
    config = (
        base_config
        if args.use_app_collection
        else replace(base_config, chroma_collection_name=args.collection)
    )

    print(f"File: {file_path}")
    print(f"Collection: {config.chroma_collection_name}")
    print(f"Embedding model: {config.embedding_model}")
    print(f"Embedding batch size: {config.embedding_batch_size}")
    print(f"Indexing batch size: {config.indexing_batch_size}")
    print(f"Parent context max chars: {config.parent_context_max_chars}")
    print("Debug log: logs/processing_debug.log")

    pipeline = create_default_pipeline(config=config)
    started = time.perf_counter()

    def progress(message: str, value: float) -> None:
        print(f"{value:.2%} | {message}")

    log_event(
        "cli_process_start",
        file_name=file_path.name,
        collection=config.chroma_collection_name,
    )
    print("Resetting vector database...")
    pipeline.reset()
    summary = pipeline.ingest_document(
        file_path=str(file_path),
        file_name=file_path.name,
        file_id=file_path.stem,
        progress_callback=progress,
    )
    elapsed = time.perf_counter() - started
    log_event(
        "cli_process_done",
        file_name=file_path.name,
        collection=config.chroma_collection_name,
        chunks=summary.get("num_chunks"),
        elapsed_seconds=round(elapsed, 3),
    )
    print(f"Summary: {summary}")
    print(f"Finished in {elapsed:.2f}s")


if __name__ == "__main__":
    main()
