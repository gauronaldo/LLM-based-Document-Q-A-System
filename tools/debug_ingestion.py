"""Debug long-document ingestion outside Streamlit.

Example:
    python tools/debug_ingestion.py "C:\\Users\\ADMIN\\Downloads\\file.pdf" --batches 1
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import get_config
from app.document_loader import DocumentLoader
from app.embedding_model import EmbeddingModel
from app.processing_debug import log_event, text_stats
from app.rag_pipeline import _embedding_text
from app.text_preprocessor import VietnameseTextPreprocessor
from app.text_splitter import TextSplitter
from app.vector_store import VectorStore


def main() -> None:
    parser = argparse.ArgumentParser(description="Debug document loading, chunking, and embedding.")
    parser.add_argument("file", help="Path to PDF/TXT/DOCX file.")
    parser.add_argument("--batches", type=int, default=1, help="Number of indexing batches to encode.")
    parser.add_argument("--indexing-batch-size", type=int, default=None)
    parser.add_argument("--embedding-batch-size", type=int, default=None)
    parser.add_argument("--write-vector-store", action="store_true", help="Also write debug batches to ChromaDB.")
    parser.add_argument(
        "--collection",
        default="debug_ingestion_probe",
        help="Temporary Chroma collection name used with --write-vector-store.",
    )
    args = parser.parse_args()

    config = get_config()
    file_path = Path(args.file)
    indexing_batch_size = args.indexing_batch_size or config.indexing_batch_size
    embedding_batch_size = args.embedding_batch_size or config.embedding_batch_size

    started = time.perf_counter()
    print(f"File: {file_path}")
    print(f"Embedding model: {config.embedding_model}")
    print(f"Embedding batch size: {embedding_batch_size}")
    print(f"Indexing batch size: {indexing_batch_size}")
    print(f"Write vector store: {args.write_vector_store}")
    print("Debug log: logs/processing_debug.log")

    loader = DocumentLoader()
    preprocessor = VietnameseTextPreprocessor()
    splitter = TextSplitter(parent_context_max_chars=config.parent_context_max_chars)
    embedder = EmbeddingModel(config.embedding_model, batch_size=embedding_batch_size)
    vector_store = None
    if args.write_vector_store:
        vector_store = VectorStore(
            config.chroma_persist_dir,
            args.collection,
            upsert_batch_size=indexing_batch_size,
        )
        print(f"Resetting debug collection: {args.collection}")
        vector_store.reset()

    documents = loader.load(str(file_path), file_path.name, file_path.stem)
    print(f"Loaded documents/pages: {len(documents)} | {text_stats([d['text'] for d in documents])}")

    cleaned = []
    for document in documents:
        text = preprocessor.clean(document["text"])
        if text:
            cleaned.append({"text": text, "metadata": document["metadata"]})
    print(f"Cleaned documents/pages: {len(cleaned)} | {text_stats([d['text'] for d in cleaned])}")

    chunks = splitter.split(cleaned)
    print(f"Chunks: {len(chunks)} | {text_stats([c['text'] for c in chunks])}")
    log_event("debug_ingestion_chunks_ready", chunks=len(chunks))

    max_chunks = max(0, args.batches) * indexing_batch_size
    chunks_to_test = chunks[:max_chunks]
    for start in range(0, len(chunks_to_test), indexing_batch_size):
        batch = chunks_to_test[start : start + indexing_batch_size]
        batch_no = start // indexing_batch_size + 1
        print(f"Encoding debug batch {batch_no}: chunks {start + 1}-{start + len(batch)}")
        batch_started = time.perf_counter()
        embeddings = embedder.encode(
            [_embedding_text(chunk) for chunk in batch],
            progress_callback=lambda done, total: print(f"  embedding progress {done}/{total}"),
        )
        print(
            f"  done: embeddings={len(embeddings)} "
            f"elapsed={time.perf_counter() - batch_started:.2f}s"
        )
        if vector_store is not None:
            write_started = time.perf_counter()
            print("  writing vector store...")
            log_event(
                "debug_vector_store_write_start",
                batch=batch_no,
                chunks=len(batch),
                collection=args.collection,
            )
            vector_store.add_chunks(batch, embeddings)
            log_event(
                "debug_vector_store_write_done",
                batch=batch_no,
                chunks=len(batch),
                collection=args.collection,
                elapsed_seconds=round(time.perf_counter() - write_started, 3),
            )
            print(f"  vector write done elapsed={time.perf_counter() - write_started:.2f}s")

    print(f"Finished debug run in {time.perf_counter() - started:.2f}s")


if __name__ == "__main__":
    main()
