"""Streamlit entry point for the LLM-based document QA app."""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config import get_config
from app.document_loader import DocumentLoaderError
from app.embedding_model import EmbeddingModelError
from app.llm_client import LLMClientError
from app.rag_pipeline import create_default_pipeline
from app.retriever import RetrieverError
from app.vector_store import VectorStoreError


def _safe_file_name(file_name: str) -> str:
    """Return a filesystem-safe file name."""

    return Path(file_name).name.replace("/", "_").replace("\\", "_")


def _file_id(file_bytes: bytes) -> str:
    """Create a stable short id from uploaded file bytes."""

    return hashlib.sha256(file_bytes).hexdigest()[:12]


def _save_uploaded_file(uploaded_file, raw_data_dir: Path) -> tuple[Path, str]:
    """Save an uploaded Streamlit file and return its path and file id."""

    file_bytes = uploaded_file.getvalue()
    file_id = _file_id(file_bytes)
    file_name = _safe_file_name(uploaded_file.name)
    raw_data_dir.mkdir(parents=True, exist_ok=True)
    file_path = raw_data_dir / f"{file_id}_{file_name}"
    file_path.write_bytes(file_bytes)
    return file_path, file_id


def _get_pipeline(top_k: int, similarity_threshold: float):
    """Create a pipeline for the current UI settings."""

    return create_default_pipeline(top_k=top_k, similarity_threshold=similarity_threshold)


def main() -> None:
    """Render the Streamlit document QA application."""

    config = get_config()

    st.set_page_config(
        page_title="Vietnamese Document Q&A",
        layout="wide",
    )

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "processed_summary" not in st.session_state:
        st.session_state.processed_summary = None

    st.title("LLM-based Document Q&A with Vietnamese Support")
    st.write(
        "Upload a PDF, TXT, or DOCX document, process it into a searchable "
        "RAG index, then ask grounded questions with source citations."
    )

    with st.sidebar:
        st.header("Document")
        uploaded_file = st.file_uploader("Upload document", type=["pdf", "txt", "docx"])

        st.header("Settings")
        top_k = st.slider("Top-k retrieved chunks", min_value=1, max_value=10, value=config.top_k)
        similarity_threshold = st.slider(
            "Similarity threshold",
            min_value=0.0,
            max_value=1.0,
            value=float(config.similarity_threshold),
            step=0.05,
        )

        pipeline = _get_pipeline(top_k=top_k, similarity_threshold=similarity_threshold)
        process_clicked = st.button("Process document", disabled=uploaded_file is None)
        reset_clicked = st.button("Reset vector database")

    if reset_clicked:
        try:
            pipeline.reset()
            st.session_state.chat_history = []
            st.session_state.processed_summary = None
            st.success("Vector database reset.")
        except VectorStoreError as exc:
            st.error(str(exc))

    if process_clicked and uploaded_file is not None:
        try:
            file_path, file_id = _save_uploaded_file(uploaded_file, config.raw_data_dir)
            with st.spinner("Processing document..."):
                summary = pipeline.ingest_document(
                    file_path=str(file_path),
                    file_name=uploaded_file.name,
                    file_id=file_id,
                )
            st.session_state.processed_summary = summary
            st.success(
                f"Processed {summary['file_name']} with {summary['num_chunks']} chunks."
            )
        except (DocumentLoaderError, EmbeddingModelError, VectorStoreError) as exc:
            st.error(str(exc))
        except Exception as exc:
            st.error(f"Unexpected processing error: {exc}")

    if st.session_state.processed_summary:
        summary = st.session_state.processed_summary
        st.caption(
            f"Active document: {summary['file_name']} | "
            f"Chunks: {summary['num_chunks']}"
        )
        quick_question = _render_quick_actions()
    else:
        st.warning("Upload and process a document before asking questions.")
        quick_question = None

    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    typed_question = st.chat_input("Ask a question about the processed document")
    question = quick_question or typed_question
    if question:
        previous_history = list(st.session_state.chat_history)
        st.session_state.chat_history.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        if not st.session_state.processed_summary:
            answer = "Please process a document first."
            st.session_state.chat_history.append({"role": "assistant", "content": answer})
            with st.chat_message("assistant"):
                st.warning(answer)
            return

        with st.chat_message("assistant"):
            try:
                with st.spinner("Retrieving evidence and generating answer..."):
                    result = pipeline.answer_question(
                        question,
                        chat_history=previous_history,
                    )

                st.markdown(result["answer"])
                st.session_state.chat_history.append(
                    {"role": "assistant", "content": result["answer"]}
                )

                if result["sources"]:
                    st.subheader("Sources")
                    for source in result["sources"]:
                        page = source["page"] if source["page"] is not None else "N/A"
                        st.markdown(
                            f"- **[{source.get('source_id', '?')}]** "
                            f"{source['file_name']} · page {page}"
                        )

                if result["retrieved_chunks"]:
                    st.subheader("Retrieved Evidence")
                    for index, chunk in enumerate(result["retrieved_chunks"], start=1):
                        score = chunk.get("score", 0.0)
                        metadata = chunk.get("metadata", {})
                        page = metadata.get("page", "N/A")
                        file_name = metadata.get("file_name", "unknown")
                        title = (
                            f"[{index}] {file_name} · page {page} · "
                            f"relevance {score:.0%}"
                        )
                        with st.expander(title):
                            st.caption(f"Chunk ID: {chunk.get('chunk_id', 'unknown')}")
                            st.write(chunk["text"])
            except (RetrieverError, LLMClientError, EmbeddingModelError, VectorStoreError) as exc:
                st.error(str(exc))
            except Exception as exc:
                st.error(f"Unexpected answer error: {exc}")


def _render_quick_actions() -> str | None:
    """Render prompt shortcuts and return a selected question."""

    st.write("Try asking:")
    columns = st.columns(4)
    actions = [
        ("Summarize", "Summarize this document in clear bullet points."),
        ("Key points", "Extract the key points from this document."),
        ("Requirements", "List the requirements or conditions mentioned in this document."),
        ("Explain", "Explain the main idea of this document in simple terms."),
    ]

    for column, (label, question) in zip(columns, actions):
        if column.button(label, use_container_width=True):
            return question

    return None


if __name__ == "__main__":
    main()
