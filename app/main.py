"""Streamlit entry point for the LLM-based document QA app."""

from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("STREAMLIT_SERVER_FILE_WATCHER_TYPE", "none")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from app.processing_debug import log_event


def _preload_sentence_transformers() -> None:
    """Import sentence-transformers before Streamlit initializes its runtime."""

    try:
        log_event("streamlit_preimport_sentence_transformers_start")
        from sentence_transformers import SentenceTransformer as _SentenceTransformer

        _ = _SentenceTransformer
        log_event("streamlit_preimport_sentence_transformers_done")
    except Exception as exc:
        log_event("streamlit_preimport_sentence_transformers_error", error=repr(exc))


_preload_sentence_transformers()

import streamlit as st

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


def _get_pipeline(config, top_k: int, similarity_threshold: float):
    """Create a pipeline for the current UI settings."""

    pipeline_key = (
        config.llm_provider,
        config.llm_model,
        config.embedding_model,
        config.embedding_batch_size,
        config.indexing_batch_size,
        config.parent_context_max_chars,
        str(config.chroma_persist_dir),
        config.chroma_collection_name,
        config.hybrid_alpha,
        config.use_hybrid_search,
        config.use_mmr,
        config.reranker_model,
        top_k,
        similarity_threshold,
    )
    if (
        st.session_state.get("pipeline") is None
        or st.session_state.get("pipeline_key") != pipeline_key
    ):
        st.session_state.pipeline = create_default_pipeline(
            config=config,
            top_k=top_k,
            similarity_threshold=similarity_threshold,
        )
        st.session_state.pipeline_key = pipeline_key
    return st.session_state.pipeline


def main() -> None:
    """Render the Streamlit document QA application."""

    config = get_config()

    st.set_page_config(page_title="Vietnamese Document Q&A", layout="wide")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "processed_summary" not in st.session_state:
        st.session_state.processed_summary = None
    if "pipeline" not in st.session_state:
        st.session_state.pipeline = None
    if "pipeline_key" not in st.session_state:
        st.session_state.pipeline_key = None

    st.title("LLM-based Document Q&A with Vietnamese Support")
    st.write(
        "Upload a PDF, TXT, or DOCX document, process it into a searchable "
        "RAG index, then chat with the document using grounded answers and citations."
    )

    with st.sidebar:
        st.header("Document")
        uploaded_file = st.file_uploader("Upload document", type=["pdf", "txt", "docx"])

        st.header("Status")
        st.caption(f"Provider: {config.llm_provider}")
        st.caption(f"Model: {config.llm_model}")
        st.caption(f"Embedding: {config.embedding_model.split('/')[-1]}")
        st.caption(f"Embedding batch: {config.embedding_batch_size}")
        st.caption(f"Indexing batch: {config.indexing_batch_size}")
        st.caption(f"Hybrid search: {'on' if config.use_hybrid_search else 'off'}")
        st.caption(f"MMR: {'on' if config.use_mmr else 'off'}")
        st.caption(f"Reranker: {config.reranker_model or 'lexical fallback'}")
        st.caption(f"API key: {'configured' if _is_api_key_configured(config) else 'missing'}")

        st.header("Settings")
        st.caption("Retrieval: auto top-k and auto threshold are enabled.")
        top_k = st.slider("Base top-k", min_value=1, max_value=10, value=config.top_k)
        similarity_threshold = st.slider(
            "Base similarity threshold",
            min_value=0.0,
            max_value=1.0,
            value=float(config.similarity_threshold),
            step=0.05,
        )

        pipeline = _get_pipeline(
            config=config,
            top_k=top_k,
            similarity_threshold=similarity_threshold,
        )
        process_clicked = st.button("Process document", disabled=uploaded_file is None)
        reset_clicked = st.button("Reset vector database")
        clear_chat_clicked = st.button("Clear chat")

    if clear_chat_clicked:
        st.session_state.chat_history = []
        st.success("Chat cleared.")

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
            log_event(
                "streamlit_process_clicked",
                file_name=uploaded_file.name,
                uploaded_size=getattr(uploaded_file, "size", None),
            )
            file_path, file_id = _save_uploaded_file(uploaded_file, config.raw_data_dir)
            log_event(
                "streamlit_upload_saved",
                file_name=uploaded_file.name,
                file_path=str(file_path),
                file_id=file_id,
            )
            with st.spinner("Processing document..."):
                progress_bar = st.progress(0.0)
                progress_message = st.empty()

                def update_progress(message: str, value: float) -> None:
                    progress_message.info(message)
                    progress_bar.progress(max(0.0, min(1.0, value)))

                update_progress("Resetting vector database...", 0.02)
                log_event("streamlit_pipeline_reset_start", file_name=uploaded_file.name)
                pipeline.reset()
                log_event("streamlit_pipeline_reset_done", file_name=uploaded_file.name)
                summary = pipeline.ingest_document(
                    file_path=str(file_path),
                    file_name=uploaded_file.name,
                    file_id=file_id,
                    progress_callback=update_progress,
                )
                progress_message.success("Document processed successfully.")
            st.session_state.processed_summary = summary
            st.session_state.chat_history = []
            st.success(f"Document ready: {summary['num_chunks']} chunks indexed.")
        except (DocumentLoaderError, EmbeddingModelError, VectorStoreError) as exc:
            log_event("streamlit_process_known_error", error=repr(exc))
            st.error(str(exc))
        except Exception as exc:
            log_event("streamlit_process_unexpected_error", error=repr(exc))
            st.error(f"Unexpected processing error: {exc}")
            st.exception(exc)

    if st.session_state.processed_summary:
        summary = st.session_state.processed_summary
        st.success(f"Active document: {summary['file_name']} ({summary['num_chunks']} chunks)")
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
                answer_placeholder = st.empty()

                def update_stream(partial_answer: str) -> None:
                    if partial_answer.strip():
                        answer_placeholder.markdown(partial_answer)

                with st.spinner("Retrieving evidence..."):
                    result = pipeline.answer_question(
                        question,
                        chat_history=previous_history,
                        stream_callback=update_stream,
                    )

                answer_placeholder.markdown(result["answer"])
                st.session_state.chat_history.append(
                    {"role": "assistant", "content": result["answer"]}
                )

                _render_sources(result.get("sources", []))
                _render_evidence(result.get("retrieved_chunks", []))
            except (RetrieverError, LLMClientError, EmbeddingModelError, VectorStoreError) as exc:
                st.error(str(exc))
            except Exception as exc:
                st.error(f"Unexpected answer error: {exc}")


def _render_sources(sources: list[dict]) -> None:
    """Render human-readable source citations."""

    if not sources:
        return

    st.subheader("Sources")
    for source in sources:
        page = source["page"] if source["page"] is not None else "N/A"
        source_id = source.get("source_id", "?")
        st.markdown(f"- **[{source_id}]** {source['file_name']} - page {page}")


def _render_evidence(retrieved_chunks: list[dict]) -> None:
    """Render retrieved evidence in compact expanders."""

    if not retrieved_chunks:
        return

    st.subheader("Retrieved Evidence")
    for index, chunk in enumerate(retrieved_chunks, start=1):
        score = chunk.get("score", 0.0)
        metadata = chunk.get("metadata", {})
        page = metadata.get("page", "N/A")
        file_name = metadata.get("file_name", "unknown")
        section_title = metadata.get("section_title", "Document")
        title = f"[{index}] {file_name} - page {page} - {section_title} - relevance {score:.0%}"
        with st.expander(title):
            st.caption(f"Chunk ID: {chunk.get('chunk_id', 'unknown')}")
            st.caption(f"Section: {section_title}")
            st.write(chunk["text"])


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


def _is_api_key_configured(config) -> bool:
    """Return whether the configured provider has an API key or local runtime."""

    if config.llm_provider == "gemini":
        return bool(config.gemini_api_key)
    if config.llm_provider == "openai":
        return bool(config.openai_api_key)
    if config.llm_provider == "ollama":
        return True
    return False


if __name__ == "__main__":
    main()
