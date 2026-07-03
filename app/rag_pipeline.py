"""Main RAG orchestration pipeline."""

from __future__ import annotations

import inspect
import time
from typing import Any, Callable

from app.config import AppConfig, get_config
from app.answer_postprocessor import (
    looks_like_refusal,
    postprocess_answer_behavior,
)
from app.behavior_classifier import classify_expected_behavior
from app.citation_validator import post_check_citations
from app.context_support import NO_SUPPORT, estimate_context_support
from app.document_loader import DocumentLoader
from app.embedding_model import EmbeddingModel
from app.llm_client import LLMClient
from app.prompt_template import (
    answer_needs_repair,
    answer_language_mismatch,
    build_answer_required_prompt,
    build_answer_repair_prompt,
    build_prompt,
    build_language_repair_prompt,
    extract_sources,
    normalize_answer,
    refusal_for_question,
)
from app.processing_debug import log_event, text_stats
from app.query_intent import detect_query_intent, retrieval_plan_for_intent
from app.query_profiles import QueryProfile, load_query_profile
from app.query_rewriter import build_retrieval_query
from app.retriever import Retriever
from app.reranker import Reranker
from app.text_preprocessor import VietnameseTextPreprocessor
from app.text_splitter import TextSplitter
from app.vector_store import VectorStore


class RAGPipeline:
    """Coordinate document ingestion and grounded conversational answering."""

    def __init__(
        self,
        document_loader: Any,
        preprocessor: Any,
        splitter: Any,
        embedder: Any,
        vector_store: Any,
        retriever: Any,
        llm: Any,
        indexing_batch_size: int = 128,
        query_profile: QueryProfile | None = None,
    ):
        if indexing_batch_size <= 0:
            raise ValueError("indexing_batch_size must be greater than 0.")

        self.document_loader = document_loader
        self.preprocessor = preprocessor
        self.splitter = splitter
        self.embedder = embedder
        self.vector_store = vector_store
        self.retriever = retriever
        self.llm = llm
        self.indexing_batch_size = indexing_batch_size
        self.query_profile = query_profile or load_query_profile()

    def ingest_document(
        self,
        file_path: str,
        file_name: str,
        file_id: str,
        progress_callback: Callable[[str, float], None] | None = None,
    ) -> dict[str, Any]:
        """Load, clean, chunk, embed, and store a document."""

        _notify_progress(progress_callback, "Loading document text...", 0.05)
        ingest_started = time.perf_counter()
        log_event("ingest_start", file_name=file_name, file_id=file_id)
        documents = self.document_loader.load(file_path, file_name, file_id)
        log_event(
            "document_load_done",
            documents=len(documents),
            **text_stats([document.get("text", "") for document in documents]),
        )
        processed_documents = []

        _notify_progress(progress_callback, "Cleaning extracted text...", 0.2)
        for document in documents:
            clean_text = self.preprocessor.clean(document["text"])
            if clean_text:
                processed_documents.append(
                    {
                        "text": clean_text,
                        "metadata": document["metadata"],
                    }
                )
        log_event(
            "document_clean_done",
            documents=len(processed_documents),
            **text_stats([document.get("text", "") for document in processed_documents]),
        )

        _notify_progress(progress_callback, "Splitting text into searchable chunks...", 0.35)
        chunks = self.splitter.split(processed_documents)
        log_event(
            "chunk_split_done",
            chunks=len(chunks),
            **text_stats([chunk.get("text", "") for chunk in chunks]),
        )
        if not chunks:
            return {
                "status": "empty",
                "file_name": file_name,
                "num_documents": len(documents),
                "num_chunks": 0,
            }

        if len(chunks) >= 2000:
            _notify_progress(
                progress_callback,
                f"Large document detected: {len(chunks)} chunks. Indexing in batches...",
                0.5,
            )

        _notify_progress(
            progress_callback,
            f"Loading embedding model and indexing {len(chunks)} chunks...",
            0.55,
        )
        _index_chunks_with_progress(
            embedder=self.embedder,
            vector_store=self.vector_store,
            chunks=chunks,
            indexing_batch_size=self.indexing_batch_size,
            progress_callback=progress_callback,
        )
        _clear_retriever_cache(self.retriever)
        log_event(
            "ingest_done",
            file_name=file_name,
            chunks=len(chunks),
            elapsed_seconds=round(time.perf_counter() - ingest_started, 3),
        )
        _notify_progress(progress_callback, "Document indexing complete.", 1.0)

        return {
            "status": "success",
            "file_name": file_name,
            "num_documents": len(documents),
            "num_chunks": len(chunks),
        }

    def answer_question(
        self,
        question: str,
        chat_history: list[dict[str, str]] | None = None,
        stream_callback: Callable[[str], None] | None = None,
    ) -> dict[str, Any]:
        """Answer a question conversationally using retrieved document context."""

        clean_question = self.preprocessor.clean_query(question)
        if not clean_question:
            return {
                "answer": refusal_for_question(clean_question),
                "sources": [],
                "retrieved_chunks": [],
            }

        intent = detect_query_intent(clean_question)
        expected_behavior = classify_expected_behavior(clean_question)
        retrieval_query = build_retrieval_query(
            clean_question,
            chat_history,
            query_profile=self.query_profile,
        )
        retrieval_plan = retrieval_plan_for_intent(
            intent=intent,
            default_top_k=self.retriever.top_k,
            default_threshold=self.retriever.similarity_threshold,
        )
        retrieved_chunks = self.retriever.retrieve(
            retrieval_query,
            top_k=retrieval_plan.top_k,
            similarity_threshold=retrieval_plan.similarity_threshold,
            auto=True,
            intent=intent,
            original_question=clean_question,
        )

        if not retrieved_chunks:
            return {
                "answer": refusal_for_question(clean_question),
                "sources": [],
                "retrieved_chunks": [],
            }

        support_level = estimate_context_support(clean_question, retrieved_chunks)
        prompt = build_prompt(
            question=clean_question,
            retrieved_chunks=retrieved_chunks,
            chat_history=chat_history,
            intent=intent,
            expected_behavior=expected_behavior,
            profile_instruction=self.query_profile.prompt_extension,
        )
        raw_answer = self._generate_answer(prompt, stream_callback=stream_callback)
        answer = normalize_answer(
            raw_answer,
            question=clean_question,
            retrieved_chunks=retrieved_chunks,
        )
        if answer_language_mismatch(answer, clean_question):
            answer = normalize_answer(
                self.llm.generate(build_language_repair_prompt(answer, clean_question)),
                question=clean_question,
                retrieved_chunks=retrieved_chunks,
            )
            if stream_callback:
                stream_callback(answer)

        if answer_needs_repair(answer):
            answer = normalize_answer(
                self.llm.generate(
                    build_answer_repair_prompt(
                        answer=answer,
                        question=clean_question,
                        retrieved_chunks=retrieved_chunks,
                        intent=intent,
                        expected_behavior=expected_behavior,
                        profile_instruction=self.query_profile.prompt_extension,
                    )
                ),
                question=clean_question,
                retrieved_chunks=retrieved_chunks,
            )
            if answer_language_mismatch(answer, clean_question):
                answer = normalize_answer(
                    self.llm.generate(build_language_repair_prompt(answer, clean_question)),
                    question=clean_question,
                    retrieved_chunks=retrieved_chunks,
                )
            if answer_needs_repair(answer):
                if support_level == NO_SUPPORT:
                    answer = refusal_for_question(clean_question)
                else:
                    answer = normalize_answer(
                        self.llm.generate(
                            build_answer_required_prompt(
                                answer=answer,
                                question=clean_question,
                                retrieved_chunks=retrieved_chunks,
                                intent=intent,
                                expected_behavior=expected_behavior,
                                profile_instruction=self.query_profile.prompt_extension,
                            )
                        ),
                        question=clean_question,
                        retrieved_chunks=retrieved_chunks,
                    )
            if stream_callback:
                stream_callback(answer)

        if looks_like_refusal(answer) and support_level != NO_SUPPORT:
            answer = normalize_answer(
                self.llm.generate(
                    build_answer_required_prompt(
                        answer=answer,
                        question=clean_question,
                        retrieved_chunks=retrieved_chunks,
                        intent=intent,
                        expected_behavior=expected_behavior,
                        profile_instruction=self.query_profile.prompt_extension,
                    )
                ),
                question=clean_question,
                retrieved_chunks=retrieved_chunks,
            )
            if answer_language_mismatch(answer, clean_question):
                answer = normalize_answer(
                    self.llm.generate(build_language_repair_prompt(answer, clean_question)),
                    question=clean_question,
                    retrieved_chunks=retrieved_chunks,
                )
            if answer_needs_repair(answer):
                answer = refusal_for_question(clean_question)
            if stream_callback:
                stream_callback(answer)

        answer = postprocess_answer_behavior(
            answer=answer,
            question=clean_question,
            support_level=support_level,
            expected_behavior=expected_behavior,
        )
        answer = post_check_citations(answer, clean_question, retrieved_chunks)

        return {
            "answer": answer,
            "sources": extract_sources(retrieved_chunks),
            "retrieved_chunks": retrieved_chunks,
            "intent": intent,
            "retrieval_query": retrieval_query,
        }

    def _generate_answer(
        self,
        prompt: str,
        stream_callback: Callable[[str], None] | None = None,
    ) -> str:
        if not stream_callback or not hasattr(self.llm, "generate_stream"):
            return self.llm.generate(prompt)

        chunks: list[str] = []
        for chunk in self.llm.generate_stream(prompt):
            chunks.append(chunk)
            stream_callback("".join(chunks))
        return "".join(chunks)

    def reset(self) -> None:
        """Reset the underlying vector store."""

        self.vector_store.reset()
        _clear_retriever_cache(self.retriever)


def create_default_pipeline(
    config: AppConfig | None = None,
    top_k: int | None = None,
    similarity_threshold: float | None = None,
) -> RAGPipeline:
    """Create a production pipeline from application configuration."""

    config = config or get_config()
    query_profile = load_query_profile(config.document_profile)
    embedder = EmbeddingModel(
        config.embedding_model,
        batch_size=config.embedding_batch_size,
    )
    vector_store = VectorStore(
        config.chroma_persist_dir,
        config.chroma_collection_name,
        upsert_batch_size=config.indexing_batch_size,
    )
    retriever = Retriever(
        vector_store=vector_store,
        embedder=embedder,
        top_k=top_k or config.top_k,
        similarity_threshold=(
            config.similarity_threshold
            if similarity_threshold is None
            else similarity_threshold
        ),
        hybrid_alpha=config.hybrid_alpha,
        use_hybrid_search=config.use_hybrid_search,
        use_mmr=config.use_mmr,
        reranker=Reranker(model_name=config.reranker_model),
        query_profile=query_profile,
        use_multi_query=config.use_multi_query,
        multi_query_count=config.multi_query_count,
    )
    llm = LLMClient(
        provider=config.llm_provider,
        model_name=config.llm_model,
        gemini_api_key=config.gemini_api_key,
        openai_api_key=config.openai_api_key,
    )

    return RAGPipeline(
        document_loader=DocumentLoader(),
        preprocessor=VietnameseTextPreprocessor(),
        splitter=TextSplitter(parent_context_max_chars=config.parent_context_max_chars),
        embedder=embedder,
        vector_store=vector_store,
        retriever=retriever,
        llm=llm,
        indexing_batch_size=config.indexing_batch_size,
        query_profile=query_profile,
    )


def _embedding_text(chunk: dict[str, Any]) -> str:
    metadata = chunk.get("metadata", {})
    section_title = metadata.get("section_title", "")
    if not section_title or section_title == "Document":
        return chunk["text"]
    return f"Section: {section_title}\n{chunk['text']}"


def _notify_progress(
    progress_callback: Callable[[str, float], None] | None,
    message: str,
    value: float,
) -> None:
    if progress_callback:
        progress_callback(message, value)


def _index_chunks_with_progress(
    embedder: Any,
    vector_store: Any,
    chunks: list[dict[str, Any]],
    indexing_batch_size: int,
    progress_callback: Callable[[str, float], None] | None,
) -> None:
    total = len(chunks)

    def update_embedding_progress(completed: int, total: int) -> None:
        if total <= 0:
            return
        value = 0.55 + (0.40 * completed / total)
        _notify_progress(
            progress_callback,
            f"Encoding and indexing chunks {completed}/{total}...",
            min(value, 0.95),
        )

    completed = 0
    for batch in _batches(chunks, indexing_batch_size):
        batch_start = completed + 1
        batch_end = completed + len(batch)
        log_event(
            "index_batch_start",
            batch_start=batch_start,
            batch_end=batch_end,
            total=total,
            **text_stats([chunk.get("text", "") for chunk in batch]),
        )
        _notify_progress(
            progress_callback,
            f"Encoding chunks {batch_start}-{batch_end}/{total}...",
            0.55 + (0.40 * completed / total),
        )

        texts = [_embedding_text(chunk) for chunk in batch]
        if _supports_progress_callback(embedder):
            embeddings = embedder.encode(
                texts,
                progress_callback=lambda done, _total, offset=completed: update_embedding_progress(
                    offset + done,
                    total,
                ),
            )
        else:
            embeddings = embedder.encode(texts)

        _notify_progress(
            progress_callback,
            f"Writing chunks {batch_start}-{batch_end}/{total} to vector database...",
            0.55 + (0.40 * batch_end / total),
        )
        log_event("vector_store_batch_write_start", batch_start=batch_start, batch_end=batch_end)
        vector_store.add_chunks(batch, embeddings)
        log_event("vector_store_batch_write_done", batch_start=batch_start, batch_end=batch_end)
        completed = batch_end
        update_embedding_progress(completed, total)


def _batches(items: list[Any], batch_size: int) -> list[list[Any]]:
    return [items[index : index + batch_size] for index in range(0, len(items), batch_size)]


def _should_retry_refusal(question: str, retrieved_chunks: list[dict[str, Any]]) -> bool:
    return estimate_context_support(question, retrieved_chunks) != NO_SUPPORT



def _supports_progress_callback(embedder: Any) -> bool:
    try:
        signature = inspect.signature(embedder.encode)
    except (TypeError, ValueError):
        return False
    return "progress_callback" in signature.parameters


def _clear_retriever_cache(retriever: Any) -> None:
    clear_cache = getattr(retriever, "clear_cache", None)
    if callable(clear_cache):
        clear_cache()
