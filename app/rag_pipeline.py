"""Main RAG orchestration pipeline."""

from __future__ import annotations

from typing import Any

from app.config import AppConfig, get_config
from app.document_loader import DocumentLoader
from app.embedding_model import EmbeddingModel
from app.llm_client import LLMClient
from app.prompt_template import build_prompt, extract_sources, refusal_for_question
from app.query_intent import detect_query_intent, retrieval_plan_for_intent
from app.retriever import Retriever
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
    ):
        self.document_loader = document_loader
        self.preprocessor = preprocessor
        self.splitter = splitter
        self.embedder = embedder
        self.vector_store = vector_store
        self.retriever = retriever
        self.llm = llm

    def ingest_document(self, file_path: str, file_name: str, file_id: str) -> dict[str, Any]:
        """Load, clean, chunk, embed, and store a document."""

        documents = self.document_loader.load(file_path, file_name, file_id)
        processed_documents = []

        for document in documents:
            clean_text = self.preprocessor.clean(document["text"])
            if clean_text:
                processed_documents.append(
                    {
                        "text": clean_text,
                        "metadata": document["metadata"],
                    }
                )

        chunks = self.splitter.split(processed_documents)
        if not chunks:
            return {
                "status": "empty",
                "file_name": file_name,
                "num_documents": len(documents),
                "num_chunks": 0,
            }

        embeddings = self.embedder.encode([chunk["text"] for chunk in chunks])
        self.vector_store.add_chunks(chunks, embeddings)

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
        retrieval_plan = retrieval_plan_for_intent(
            intent=intent,
            default_top_k=self.retriever.top_k,
            default_threshold=self.retriever.similarity_threshold,
        )
        retrieved_chunks = self.retriever.retrieve(
            clean_question,
            top_k=retrieval_plan.top_k,
            similarity_threshold=retrieval_plan.similarity_threshold,
        )

        if not retrieved_chunks:
            return {
                "answer": refusal_for_question(clean_question),
                "sources": [],
                "retrieved_chunks": [],
            }

        prompt = build_prompt(
            question=clean_question,
            retrieved_chunks=retrieved_chunks,
            chat_history=chat_history,
            intent=intent,
        )
        answer = self.llm.generate(prompt)

        return {
            "answer": answer,
            "sources": extract_sources(retrieved_chunks),
            "retrieved_chunks": retrieved_chunks,
            "intent": intent,
        }

    def reset(self) -> None:
        """Reset the underlying vector store."""

        self.vector_store.reset()


def create_default_pipeline(
    config: AppConfig | None = None,
    top_k: int | None = None,
    similarity_threshold: float | None = None,
) -> RAGPipeline:
    """Create a production pipeline from application configuration."""

    config = config or get_config()
    embedder = EmbeddingModel(config.embedding_model)
    vector_store = VectorStore(config.chroma_persist_dir, config.chroma_collection_name)
    retriever = Retriever(
        vector_store=vector_store,
        embedder=embedder,
        top_k=top_k or config.top_k,
        similarity_threshold=(
            config.similarity_threshold
            if similarity_threshold is None
            else similarity_threshold
        ),
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
        splitter=TextSplitter(),
        embedder=embedder,
        vector_store=vector_store,
        retriever=retriever,
        llm=llm,
    )
