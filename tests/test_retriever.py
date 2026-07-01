import pytest

from app.retriever import Retriever, RetrieverError


class FakeEmbedder:
    def __init__(self, embeddings=None, error: Exception | None = None):
        self.embeddings = embeddings or [[0.1, 0.2]]
        self.error = error
        self.seen_texts = None

    def encode(self, texts: list[str]):
        self.seen_texts = texts
        if self.error:
            raise self.error
        return self.embeddings


class FakeVectorStore:
    def __init__(self, results=None, error: Exception | None = None):
        self.results = results or []
        self.error = error
        self.seen_embedding = None
        self.seen_top_k = None

    def search(self, query_embedding: list[float], top_k: int):
        self.seen_embedding = query_embedding
        self.seen_top_k = top_k
        if self.error:
            raise self.error
        return self.results


def test_retrieve_encodes_question_and_filters_by_threshold() -> None:
    embedder = FakeEmbedder()
    vector_store = FakeVectorStore(
        results=[
            {"chunk_id": "low", "score": 0.2},
            {"chunk_id": "high", "score": 0.9},
            {"chunk_id": "mid", "score": 0.5},
        ]
    )
    retriever = Retriever(vector_store, embedder, top_k=3, similarity_threshold=0.4)

    results = retriever.retrieve("  Sinh viên cần bao nhiêu tín chỉ?  ")

    assert embedder.seen_texts == ["Sinh viên cần bao nhiêu tín chỉ?"]
    assert vector_store.seen_embedding == [0.1, 0.2]
    assert vector_store.seen_top_k == 3
    assert [result["chunk_id"] for result in results] == ["high", "mid"]


def test_retrieve_can_override_similarity_threshold() -> None:
    vector_store = FakeVectorStore(results=[{"chunk_id": "low", "score": 0.0}])
    retriever = Retriever(vector_store, FakeEmbedder(), similarity_threshold=0.5)

    results = retriever.retrieve("Summarize", similarity_threshold=0.0)

    assert [result["chunk_id"] for result in results] == ["low"]


def test_retrieve_can_override_top_k() -> None:
    vector_store = FakeVectorStore(results=[{"chunk_id": "high", "score": 0.9}])
    retriever = Retriever(vector_store, FakeEmbedder(), top_k=3)

    retriever.retrieve("Explain", top_k=7)

    assert vector_store.seen_top_k == 7


def test_retrieve_returns_empty_for_blank_question() -> None:
    retriever = Retriever(FakeVectorStore(), FakeEmbedder())

    assert retriever.retrieve("   ") == []


def test_retrieve_returns_empty_when_embedder_returns_no_embedding() -> None:
    retriever = Retriever(FakeVectorStore(), FakeEmbedder(embeddings=[]))

    assert retriever.retrieve("Câu hỏi") == []


def test_retrieve_wraps_embedder_errors() -> None:
    retriever = Retriever(FakeVectorStore(), FakeEmbedder(error=RuntimeError("boom")))

    with pytest.raises(RetrieverError, match="Failed to encode question"):
        retriever.retrieve("Câu hỏi")


def test_retrieve_wraps_vector_store_errors() -> None:
    retriever = Retriever(FakeVectorStore(error=RuntimeError("boom")), FakeEmbedder())

    with pytest.raises(RetrieverError, match="Failed to retrieve chunks"):
        retriever.retrieve("Câu hỏi")


def test_retriever_validates_settings() -> None:
    with pytest.raises(ValueError, match="top_k"):
        Retriever(FakeVectorStore(), FakeEmbedder(), top_k=0)

    with pytest.raises(ValueError, match="similarity_threshold"):
        Retriever(FakeVectorStore(), FakeEmbedder(), similarity_threshold=1.5)
