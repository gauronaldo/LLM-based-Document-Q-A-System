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


class FakeHybridVectorStore(FakeVectorStore):
    def __init__(self, results=None, all_chunks=None, error: Exception | None = None):
        super().__init__(results=results, error=error)
        self.all_chunks = all_chunks or []
        self.get_all_calls = 0

    def get_all(self):
        self.get_all_calls += 1
        return self.all_chunks


class FakeReranker:
    def __init__(self):
        self.calls = []

    def rerank(self, query, candidates, top_k):
        self.calls.append((query, candidates, top_k))
        return candidates[:top_k]


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
    assert vector_store.seen_top_k == 40
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

    assert vector_store.seen_top_k == 56


def test_retrieve_can_override_candidate_k() -> None:
    vector_store = FakeVectorStore(results=[{"chunk_id": "high", "score": 0.9}])
    retriever = Retriever(vector_store, FakeEmbedder(), top_k=3)

    retriever.retrieve("Explain", top_k=2, candidate_k=6)

    assert vector_store.seen_top_k == 6


def test_hybrid_search_adds_keyword_matches() -> None:
    vector_store = FakeHybridVectorStore(
        results=[{"chunk_id": "semantic", "text": "general internship", "score": 0.7, "metadata": {}}],
        all_chunks=[
            {"chunk_id": "keyword", "text": "Gemini API quota policy", "metadata": {}},
            {"chunk_id": "other", "text": "general internship policy", "metadata": {}},
        ],
    )
    retriever = Retriever(vector_store, FakeEmbedder(), top_k=2, similarity_threshold=0.0)

    results = retriever.retrieve("Gemini quota")

    assert "keyword" in [result["chunk_id"] for result in results]


def test_hybrid_search_reuses_cached_all_chunks() -> None:
    vector_store = FakeHybridVectorStore(
        results=[{"chunk_id": "semantic", "text": "general internship", "score": 0.7, "metadata": {}}],
        all_chunks=[{"chunk_id": "keyword", "text": "Gemini API quota policy", "metadata": {}}],
    )
    retriever = Retriever(vector_store, FakeEmbedder(), top_k=1, similarity_threshold=0.0)

    retriever.retrieve("Gemini quota")
    retriever.retrieve("Gemini quota")

    assert vector_store.get_all_calls == 1


def test_clear_cache_reloads_all_chunks() -> None:
    vector_store = FakeHybridVectorStore(
        results=[{"chunk_id": "semantic", "text": "general internship", "score": 0.7, "metadata": {}}],
        all_chunks=[{"chunk_id": "keyword", "text": "Gemini API quota policy", "metadata": {}}],
    )
    retriever = Retriever(vector_store, FakeEmbedder(), top_k=1, similarity_threshold=0.0)

    retriever.retrieve("Gemini quota")
    retriever.clear_cache()
    retriever.retrieve("Gemini quota")

    assert vector_store.get_all_calls == 2


def test_section_aware_retrieval_matches_section_title_without_query_expansion() -> None:
    vector_store = FakeHybridVectorStore(
        results=[],
        all_chunks=[
            {
                "chunk_id": "conclusion",
                "text": "The final section summarizes the wage and employment findings.",
                "metadata": {"section_title": "Concluding Remarks"},
            },
            {
                "chunk_id": "intro",
                "text": "The paper introduces the research question.",
                "metadata": {"section_title": "Introduction"},
            },
        ],
    )
    retriever = Retriever(vector_store, FakeEmbedder(), top_k=1, similarity_threshold=0.3)

    results = retriever.retrieve("give me the conclusion")

    assert [result["chunk_id"] for result in results] == ["conclusion"]
    assert results[0]["metadata"]["section_title"] == "Concluding Remarks"
    assert "Concluding Remarks" not in results[0]["text"]


def test_section_match_can_pass_default_threshold_for_natural_conclusion_question() -> None:
    vector_store = FakeHybridVectorStore(
        results=[
            {
                "chunk_id": "conclusion",
                "text": "The paper concludes by summarizing wage and employment findings.",
                "score": 0.26,
                "metadata": {"section_title": "Concluding Remarks"},
            },
            {
                "chunk_id": "method",
                "text": "The empirical strategy estimates effects across markets.",
                "score": 0.32,
                "metadata": {"section_title": "Empirical Strategy"},
            },
        ],
        all_chunks=[
            {
                "chunk_id": "conclusion",
                "text": "The paper concludes by summarizing wage and employment findings.",
                "metadata": {"section_title": "Concluding Remarks"},
            },
            {
                "chunk_id": "method",
                "text": "The empirical strategy estimates effects across markets.",
                "metadata": {"section_title": "Empirical Strategy"},
            },
        ],
    )
    retriever = Retriever(vector_store, FakeEmbedder(), top_k=1, similarity_threshold=0.3)

    results = retriever.retrieve("What is the overall conclusion of the paper?")

    assert [result["chunk_id"] for result in results] == ["conclusion"]


def test_retrieval_handles_singular_plural_and_vietnamese_question_words() -> None:
    vector_store = FakeHybridVectorStore(
        results=[],
        all_chunks=[
            {
                "chunk_id": "data-sources",
                "text": "The study combines household survey data and administrative wage records.",
                "metadata": {"section_title": "Data Sources"},
            },
            {
                "chunk_id": "methods",
                "text": "The empirical strategy estimates wage and employment effects.",
                "metadata": {"section_title": "Empirical Strategy"},
            },
        ],
    )
    retriever = Retriever(vector_store, FakeEmbedder(), top_k=1, similarity_threshold=0.3)

    results = retriever.retrieve("data source tới từ đâu?")

    assert [result["chunk_id"] for result in results] == ["data-sources"]


def test_entity_aware_retrieval_prefers_exact_luu_bi_over_luu_bieu() -> None:
    vector_store = FakeVectorStore(
        results=[
            {
                "chunk_id": "wrong-luu-bieu",
                "text": "Lưu Biểu và bảy danh sĩ khác được gọi là Giang hạ bát tuấn.",
                "score": 0.92,
                "metadata": {},
            },
            {
                "chunk_id": "correct-luu-bi",
                "text": "Lưu Bị cùng Quan Vũ và Trương Phi kết nghĩa anh em.",
                "score": 0.35,
                "metadata": {},
            },
        ]
    )
    retriever = Retriever(vector_store, FakeEmbedder(), top_k=1, similarity_threshold=0.3)

    results = retriever.retrieve("Ai là anh em kết nghĩa của Lưu Bị?", auto=True)

    assert [result["chunk_id"] for result in results] == ["correct-luu-bi"]


def test_entity_aware_retrieval_refuses_when_entity_is_missing() -> None:
    vector_store = FakeVectorStore(
        results=[
            {
                "chunk_id": "wrong-truong-phi",
                "text": "Trương Phi còn được gọi là Dực Đức.",
                "score": 0.95,
                "metadata": {},
            }
        ]
    )
    retriever = Retriever(vector_store, FakeEmbedder(), top_k=1, similarity_threshold=0.0)

    results = retriever.retrieve("Các tên gọi khác của Quan Vũ là gì?", auto=True)

    assert results == []


def test_parent_context_expands_returned_text() -> None:
    vector_store = FakeVectorStore(
        results=[
            {
                "chunk_id": "child",
                "text": "short child",
                "score": 0.9,
                "metadata": {"parent_text": "Long parent section with short child and details."},
            }
        ]
    )
    retriever = Retriever(vector_store, FakeEmbedder(), top_k=1, similarity_threshold=0.0)

    results = retriever.retrieve("explain child details with enough context")

    assert results[0]["text"] == "Long parent section with short child and details."
    assert results[0]["matched_text"] == "short child"


def test_parent_context_not_expanded_for_simple_fact_question() -> None:
    vector_store = FakeVectorStore(
        results=[
            {
                "chunk_id": "child",
                "text": "short child",
                "score": 0.9,
                "metadata": {"parent_text": "Long parent section with short child and details."},
            }
        ]
    )
    retriever = Retriever(vector_store, FakeEmbedder(), top_k=1, similarity_threshold=0.0)

    results = retriever.retrieve("child")

    assert results[0]["text"] == "short child"


def test_auto_retrieval_skips_rerank_for_confident_simple_query() -> None:
    reranker = FakeReranker()
    vector_store = FakeVectorStore(
        results=[
            {"chunk_id": "best", "text": "salary amount", "score": 0.95, "metadata": {}},
            {"chunk_id": "other", "text": "unrelated", "score": 0.2, "metadata": {}},
        ]
    )
    retriever = Retriever(
        vector_store,
        FakeEmbedder(),
        top_k=5,
        similarity_threshold=0.0,
        reranker=reranker,
    )

    results = retriever.retrieve("salary?", auto=True)

    assert [result["chunk_id"] for result in results] == ["best"]
    assert reranker.calls == []
    assert vector_store.seen_top_k == 30


def test_auto_retrieval_uses_rerank_when_scores_are_close() -> None:
    reranker = FakeReranker()
    vector_store = FakeVectorStore(
        results=[
            {"chunk_id": "best", "text": "salary amount", "score": 0.71, "metadata": {}},
            {"chunk_id": "close", "text": "salary range", "score": 0.68, "metadata": {}},
        ]
    )
    retriever = Retriever(
        vector_store,
        FakeEmbedder(),
        top_k=5,
        similarity_threshold=0.0,
        reranker=reranker,
    )

    retriever.retrieve("salary?", auto=True)

    assert len(reranker.calls) == 1


def test_retriever_refuses_yes_no_question_with_missing_distinctive_term() -> None:
    vector_store = FakeVectorStore(
        results=[
            {
                "chunk_id": "minimum-wage",
                "text": "The paper studies minimum wage effects on formal wages.",
                "score": 0.8,
                "metadata": {},
            }
        ]
    )
    retriever = Retriever(vector_store, FakeEmbedder(), top_k=1, similarity_threshold=0.0)

    results = retriever.retrieve("Does the paper study cryptocurrency markets?")

    assert results == []


def test_retriever_refuses_yes_no_question_even_with_broad_topic_overlap() -> None:
    vector_store = FakeVectorStore(
        results=[
            {
                "chunk_id": "minimum-wage",
                "text": "The paper studies minimum wage effects across labor markets.",
                "score": 0.8,
                "metadata": {},
            }
        ]
    )
    retriever = Retriever(vector_store, FakeEmbedder(), top_k=1, similarity_threshold=0.0)

    results = retriever.retrieve("Does the paper study the effect of minimum wages on cryptocurrency markets?")

    assert results == []


def test_retriever_refuses_yes_no_question_with_unsupported_extra_claims() -> None:
    vector_store = FakeVectorStore(
        results=[
            {
                "chunk_id": "minimum-wage",
                "text": "The paper estimates minimum wage effects on formal wages in Colombia.",
                "score": 0.8,
                "metadata": {},
            }
        ]
    )
    retriever = Retriever(vector_store, FakeEmbedder(), top_k=1, similarity_threshold=0.0)

    results = retriever.retrieve(
        "Does the paper estimate minimum wage effects on firm profits or product prices in Colombia?"
    )

    assert results == []


def test_retriever_refuses_yes_no_question_when_claim_terms_are_split_across_chunks() -> None:
    vector_store = FakeVectorStore(
        results=[
            {
                "chunk_id": "minimum-wage",
                "text": "The paper estimates minimum wage effects on formal wages in Colombia.",
                "score": 0.8,
                "metadata": {},
            },
            {
                "chunk_id": "references",
                "text": "Related studies discuss financial markets and product prices.",
                "score": 0.7,
                "metadata": {},
            },
        ]
    )
    retriever = Retriever(vector_store, FakeEmbedder(), top_k=2, similarity_threshold=0.0)

    results = retriever.retrieve(
        "Does the paper estimate minimum wage effects on firm profits or product prices in Colombia?"
    )

    assert results == []


def test_retriever_allows_yes_no_question_when_claim_terms_are_supported() -> None:
    vector_store = FakeVectorStore(
        results=[
            {
                "chunk_id": "formal-wages",
                "text": "The paper estimates minimum wage effects on formal wages in Colombia.",
                "score": 0.8,
                "metadata": {},
            }
        ]
    )
    retriever = Retriever(vector_store, FakeEmbedder(), top_k=1, similarity_threshold=0.0)

    results = retriever.retrieve("Does the paper estimate minimum wage effects on formal wages in Colombia?")

    assert [result["chunk_id"] for result in results] == ["formal-wages"]


def test_mmr_preserves_top_reranked_candidate() -> None:
    candidates = [
        {"chunk_id": "best", "text": "alpha beta", "score": 0.9},
        {"chunk_id": "second", "text": "alpha gamma", "score": 0.8},
        {"chunk_id": "third", "text": "delta epsilon", "score": 0.7},
    ]

    selected = Retriever._select_with_preserved_top_candidate("alpha", candidates, top_k=2)

    assert selected[0]["chunk_id"] == "best"
    assert len(selected) == 2


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
