from app.reranker import Reranker


def test_lexical_reranker_prefers_query_overlap() -> None:
    chunks = [
        {"chunk_id": "low", "text": "general internship policy", "score": 0.9},
        {"chunk_id": "high", "text": "Gemini API quota limit", "score": 0.4},
    ]

    results = Reranker().rerank("Gemini quota", chunks, top_k=1)

    assert results[0]["chunk_id"] == "high"
    assert results[0]["rerank_score"] > 0


def test_reranker_returns_empty_for_invalid_top_k() -> None:
    assert Reranker().rerank("question", [{"chunk_id": "x", "text": "x"}], top_k=0) == []


def test_lexical_reranker_uses_section_title_metadata() -> None:
    chunks = [
        {
            "chunk_id": "body",
            "text": "The paper discusses minimum wage methods.",
            "metadata": {"section_title": "Empirical Strategy"},
            "score": 0.8,
        },
        {
            "chunk_id": "conclusion",
            "text": "The paper summarizes the findings.",
            "metadata": {"section_title": "Concluding Remarks"},
            "score": 0.3,
        },
    ]

    results = Reranker().rerank("conclusion of the paper", chunks, top_k=1)

    assert results[0]["chunk_id"] == "conclusion"
