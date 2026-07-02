from app.keyword_search import bm25_search, lexical_similarity, tokenize


def test_tokenize_normalizes_light_english_plural_forms() -> None:
    tokens = tokenize("Data sources, wage effects, and empirical studies")

    assert "source" in tokens
    assert "wage" in tokens
    assert "effect" in tokens
    assert "study" in tokens


def test_tokenize_supports_english_and_vietnamese() -> None:
    tokens = tokenize("AI Intern cần Gemini API.")

    assert tokens == ["ai", "intern", "cần", "gemini", "api"]


def test_bm25_search_prioritizes_keyword_matches() -> None:
    chunks = [
        {"chunk_id": "a", "text": "Internship policy and allowance"},
        {"chunk_id": "b", "text": "Gemini API quota and model limits"},
    ]

    results = bm25_search("Gemini quota", chunks, top_k=1)

    assert results[0]["chunk_id"] == "b"
    assert 0 < results[0]["keyword_score"] <= 1


def test_bm25_search_matches_singular_query_to_plural_document_terms() -> None:
    chunks = [
        {"chunk_id": "data", "text": "The paper uses Data Sources from household surveys."},
        {"chunk_id": "other", "text": "The paper discusses robustness checks."},
    ]

    results = bm25_search("data source", chunks, top_k=1)

    assert results[0]["chunk_id"] == "data"


def test_lexical_similarity_returns_token_overlap_ratio() -> None:
    score = lexical_similarity("Gemini API quota", "Gemini API limits")

    assert 0 < score < 1
