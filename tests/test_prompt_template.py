from app.prompt_template import (
    REFUSAL_EN,
    REFUSAL_VI,
    build_chat_history,
    build_context,
    build_prompt,
    extract_sources,
    refusal_for_question,
)


def test_build_context_formats_sources() -> None:
    chunks = [
        {
            "chunk_id": "abc_page_1_chunk_0",
            "text": "Sinh vi\u00ean c\u1ea7n 65% t\u00edn ch\u1ec9.",
            "metadata": {"file_name": "quy_che.pdf", "page": 1},
        }
    ]

    context = build_context(chunks)

    assert "[1]" in context
    assert "File: quy_che.pdf" in context
    assert "Page: 1" in context
    assert "Chunk ID: abc_page_1_chunk_0" in context
    assert "Sinh vi\u00ean c\u1ea7n 65% t\u00edn ch\u1ec9." in context


def test_build_chat_history_formats_recent_messages() -> None:
    history = build_chat_history(
        [
            {"role": "user", "content": "Previous question"},
            {"role": "assistant", "content": "Previous answer"},
        ]
    )

    assert "User: Previous question" in history
    assert "Assistant: Previous answer" in history


def test_build_prompt_contains_conversational_grounding_rules() -> None:
    prompt = build_prompt(
        "Sinh vi\u00ean c\u1ea7n bao nhi\u00eau t\u00edn ch\u1ec9?",
        [
            {
                "chunk_id": "chunk-1",
                "text": "Sinh vi\u00ean c\u1ea7n 65% t\u00edn ch\u1ec9.",
                "metadata": {"file_name": "sample.pdf", "page": 2},
            }
        ],
        chat_history=[{"role": "user", "content": "Previous question"}],
        intent="explanation",
    )

    assert "document-grounded chatbot" in prompt
    assert "Synthesize the answer in your own words" in prompt
    assert REFUSAL_VI in prompt
    assert REFUSAL_EN in prompt
    assert "Sinh vi\u00ean c\u1ea7n bao nhi\u00eau t\u00edn ch\u1ec9?" in prompt
    assert "Previous question" in prompt


def test_extract_sources_keeps_context_order_and_labels() -> None:
    chunks = [
        {"chunk_id": "c1", "metadata": {"file_name": "a.pdf", "page": 1}},
        {"chunk_id": "c1", "metadata": {"file_name": "a.pdf", "page": 1}},
        {"chunk_id": "c2", "metadata": {"file_name": "a.pdf", "page": 2}},
    ]

    assert extract_sources(chunks) == [
        {"source_id": 1, "file_name": "a.pdf", "page": 1, "chunk_id": "c1"},
        {"source_id": 2, "file_name": "a.pdf", "page": 1, "chunk_id": "c1"},
        {"source_id": 3, "file_name": "a.pdf", "page": 2, "chunk_id": "c2"},
    ]


def test_refusal_for_question_matches_likely_language() -> None:
    assert refusal_for_question("Sinh vi\u00ean c\u1ea7n g\u00ec?") == REFUSAL_VI
    assert refusal_for_question("What are the requirements?") == REFUSAL_EN
