from app.prompt_template import (
    REFUSAL_EN,
    REFUSAL_VI,
    answer_language_mismatch,
    build_chat_history,
    build_context,
    build_language_repair_prompt,
    build_prompt,
    extract_sources,
    normalize_answer,
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


def test_build_context_compresses_long_chunks() -> None:
    chunks = [
        {
            "chunk_id": "long",
            "text": " ".join(f"word{index}" for index in range(500)),
            "metadata": {"file_name": "paper.pdf", "page": 1},
        }
    ]

    context = build_context(chunks, max_chunk_chars=500, max_total_chars=800)

    assert "[...context shortened...]" in context
    assert len(context) < 800


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
    assert "Match the language of the current User Question exactly." in prompt
    assert "If the current User Question is English, answer in English." in prompt
    assert "Do not include translations, bilingual explanations" in prompt
    assert "Never use previous assistant answers as evidence." in prompt
    assert "similarly named people/entities" in prompt
    assert "Do not write citation labels as [Source 1]" in prompt
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


def test_normalize_answer_rewrites_source_labels_and_adds_missing_citation() -> None:
    chunks = [{"chunk_id": "c1", "metadata": {"file_name": "paper.pdf", "page": 1}}]

    assert normalize_answer("The answer uses evidence. [Source 1]", "What?", chunks) == (
        "The answer uses evidence. [1]"
    )
    assert normalize_answer("The answer uses evidence.", "What?", chunks) == (
        "The answer uses evidence. [1]"
    )


def test_normalize_answer_removes_extra_vietnamese_translation_for_english_question() -> None:
    chunks = [{"chunk_id": "c1", "metadata": {"file_name": "paper.pdf", "page": 1}}]
    answer = (
        "The paper studies minimum wage effects. [1]\n\n"
        "Theo tài liệu, bài báo nghiên cứu tác động của lương tối thiểu. [1]"
    )

    assert normalize_answer(answer, "What does the paper study?", chunks) == (
        "The paper studies minimum wage effects. [1]"
    )


def test_answer_language_mismatch_detects_vietnamese_answer_to_english_question() -> None:
    answer = "Theo tài liệu, bài báo nghiên cứu lương tối thiểu. [1]"

    assert answer_language_mismatch(answer, "What does the paper study?") is True
    assert answer_language_mismatch(answer, "Bài báo nghiên cứu gì?") is False


def test_build_language_repair_prompt_preserves_citation_contract() -> None:
    prompt = build_language_repair_prompt(
        "Theo tài liệu, bài báo nghiên cứu lương tối thiểu. [1]",
        "What does the paper study?",
    )

    assert "English only" in prompt
    assert "Preserve compact numeric citations" in prompt
    assert "[1]" in prompt
