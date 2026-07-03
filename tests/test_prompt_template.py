from app.prompt_template import (
    REFUSAL_EN,
    REFUSAL_VI,
    answer_needs_repair,
    answer_language_mismatch,
    build_answer_required_prompt,
    build_answer_repair_prompt,
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
    assert "Do not attach citations to insufficient-context" in prompt
    assert "instead of refusing too early" in prompt
    assert "Sinh vi\u00ean c\u1ea7n bao nhi\u00eau t\u00edn ch\u1ec9?" in prompt
    assert "Previous question" in prompt


def test_build_prompt_for_unsupported_claim_requires_clear_behavior() -> None:
    prompt = build_prompt(
        "Does the document claim everyone is eligible?",
        [
            {
                "chunk_id": "chunk-1",
                "text": "Eligibility depends on meeting the stated requirements.",
                "metadata": {"file_name": "sample.pdf", "page": 2},
            }
        ],
        expected_behavior="state_not_supported",
    )

    assert "The document does not support this claim." in prompt


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
    assert normalize_answer("(The answer uses evidence. [1])", "What?", chunks) == (
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


def test_normalize_answer_converts_document_no_info_answer_to_refusal() -> None:
    chunks = [{"chunk_id": "c1", "metadata": {"file_name": "paper.pdf", "page": 1}}]

    assert normalize_answer(
        "No, the document does not address cryptocurrency markets. [1]",
        "Does the paper estimate effects on cryptocurrency markets?",
        chunks,
    ) == REFUSAL_EN
    assert normalize_answer(
        "No, I did not find this information in the provided document. [1]",
        "Does the paper study Vietnam?",
        chunks,
    ) == REFUSAL_EN


def test_normalize_answer_keeps_supported_negative_finding() -> None:
    chunks = [{"chunk_id": "c1", "metadata": {"file_name": "paper.pdf", "page": 1}}]

    assert normalize_answer(
        "No, the paper does not find evidence of formal-to-informal spillovers. [1]",
        "Does the paper find evidence of formal-to-informal spillovers?",
        chunks,
    ) == "No, the paper does not find evidence of formal-to-informal spillovers. [1]"


def test_build_language_repair_prompt_preserves_citation_contract() -> None:
    prompt = build_language_repair_prompt(
        "Theo tài liệu, bài báo nghiên cứu lương tối thiểu. [1]",
        "What does the paper study?",
    )

    assert "English only" in prompt
    assert "Preserve compact numeric citations" in prompt
    assert "[1]" in prompt


def test_answer_needs_repair_detects_citation_only_and_meta_answers() -> None:
    assert answer_needs_repair("[1, 2]") is True
    assert answer_needs_repair("Note: I've answered directly first. [1]") is True
    assert answer_needs_repair("Citations: [1]\nFile: paper.pdf\nPage: 2\nContent: text") is True
    assert answer_needs_repair("[1] PerezPerez2020.pdf, Page 10") is True
    assert answer_needs_repair("(Translation: The answer is available.) [1]") is True
    assert answer_needs_repair("The paper studies minimum wage effects. [1]") is False
    assert answer_needs_repair(REFUSAL_EN) is False


def test_build_answer_repair_prompt_includes_context_and_invalid_answer() -> None:
    chunks = [
        {
            "chunk_id": "c1",
            "text": "The paper studies formal and informal wages.",
            "metadata": {"file_name": "paper.pdf", "page": 1},
        }
    ]

    prompt = build_answer_repair_prompt("[1, 2]", "What are the outcomes?", chunks)

    assert "previous answer was invalid" in prompt
    assert "Do not return only citations" in prompt
    assert "Do not output source metadata" in prompt
    assert "The paper studies formal and informal wages." in prompt
    assert "Invalid previous answer:" in prompt


def test_build_answer_repair_prompt_warns_against_premature_refusal() -> None:
    chunks = [
        {
            "chunk_id": "c1",
            "text": "The paper explains sample splits by education and sex.",
            "metadata": {"file_name": "paper.pdf", "page": 1},
        }
    ]

    prompt = build_answer_repair_prompt(
        REFUSAL_EN,
        "Why does the paper split the sample by sex?",
        chunks,
    )

    assert "answer from that evidence instead of refusing too early" in prompt
    assert "Do not attach citations to insufficient-context" in prompt


def test_build_answer_required_prompt_blocks_no_information_refusal() -> None:
    chunks = [
        {
            "chunk_id": "c1",
            "text": "The paper describes the main policy question.",
            "metadata": {"file_name": "paper.pdf", "page": 1},
        }
    ]

    prompt = build_answer_required_prompt(
        REFUSAL_EN,
        "What is the main policy question?",
        chunks,
    )

    assert "retrieved Context contains relevant evidence" in prompt
    assert "Do not say that the information is not found" in prompt
