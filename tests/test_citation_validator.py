from app.citation_validator import post_check_citations


def test_post_check_citations_repairs_weak_citation_to_stronger_chunk() -> None:
    chunks = [
        {"chunk_id": "a", "text": "The document discusses project goals.", "metadata": {}},
        {"chunk_id": "b", "text": "The source data comes from customer support tickets.", "metadata": {}},
    ]

    answer = post_check_citations(
        "The source data comes from customer support tickets. [1]",
        "Where does the source data come from?",
        chunks,
    )

    assert answer == "The source data comes from customer support tickets. [2]"


def test_post_check_citations_prefers_distinctive_sentence_terms() -> None:
    chunks = [
        {
            "chunk_id": "abstract",
            "text": "The paper estimates minimum wage effects on formal wages.",
            "metadata": {},
        },
        {
            "chunk_id": "compliance",
            "text": "The formal wage estimates are smaller than the mechanical increase implied by full compliance.",
            "metadata": {},
        },
    ]

    answer = post_check_citations(
        "The formal wage estimates are smaller than full compliance. [1]",
        "How should the estimates be interpreted relative to full compliance?",
        chunks,
    )

    assert answer == "The formal wage estimates are smaller than full compliance. [2]"


def test_post_check_citations_keeps_supported_citation() -> None:
    chunks = [
        {"chunk_id": "a", "text": "The source data comes from customer support tickets.", "metadata": {}},
        {"chunk_id": "b", "text": "The document discusses project goals.", "metadata": {}},
    ]

    answer = post_check_citations(
        "The source data comes from customer support tickets. [1]",
        "Where does the source data come from?",
        chunks,
    )

    assert answer == "The source data comes from customer support tickets. [1]"


def test_post_check_citations_adds_missing_citation_when_support_is_clear() -> None:
    chunks = [
        {
            "chunk_id": "data",
            "text": "The source data comes from household surveys and administrative wage records.",
            "metadata": {},
        }
    ]

    answer = post_check_citations(
        "The source data comes from household surveys and administrative wage records.",
        "Where does the source data come from?",
        chunks,
    )

    assert answer == "The source data comes from household surveys and administrative wage records [1]."


def test_post_check_citations_does_not_add_weak_missing_citation() -> None:
    chunks = [
        {"chunk_id": "data", "text": "The document discusses wage records.", "metadata": {}}
    ]

    answer = post_check_citations(
        "It is useful.",
        "Where does the source data come from?",
        chunks,
    )

    assert answer == "It is useful."


def test_post_check_citations_strips_refusal_citation() -> None:
    chunks = [
        {
            "chunk_id": "data",
            "text": "The document discusses wage records.",
            "metadata": {},
        }
    ]

    answer = post_check_citations(
        "I could not find this information in the provided document. [1]",
        "Does the paper study cryptocurrency markets?",
        chunks,
    )

    assert answer == "I could not find this information in the provided document."
