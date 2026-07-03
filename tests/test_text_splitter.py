from app.text_splitter import TextSplitter


def test_splitter_preserves_metadata_and_builds_chunk_ids() -> None:
    documents = [
        {
            "text": "Điều 1. Quy định chung.\n\nĐiều 2. Điều kiện thực tập.",
            "metadata": {
                "file_id": "abc123",
                "file_name": "quy_che.pdf",
                "page": 3,
            },
        }
    ]

    chunks = TextSplitter(chunk_size=40, chunk_overlap=5).split(documents)

    assert chunks
    assert chunks[0]["chunk_id"] == "abc123_page_3_chunk_0"
    assert chunks[0]["metadata"]["file_name"] == "quy_che.pdf"
    assert chunks[0]["metadata"]["page"] == 3
    assert chunks[0]["metadata"]["chunk_index"] == 0


def test_splitter_skips_empty_documents() -> None:
    chunks = TextSplitter().split([{"text": " ", "metadata": {"file_id": "x"}}])

    assert chunks == []


def test_parent_context_is_capped_for_large_sections() -> None:
    text = "Heading\n\n" + ("long parent context " * 50)

    chunks = TextSplitter(
        chunk_size=80,
        chunk_overlap=0,
        parent_context_max_chars=120,
    ).split([{"text": text, "metadata": {"file_id": "doc", "page": 1}}])

    assert chunks
    assert len(chunks[0]["metadata"]["parent_text"]) <= 120


def test_splitter_builds_docx_paragraph_chunk_id() -> None:
    documents = [
        {
            "text": "Điều kiện thực tập.",
            "metadata": {
                "file_id": "docx1",
                "file_name": "sample.docx",
                "page": None,
                "paragraph_index": 2,
            },
        }
    ]

    chunks = TextSplitter().split(documents)

    assert chunks[0]["chunk_id"] == "docx1_paragraph_2_chunk_0"
    assert chunks[0]["metadata"]["paragraph_index"] == 2


def test_overlap_does_not_start_mid_word() -> None:
    splitter = TextSplitter(chunk_size=35, chunk_overlap=10)

    chunk = splitter._with_overlap(
        "Candidates receive internship support",
        "They can become full-time employees.",
    )

    assert chunk.startswith("support\n\n")
    assert not chunk.startswith("pport")


def test_splitter_adds_section_parent_metadata() -> None:
    documents = [
        {
            "text": "Section 1 Overview\n\nCandidates get training.\n\nSection 2 Benefits\n\nCandidates get allowance.",
            "metadata": {"file_id": "doc", "file_name": "sample.txt", "page": None},
        }
    ]

    chunks = TextSplitter(chunk_size=80, chunk_overlap=0).split(documents)

    assert chunks[0]["metadata"]["section_title"] == "Section 1 Overview"
    assert chunks[0]["metadata"]["parent_id"] == "section_0"
    assert "Candidates get training." in chunks[0]["metadata"]["parent_text"]
    assert chunks[-1]["metadata"]["section_title"] == "Section 2 Benefits"


def test_splitter_detects_common_academic_headings(monkeypatch) -> None:
    monkeypatch.setenv("DOCUMENT_PROFILE", "academic")
    documents = [
        {
            "text": "Introduction\n\nOpening text.\n\nConcluding Remarks\n\nFinal text.",
            "metadata": {"file_id": "paper", "file_name": "paper.pdf", "page": 40},
        }
    ]

    chunks = TextSplitter(chunk_size=80, chunk_overlap=0).split(documents)

    assert chunks[0]["metadata"]["section_title"] == "Introduction"
    assert chunks[-1]["metadata"]["section_title"] == "Concluding Remarks"


def test_splitter_detects_data_sources_heading() -> None:
    documents = [
        {
            "text": "Data Sources\n\nThe paper uses survey and administrative records.",
            "metadata": {"file_id": "paper", "file_name": "paper.pdf", "page": 8},
        }
    ]

    chunks = TextSplitter(chunk_size=120, chunk_overlap=0).split(documents)

    assert chunks[0]["metadata"]["section_title"] == "Data Sources"


def test_splitter_does_not_treat_numbered_sentences_as_headings() -> None:
    documents = [
        {
            "text": (
                "Results\n\n"
                "70. I also show in Table B.2 that estimates are larger for some cities, "
                "although this relationship is weaker than the formal sector results."
            ),
            "metadata": {"file_id": "paper", "file_name": "paper.pdf", "page": 29},
        }
    ]

    chunks = TextSplitter(chunk_size=220, chunk_overlap=0).split(documents)

    assert chunks
    assert all(
        "70. I also show" not in chunk["metadata"]["section_title"]
        for chunk in chunks
    )


def test_splitter_detects_numbered_heading_split_across_lines() -> None:
    documents = [
        {
            "text": "8\nResults\n\nMain results.\n\n9\nConcluding Remarks\n\nFinal text.",
            "metadata": {"file_id": "paper", "file_name": "paper.pdf", "page": 34},
        }
    ]

    chunks = TextSplitter(chunk_size=80, chunk_overlap=0).split(documents)

    assert chunks[0]["metadata"]["section_title"] == "8 Results"
    assert chunks[-1]["metadata"]["section_title"] == "9 Concluding Remarks"


def test_splitter_detects_heading_without_blank_line_boundaries() -> None:
    documents = [
        {
            "text": (
                "Previous section final sentence.\n"
                "9\n"
                "Concluding Remarks\n"
                "This paper studies minimum wage effects in Colombia.\n"
                "The final section discusses wages and employment."
            ),
            "metadata": {"file_id": "paper", "file_name": "paper.pdf", "page": 34},
        }
    ]

    chunks = TextSplitter(chunk_size=180, chunk_overlap=0).split(documents)

    assert chunks[0]["metadata"]["section_title"] == "Document"
    assert chunks[-1]["metadata"]["section_title"] == "9 Concluding Remarks"
    assert "minimum wage effects" in chunks[-1]["metadata"]["parent_text"]
