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
