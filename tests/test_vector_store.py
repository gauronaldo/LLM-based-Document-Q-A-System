from app.vector_store import VectorStore


class FakeCollection:
    def __init__(self):
        self.upsert_payload = None
        self.upsert_payloads = []

    def upsert(self, ids, documents, embeddings, metadatas) -> None:
        self.upsert_payload = {
            "ids": ids,
            "documents": documents,
            "embeddings": embeddings,
            "metadatas": metadatas,
        }
        self.upsert_payloads.append(self.upsert_payload)

    def query(self, query_embeddings, n_results, include):
        assert query_embeddings == [[0.1, 0.2]]
        assert n_results == 2
        assert include == ["documents", "metadatas", "distances"]
        return {
            "ids": [["chunk-1"]],
            "documents": [["Nội dung chunk"]],
            "metadatas": [[{"file_name": "sample.pdf", "page": 1}]],
            "distances": [[0.2]],
        }

    def get(self, include):
        assert include == ["documents", "metadatas"]
        return {
            "ids": ["chunk-1", "chunk-2"],
            "documents": ["Ná»™i dung chunk", "Another chunk"],
            "metadatas": [{"file_name": "sample.pdf", "page": 1}, None],
        }


def test_add_chunks_upserts_chroma_payload_with_clean_metadata() -> None:
    store = VectorStore("vector_db", "test_collection")
    collection = FakeCollection()
    store._collection = collection

    store.add_chunks(
        chunks=[
            {
                "chunk_id": "chunk-1",
                "text": "Nội dung",
                "metadata": {
                    "file_name": "sample.pdf",
                    "page": 1,
                    "empty": None,
                    "tags": ["a", "b"],
                },
            }
        ],
        embeddings=[[0.1, 0.2]],
    )

    assert collection.upsert_payload == {
        "ids": ["chunk-1"],
        "documents": ["Nội dung"],
        "embeddings": [[0.1, 0.2]],
        "metadatas": [{"file_name": "sample.pdf", "page": 1, "tags": "['a', 'b']"}],
    }


def test_add_chunks_upserts_large_payload_in_batches() -> None:
    store = VectorStore("vector_db", "test_collection", upsert_batch_size=2)
    collection = FakeCollection()
    store._collection = collection

    chunks = [
        {"chunk_id": f"chunk-{index}", "text": f"text {index}", "metadata": {}}
        for index in range(5)
    ]
    embeddings = [[float(index)] for index in range(5)]

    store.add_chunks(chunks, embeddings)

    assert [payload["ids"] for payload in collection.upsert_payloads] == [
        ["chunk-0", "chunk-1"],
        ["chunk-2", "chunk-3"],
        ["chunk-4"],
    ]


def test_search_formats_results_with_scores() -> None:
    store = VectorStore("vector_db", "test_collection")
    store._collection = FakeCollection()

    results = store.search([0.1, 0.2], top_k=2)

    assert results == [
        {
            "chunk_id": "chunk-1",
            "text": "Nội dung chunk",
            "metadata": {"file_name": "sample.pdf", "page": 1},
            "score": 0.8,
            "distance": 0.2,
        }
    ]


def test_get_all_formats_chunks_for_keyword_search() -> None:
    store = VectorStore("vector_db", "test_collection")
    store._collection = FakeCollection()

    results = store.get_all()

    assert results == [
        {
            "chunk_id": "chunk-1",
            "text": "Ná»™i dung chunk",
            "metadata": {"file_name": "sample.pdf", "page": 1},
            "score": 0.0,
        },
        {
            "chunk_id": "chunk-2",
            "text": "Another chunk",
            "metadata": {},
            "score": 0.0,
        },
    ]


def test_add_chunks_rejects_mismatched_lengths() -> None:
    store = VectorStore("vector_db", "test_collection")

    try:
        store.add_chunks([{"chunk_id": "x", "text": "x", "metadata": {}}], [])
    except ValueError as exc:
        assert "same length" in str(exc)
    else:
        raise AssertionError("Expected ValueError")
