from app.embedding_model import EmbeddingModel


class FakeSentenceTransformer:
    def __init__(self):
        self.calls = []

    def encode(
        self,
        texts: list[str],
        convert_to_numpy: bool,
        normalize_embeddings: bool,
        show_progress_bar: bool,
    ):
        assert convert_to_numpy is True
        assert normalize_embeddings is True
        assert show_progress_bar is False
        self.calls.append(texts)
        return FakeArray([[float(len(text)), 1.0] for text in texts])


class FakeArray:
    def __init__(self, rows: list[list[float]]):
        self.rows = rows

    def __iter__(self):
        return iter(FakeEmbedding(row) for row in self.rows)


class FakeEmbedding:
    def __init__(self, row: list[float]):
        self.row = row

    def tolist(self) -> list[float]:
        return self.row


def test_encode_returns_empty_list_for_empty_input() -> None:
    assert EmbeddingModel("fake-model").encode([]) == []


def test_encode_returns_python_lists(monkeypatch) -> None:
    embedder = EmbeddingModel("fake-model")
    monkeypatch.setattr(embedder, "_load_model", lambda: FakeSentenceTransformer())

    embeddings = embedder.encode(["hello", "intern"])

    assert embeddings == [[5.0, 1.0], [6.0, 1.0]]


def test_encode_batches_large_input_and_reports_progress(monkeypatch) -> None:
    model = FakeSentenceTransformer()
    embedder = EmbeddingModel("fake-model", batch_size=2)
    progress = []
    monkeypatch.setattr(embedder, "_load_model", lambda: model)

    embeddings = embedder.encode(
        ["one", "two", "three", "four", "five"],
        progress_callback=lambda completed, total: progress.append((completed, total)),
    )

    assert model.calls == [["one", "two"], ["three", "four"], ["five"]]
    assert embeddings == [[3.0, 1.0], [3.0, 1.0], [5.0, 1.0], [4.0, 1.0], [4.0, 1.0]]
    assert progress == [(2, 5), (4, 5), (5, 5)]


def test_local_hash_embedding_is_deterministic_and_does_not_load_model(monkeypatch) -> None:
    embedder = EmbeddingModel("local-hash")
    monkeypatch.setattr(
        embedder,
        "_load_model",
        lambda: (_ for _ in ()).throw(AssertionError("should not load sentence-transformers")),
    )

    first = embedder.encode(["minimum wage data source"])[0]
    second = embedder.encode(["minimum wage data source"])[0]

    assert first == second
    assert len(first) == 384
    assert any(value != 0.0 for value in first)
