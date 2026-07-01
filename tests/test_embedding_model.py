from app.embedding_model import EmbeddingModel


class FakeSentenceTransformer:
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

    embeddings = embedder.encode(["xin chào", "thực tập"])

    assert embeddings == [[8.0, 1.0], [8.0, 1.0]]
