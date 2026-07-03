from app.prompt_template import REFUSAL_EN, REFUSAL_VI
from app.rag_pipeline import RAGPipeline, _should_retry_refusal


class FakeDocumentLoader:
    def load(self, file_path: str, file_name: str, file_id: str):
        return [
            {
                "text": "Sinh vi\u00ean ph\u1ea3i ho\u00e0n\nth\u00e0nh 65 % t\u00edn ch\u1ec9.",
                "metadata": {"file_id": file_id, "file_name": file_name, "page": 1},
            }
        ]


class FakePreprocessor:
    def clean(self, text: str) -> str:
        return text.replace("\n", " ").replace("65 %", "65%").strip()

    def clean_query(self, query: str) -> str:
        return query.strip()


class FakeSplitter:
    def split(self, documents):
        return [
            {
                "chunk_id": "doc_page_1_chunk_0",
                "text": documents[0]["text"],
                "metadata": {**documents[0]["metadata"], "chunk_index": 0},
            }
        ]


class FakeEmbedder:
    def encode(self, texts, progress_callback=None):
        self.last_texts = texts
        if progress_callback:
            progress_callback(len(texts), len(texts))
        return [[0.1, 0.2] for _ in texts]


class FakeVectorStore:
    def __init__(self):
        self.added = None
        self.reset_called = False

    def add_chunks(self, chunks, embeddings) -> None:
        self.added = (chunks, embeddings)

    def reset(self) -> None:
        self.reset_called = True


class FakeRetriever:
    def __init__(self, results):
        self.results = results
        self.top_k = 5
        self.similarity_threshold = 0.3
        self.last_top_k = None
        self.last_similarity_threshold = None

    def retrieve(self, question: str, top_k=None, similarity_threshold=None, **kwargs):
        self.last_question = question
        self.last_top_k = top_k
        self.last_similarity_threshold = similarity_threshold
        self.last_kwargs = kwargs
        return self.results


class FakeLLM:
    def __init__(self, responses=None):
        self.prompts = []
        self.responses = responses or ["Sinh vi\u00ean c\u1ea7n ho\u00e0n th\u00e0nh 65% t\u00edn ch\u1ec9. [Source 1]"]

    def generate(self, prompt: str) -> str:
        self.prompts.append(prompt)
        if len(self.prompts) <= len(self.responses):
            return self.responses[len(self.prompts) - 1]
        return self.responses[-1]

    def generate_stream(self, prompt: str):
        response = self.generate(prompt)
        midpoint = max(1, len(response) // 2)
        yield response[:midpoint]
        yield response[midpoint:]


def make_pipeline(retrieved_chunks):
    vector_store = FakeVectorStore()
    llm = FakeLLM()
    pipeline = RAGPipeline(
        document_loader=FakeDocumentLoader(),
        preprocessor=FakePreprocessor(),
        splitter=FakeSplitter(),
        embedder=FakeEmbedder(),
        vector_store=vector_store,
        retriever=FakeRetriever(retrieved_chunks),
        llm=llm,
    )
    return pipeline, vector_store, llm


def make_pipeline_with_llm(retrieved_chunks, llm):
    vector_store = FakeVectorStore()
    pipeline = RAGPipeline(
        document_loader=FakeDocumentLoader(),
        preprocessor=FakePreprocessor(),
        splitter=FakeSplitter(),
        embedder=FakeEmbedder(),
        vector_store=vector_store,
        retriever=FakeRetriever(retrieved_chunks),
        llm=llm,
    )
    return pipeline, vector_store, llm


def test_ingest_document_stores_chunks_and_embeddings() -> None:
    pipeline, vector_store, _ = make_pipeline([])

    summary = pipeline.ingest_document("path.txt", "sample.txt", "doc")

    chunks, embeddings = vector_store.added
    assert summary == {
        "status": "success",
        "file_name": "sample.txt",
        "num_documents": 1,
        "num_chunks": 1,
    }
    assert chunks[0]["text"] == "Sinh vi\u00ean ph\u1ea3i ho\u00e0n th\u00e0nh 65% t\u00edn ch\u1ec9."
    assert embeddings == [[0.1, 0.2]]


def test_ingest_document_reports_progress() -> None:
    pipeline, _, _ = make_pipeline([])
    progress_events = []

    pipeline.ingest_document(
        "path.txt",
        "sample.txt",
        "doc",
        progress_callback=lambda message, value: progress_events.append((message, value)),
    )

    messages = [message for message, _ in progress_events]
    assert "Loading document text..." in messages
    assert "Loading embedding model and indexing 1 chunks..." in messages
    assert "Encoding and indexing chunks 1/1..." in messages
    assert progress_events[-1] == ("Document indexing complete.", 1.0)


def test_answer_question_refuses_without_retrieved_chunks_and_skips_llm() -> None:
    pipeline, _, llm = make_pipeline([])

    result = pipeline.answer_question("What is the tuition fee?")

    assert result == {"answer": REFUSAL_EN, "sources": [], "retrieved_chunks": []}
    assert llm.prompts == []


def test_answer_question_returns_answer_sources_evidence_and_intent() -> None:
    retrieved = [
        {
            "chunk_id": "doc_page_1_chunk_0",
            "text": "Sinh vi\u00ean c\u1ea7n 65% t\u00edn ch\u1ec9.",
            "metadata": {"file_name": "sample.txt", "page": 1},
            "score": 0.9,
        }
    ]
    pipeline, _, llm = make_pipeline(retrieved)

    result = pipeline.answer_question(
        "Sinh vi\u00ean c\u1ea7n bao nhi\u00eau t\u00edn ch\u1ec9?",
        chat_history=[{"role": "user", "content": "Previous question"}],
    )

    assert result["answer"] == "Sinh vi\u00ean c\u1ea7n ho\u00e0n th\u00e0nh 65% t\u00edn ch\u1ec9. [1]"
    assert result["sources"] == [
        {
            "source_id": 1,
            "file_name": "sample.txt",
            "page": 1,
            "chunk_id": "doc_page_1_chunk_0",
        }
    ]
    assert result["retrieved_chunks"] == retrieved
    assert result["intent"] == "qa"
    assert pipeline.retriever.last_kwargs["auto"] is True
    assert pipeline.retriever.last_kwargs["intent"] == "qa"
    assert "Context:" in llm.prompts[0]
    assert "Previous question" in llm.prompts[0]


def test_answer_question_streams_partial_answer() -> None:
    retrieved = [
        {
            "chunk_id": "doc_page_1_chunk_0",
            "text": "The paper studies minimum wage effects.",
            "metadata": {"file_name": "paper.pdf", "page": 1},
            "score": 0.9,
        }
    ]
    pipeline, _, _ = make_pipeline(retrieved)
    streamed = []

    result = pipeline.answer_question(
        "What does the paper study?",
        stream_callback=lambda partial: streamed.append(partial),
    )

    assert streamed
    assert result["answer"].endswith("[1]")


def test_answer_question_repairs_language_mismatch_for_english_question() -> None:
    retrieved = [
        {
            "chunk_id": "doc_page_1_chunk_0",
            "text": "The paper studies minimum wage effects.",
            "metadata": {"file_name": "paper.pdf", "page": 1},
            "score": 0.9,
        }
    ]
    llm = FakeLLM(
        responses=[
            "Theo tài liệu, bài báo nghiên cứu lương tối thiểu. [Source 1]",
            "The paper studies minimum wage effects. [1]",
        ]
    )
    pipeline, _, llm = make_pipeline_with_llm(retrieved, llm)

    result = pipeline.answer_question("What does the paper study?")

    assert result["answer"] == "The paper studies minimum wage effects. [1]"
    assert len(llm.prompts) == 2
    assert "Rewrite the answer below in English only" in llm.prompts[1]


def test_answer_question_repairs_citation_only_answer() -> None:
    retrieved = [
        {
            "chunk_id": "doc_page_1_chunk_0",
            "text": "The paper studies formal wages and informal wages.",
            "metadata": {"file_name": "paper.pdf", "page": 1},
            "score": 0.9,
        }
    ]
    llm = FakeLLM(
        responses=[
            "[1, 2]",
            "The paper studies formal wages and informal wages. [1]",
        ]
    )
    pipeline, _, llm = make_pipeline_with_llm(retrieved, llm)

    result = pipeline.answer_question("What outcomes does the paper study?")

    assert result["answer"] == "The paper studies formal wages and informal wages. [1]"
    assert len(llm.prompts) == 2
    assert "previous answer was invalid" in llm.prompts[1]
    assert "Do not return only citations" in llm.prompts[1]


def test_answer_question_repairs_meta_answer() -> None:
    retrieved = [
        {
            "chunk_id": "doc_page_1_chunk_0",
            "text": "The paper uses an unexpected real minimum wage increase.",
            "metadata": {"file_name": "paper.pdf", "page": 1},
            "score": 0.9,
        }
    ]
    llm = FakeLLM(
        responses=[
            "Note: I've answered directly first, then added citations. [1]",
            "The paper uses an unexpected real minimum wage increase. [1]",
        ]
    )
    pipeline, _, llm = make_pipeline_with_llm(retrieved, llm)

    result = pipeline.answer_question("What shock does the paper use?")

    assert result["answer"] == "The paper uses an unexpected real minimum wage increase. [1]"
    assert len(llm.prompts) == 2


def test_answer_question_retries_refusal_when_context_is_relevant() -> None:
    retrieved = [
        {
            "chunk_id": "doc_page_8_chunk_0",
            "text": (
                "A commission representing the government, firms, and unions "
                "negotiated Colombia's minimum wage."
            ),
            "metadata": {"file_name": "paper.pdf", "page": 8},
            "score": 0.9,
        }
    ]
    llm = FakeLLM(
        responses=[
            REFUSAL_EN,
            "A commission representing the government, firms, and unions determined the minimum wage. [1]",
        ]
    )
    pipeline, _, llm = make_pipeline_with_llm(retrieved, llm)

    result = pipeline.answer_question(
        "What institutional process determined Colombia's minimum wage?"
    )

    assert result["answer"] == (
        "A commission representing the government, firms, and unions determined the minimum wage. [1]"
    )
    assert len(llm.prompts) == 2
    assert "retrieved Context contains relevant evidence" in llm.prompts[1]


def test_refusal_retry_uses_moderate_evidence_not_exact_keyword_match() -> None:
    chunks = [
        {
            "chunk_id": "doc_page_8_chunk_0",
            "text": "The paper explains sample splits for workers by education level and sex.",
            "metadata": {"file_name": "paper.pdf", "page": 8},
            "base_score": 0.66,
        }
    ]

    assert _should_retry_refusal(
        "Why do the authors focus on low-education workers and split the sample by sex?",
        chunks,
    ) is True


def test_answer_question_does_not_fallback_to_refusal_when_supported_repair_fails() -> None:
    retrieved = [
        {
            "chunk_id": "doc_page_1_chunk_0",
            "text": "The paper uses an unexpected real minimum wage increase.",
            "metadata": {"file_name": "paper.pdf", "page": 1},
            "score": 0.9,
        }
    ]
    llm = FakeLLM(
        responses=[
            "[1, 2]",
            "[1] paper.pdf, Page 1",
            "The paper uses an unexpected real minimum wage increase. [1]",
        ]
    )
    pipeline, _, llm = make_pipeline_with_llm(retrieved, llm)

    result = pipeline.answer_question("What shock does the paper use?")

    assert result["answer"] == "The paper uses an unexpected real minimum wage increase. [1]"
    assert len(llm.prompts) == 3
    assert "retrieved Context contains relevant evidence" in llm.prompts[2]


def test_answer_question_vietnamese_refusal() -> None:
    pipeline, _, _ = make_pipeline([])

    result = pipeline.answer_question("Sinh vi\u00ean c\u00f3 \u0111\u01b0\u1ee3c ngh\u1ec9 60 ng\u00e0y kh\u00f4ng?")

    assert result["answer"] == REFUSAL_VI


def test_summary_request_uses_relaxed_threshold_and_more_chunks() -> None:
    retrieved = [
        {
            "chunk_id": "doc_page_1_chunk_0",
            "text": "Internship JD content.",
            "metadata": {"file_name": "jd.pdf", "page": 1},
            "score": 0.0,
        }
    ]
    pipeline, _, _ = make_pipeline(retrieved)

    pipeline.answer_question("Summarize")

    assert pipeline.retriever.last_top_k == 8
    assert pipeline.retriever.last_similarity_threshold == 0.0


def test_explanation_request_uses_expanded_retrieval() -> None:
    retrieved = [
        {
            "chunk_id": "doc_page_1_chunk_0",
            "text": "Internship JD content.",
            "metadata": {"file_name": "jd.pdf", "page": 1},
            "score": 0.2,
        }
    ]
    pipeline, _, _ = make_pipeline(retrieved)

    pipeline.answer_question("Explain the main idea")

    assert pipeline.retriever.last_top_k == 7
    assert pipeline.retriever.last_similarity_threshold == 0.2
