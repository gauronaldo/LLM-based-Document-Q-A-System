from pathlib import Path

from evaluation.evaluate import (
    CoreEvalResult,
    _default_huggingface_embedding_model,
    _default_ragas_embedding_provider,
    _missing_framework_message,
    _normalize_ragas_provider,
    _ragas_judge_max_tokens,
    _ragas_llm_factory_provider,
    load_questions,
    summarize_core_results,
    write_core_report,
    write_core_results,
)


def test_summarize_core_results_reports_required_metrics() -> None:
    results = [
        CoreEvalResult(
            question="Answerable?",
            difficulty="easy",
            is_answerable=True,
            expected_answer="Expected",
            answer="Answer [1]",
            status="success",
            error="",
            attempts=1,
            latency_seconds=2.0,
            retrieved_contexts=["Context"],
            retrieved_sources=[],
            sources=[],
            context_precision=0.8,
            context_recall=0.7,
            answer_relevancy=0.9,
            faithfulness=0.6,
            citation_accuracy=1.0,
            refusal_accuracy=None,
            metric_errors={},
        ),
        CoreEvalResult(
            question="Unanswerable?",
            difficulty="hard",
            is_answerable=False,
            expected_answer="I could not find this information in the provided document.",
            answer="I could not find this information in the provided document.",
            status="success",
            error="",
            attempts=1,
            latency_seconds=4.0,
            retrieved_contexts=[],
            retrieved_sources=[],
            sources=[],
            citation_accuracy=None,
            refusal_accuracy=1.0,
            metric_errors={},
        ),
    ]

    summary = summarize_core_results(results)

    assert summary["context_precision"] == 0.8
    assert summary["context_recall"] == 0.7
    assert summary["answer_relevancy"] == 0.9
    assert summary["faithfulness"] == 0.6
    assert summary["citation_accuracy"] == 1.0
    assert summary["refusal_accuracy"] == 1.0
    assert summary["average_latency_seconds"] == 3.0


def test_summarize_core_results_reports_missing_metric_as_none() -> None:
    result = CoreEvalResult(
        question="Answerable?",
        difficulty="easy",
        is_answerable=True,
        expected_answer="Expected",
        answer="Answer [1]",
        status="success",
        error="",
        attempts=1,
        latency_seconds=2.0,
        retrieved_contexts=["Context"],
        retrieved_sources=[],
        sources=[],
        citation_accuracy=1.0,
        refusal_accuracy=None,
        metric_errors={},
    )

    summary = summarize_core_results([result])

    assert summary["refusal_accuracy"] is None


def test_write_core_results_and_report(tmp_path: Path) -> None:
    result = CoreEvalResult(
        question="What is estimated?",
        difficulty="medium",
        is_answerable=True,
        expected_answer="Minimum wage effects.",
        answer="The paper estimates minimum wage effects [1].",
        status="success",
        error="",
        attempts=1,
        latency_seconds=1.25,
        retrieved_contexts=["minimum wage effects"],
        retrieved_sources=[{"file_name": "paper.pdf", "page": 1}],
        sources=[{"source_id": 1, "file_name": "paper.pdf", "page": 1}],
        context_precision=0.8,
        context_recall=0.7,
        answer_relevancy=0.9,
        faithfulness=0.6,
        citation_accuracy=1.0,
        refusal_accuracy=None,
        metric_errors={},
    )
    csv_path = tmp_path / "core_results.csv"
    report_path = tmp_path / "core_results.md"

    write_core_results([result], csv_path)
    write_core_report([result], report_path, judge_model="gemini-2.5-flash")

    csv_content = csv_path.read_text(encoding="utf-8-sig")
    report_content = report_path.read_text(encoding="utf-8")

    assert "context_precision" in csv_content
    assert "retrieved_contexts_json" in csv_content
    assert "difficulty" in csv_content
    assert "Core RAG Evaluation Results" in report_content
    assert "Framework: `ragas`" in report_content
    assert "Medium questions: 1" in report_content
    assert "Context Precision: 80.0%" in report_content
    assert "Average Latency: 1.25s" in report_content


def test_load_questions_parses_difficulty_column() -> None:
    questions = load_questions(Path("evaluation/questions.csv"))

    assert questions
    assert questions[0].difficulty in {"easy", "medium", "hard"}
    assert {question.difficulty for question in questions} == {"easy", "medium", "hard"}


def test_missing_framework_message_explains_ragas_langchain_vertexai_dependency() -> None:
    message = _missing_framework_message(
        "RAGAS",
        ModuleNotFoundError("No module named 'langchain_community.chat_models.vertexai'"),
    )

    assert "RAGAS is installed" in message
    assert "requirements.txt" in message


def test_ragas_provider_normalization_supports_ollama() -> None:
    assert _normalize_ragas_provider("gemini") == "google"
    assert _normalize_ragas_provider("openai") == "openai"
    assert _normalize_ragas_provider("ollama") == "ollama"
    assert _ragas_llm_factory_provider("ollama") == "openai"


def test_ragas_ollama_defaults_to_local_huggingface_embeddings(monkeypatch) -> None:
    monkeypatch.setenv("EMBEDDING_MODEL", "local-hash")

    assert _default_ragas_embedding_provider("ollama") == "huggingface"
    assert _default_huggingface_embedding_model() == (
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )


def test_ragas_ollama_uses_larger_default_judge_token_budget(monkeypatch) -> None:
    monkeypatch.delenv("RAGAS_JUDGE_MAX_TOKENS", raising=False)

    assert _ragas_judge_max_tokens("ollama") == 4096
    assert _ragas_judge_max_tokens("openai") == 2048
