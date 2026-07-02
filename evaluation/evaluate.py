"""Unified core evaluation for the document RAG chatbot."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config import get_config
from app.keyword_search import lexical_similarity
from app.prompt_template import REFUSAL_EN, REFUSAL_VI
from app.rag_pipeline import create_default_pipeline


CITATION_GROUP_PATTERN = re.compile(r"\[([0-9,\s]+)\]")


CORE_METRICS = (
    "context_precision",
    "context_recall",
    "answer_relevancy",
    "faithfulness",
    "citation_accuracy",
    "refusal_accuracy",
    "average_latency_seconds",
)


@dataclass(frozen=True)
class EvaluationQuestion:
    """One benchmark question used by the unified evaluator."""

    question: str
    difficulty: str
    expected_answer: str
    source_file: str
    source_pages: tuple[int, ...]
    expected_keywords: tuple[str, ...]
    is_answerable: bool


@dataclass(frozen=True)
class CoreEvalResult:
    """One generated RAG answer plus core evaluation scores."""

    question: str
    difficulty: str
    is_answerable: bool
    expected_answer: str
    answer: str
    status: str
    error: str
    attempts: int
    latency_seconds: float
    retrieved_contexts: list[str]
    retrieved_sources: list[dict[str, Any]]
    sources: list[dict[str, Any]]
    context_precision: float | None = None
    context_recall: float | None = None
    answer_relevancy: float | None = None
    faithfulness: float | None = None
    citation_accuracy: float | None = None
    refusal_accuracy: float | None = None
    metric_errors: dict[str, str] | None = None


def load_questions(path: Path) -> list[EvaluationQuestion]:
    """Load benchmark questions from CSV."""

    questions: list[EvaluationQuestion] = []
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            questions.append(
                EvaluationQuestion(
                    question=row["question"].strip(),
                    difficulty=row.get("difficulty", "medium").strip() or "medium",
                    expected_answer=row["expected_answer"].strip(),
                    source_file=row["source_file"].strip(),
                    source_pages=_parse_pages(row),
                    expected_keywords=_parse_keywords(row.get("expected_keywords", "")),
                    is_answerable=_parse_bool(row["is_answerable"]),
                )
            )
    return questions


def run_core_evaluation(
    questions: list[EvaluationQuestion],
    documents_dir: Path,
    top_k: int,
    similarity_threshold: float,
    collection_name: str,
    persist_directory: Path,
    judge_model: str,
    judge_provider: str,
    sleep_seconds: float = 12.0,
    max_retries: int = 3,
    limit: int = 0,
    include_unanswerable_in_judge: bool = False,
    verbose: bool = True,
) -> list[CoreEvalResult]:
    """Generate answers and compute the core metrics in one pass."""

    selected_questions = questions[:limit] if limit > 0 else questions
    if verbose:
        print(f"Preparing RAGAS evaluator: questions={len(selected_questions)}")

    config = replace(
        get_config(),
        chroma_collection_name=collection_name,
        chroma_persist_dir=persist_directory,
        top_k=top_k,
        similarity_threshold=similarity_threshold,
    )
    pipeline = create_default_pipeline(
        config=config,
        top_k=top_k,
        similarity_threshold=similarity_threshold,
    )
    if verbose:
        print("Resetting evaluation vector store...")
    pipeline.reset()
    if verbose:
        print("Ingesting benchmark documents...")
    _ingest_benchmark_documents(pipeline, selected_questions, documents_dir)

    if verbose:
        print(f"Loading RAGAS judge: {judge_model}")
    judge = _build_ragas_judge(model_name=judge_model, provider=judge_provider)

    results: list[CoreEvalResult] = []
    for index, question in enumerate(selected_questions):
        if verbose:
            print(f"[{index + 1}/{len(selected_questions)}] Generating answer: {question.question}")
        response = None
        last_error: Exception | None = None
        attempts = 0
        started_at = time.perf_counter()

        for attempt in range(max_retries + 1):
            attempts = attempt + 1
            try:
                response = pipeline.answer_question(question.question, chat_history=[])
                last_error = None
                break
            except Exception as exc:
                last_error = exc
                if attempt >= max_retries or not _is_retryable_error(exc):
                    break
                time.sleep(_retry_sleep_seconds(exc, sleep_seconds))

        latency_seconds = time.perf_counter() - started_at
        if last_error is not None:
            result = _error_result(question, last_error, attempts, latency_seconds)
        else:
            result = _score_local_metrics(question, response or {}, attempts, latency_seconds)
            if _should_run_judge(question, result, include_unanswerable_in_judge):
                if verbose:
                    print(f"[{index + 1}/{len(selected_questions)}] Scoring judge metrics...")
                result = _with_judge_metrics(result, judge)

        results.append(result)
        if verbose:
            if result.error:
                print(f"[{index + 1}/{len(selected_questions)}] Error: {_shorten(result.error, 220)}")
            if result.metric_errors:
                metric_error_summary = "; ".join(
                    f"{name}: {_shorten(error, 120)}" for name, error in result.metric_errors.items()
                )
                if metric_error_summary:
                    print(
                        f"[{index + 1}/{len(selected_questions)}] Metric errors: "
                        f"{_shorten(metric_error_summary, 260)}"
                    )
            print(
                f"[{index + 1}/{len(selected_questions)}] Done "
                f"status={result.status} latency={result.latency_seconds:.2f}s"
            )

        if sleep_seconds > 0 and index < len(selected_questions) - 1:
            if verbose:
                print(f"Sleeping {sleep_seconds:.1f}s to avoid rate limits...")
            time.sleep(sleep_seconds)

    return results


def write_core_results(results: list[CoreEvalResult], output_path: Path) -> None:
    """Write row-level core evaluation results to CSV."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "question",
                "difficulty",
                "is_answerable",
                "expected_answer",
                "answer",
                "status",
                "error",
                "attempts",
                "latency_seconds",
                "context_precision",
                "context_recall",
                "answer_relevancy",
                "faithfulness",
                "citation_accuracy",
                "refusal_accuracy",
                "metric_errors_json",
                "retrieved_sources_json",
                "retrieved_contexts_json",
                "sources_json",
            ],
        )
        writer.writeheader()
        for result in results:
            writer.writerow(
                {
                    "question": result.question,
                    "difficulty": result.difficulty,
                    "is_answerable": result.is_answerable,
                    "expected_answer": result.expected_answer,
                    "answer": result.answer,
                    "status": result.status,
                    "error": result.error,
                    "attempts": result.attempts,
                    "latency_seconds": f"{result.latency_seconds:.4f}",
                    "context_precision": _optional_score(result.context_precision),
                    "context_recall": _optional_score(result.context_recall),
                    "answer_relevancy": _optional_score(result.answer_relevancy),
                    "faithfulness": _optional_score(result.faithfulness),
                    "citation_accuracy": _optional_score(result.citation_accuracy),
                    "refusal_accuracy": _optional_score(result.refusal_accuracy),
                    "metric_errors_json": json.dumps(result.metric_errors or {}, ensure_ascii=False),
                    "retrieved_sources_json": json.dumps(result.retrieved_sources, ensure_ascii=False),
                    "retrieved_contexts_json": json.dumps(result.retrieved_contexts, ensure_ascii=False),
                    "sources_json": json.dumps(result.sources, ensure_ascii=False),
                }
            )


def write_core_report(
    results: list[CoreEvalResult],
    output_path: Path,
    judge_model: str,
) -> None:
    """Write a compact Markdown report with only the core metrics."""

    summary = summarize_core_results(results)
    lines = [
        "# Core RAG Evaluation Results",
        "",
        "Framework: `ragas`",
        f"Judge model: `{judge_model}`",
        f"Total questions: {len(results)}",
        f"Successful generations: {sum(1 for result in results if result.status == 'success')}",
        f"Answerable questions: {sum(1 for result in results if result.is_answerable)}",
        f"Unanswerable questions: {sum(1 for result in results if not result.is_answerable)}",
        f"Easy questions: {sum(1 for result in results if result.difficulty == 'easy')}",
        f"Medium questions: {sum(1 for result in results if result.difficulty == 'medium')}",
        f"Hard questions: {sum(1 for result in results if result.difficulty == 'hard')}",
        "",
        "## Core Metrics",
        "",
        f"- Context Precision: {_format_percent(summary['context_precision'])}",
        f"- Context Recall: {_format_percent(summary['context_recall'])}",
        f"- Answer Relevancy: {_format_percent(summary['answer_relevancy'])}",
        f"- Faithfulness: {_format_percent(summary['faithfulness'])}",
        f"- Citation Accuracy: {_format_percent(summary['citation_accuracy'])}",
        f"- Refusal Accuracy: {_format_percent(summary['refusal_accuracy'])}",
        f"- Average Latency: {summary['average_latency_seconds']:.2f}s",
        "",
        "## Per-question Results",
        "",
        "| # | Difficulty | Type | Latency | CtxPrec | CtxRecall | AnsRel | Faith | Cite | Refusal | Error | Question | Answer |",
        "|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|---|",
    ]

    for index, result in enumerate(results, start=1):
        lines.append(
            "| {index} | {difficulty} | {type} | {latency:.2f}s | {context_precision} | {context_recall} | {answer_relevancy} | {faithfulness} | {citation_accuracy} | {refusal_accuracy} | {error} | {question} | {answer} |".format(
                index=index,
                difficulty=result.difficulty,
                type="answerable" if result.is_answerable else "unanswerable",
                latency=result.latency_seconds,
                context_precision=_format_percent(result.context_precision),
                context_recall=_format_percent(result.context_recall),
                answer_relevancy=_format_percent(result.answer_relevancy),
                faithfulness=_format_percent(result.faithfulness),
                citation_accuracy=_format_percent(result.citation_accuracy),
                refusal_accuracy=_format_percent(result.refusal_accuracy),
                error=_escape_markdown_table(_shorten(result.error, 80)),
                question=_escape_markdown_table(result.question),
                answer=_escape_markdown_table(_shorten(result.answer, 180)),
            )
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def summarize_core_results(results: list[CoreEvalResult]) -> dict[str, float | None]:
    """Compute the seven required project metrics."""

    successful = [result for result in results if result.status == "success"]
    return {
        "context_precision": _mean_optional(result.context_precision for result in results),
        "context_recall": _mean_optional(result.context_recall for result in results),
        "answer_relevancy": _mean_optional(result.answer_relevancy for result in results),
        "faithfulness": _mean_optional(result.faithfulness for result in results),
        "citation_accuracy": _mean_optional(result.citation_accuracy for result in results),
        "refusal_accuracy": _mean_optional(result.refusal_accuracy for result in results),
        "average_latency_seconds": (
            sum(result.latency_seconds for result in successful) / len(successful)
            if successful
            else 0.0
        ),
    }


def _ingest_benchmark_documents(
    pipeline: Any,
    questions: list[EvaluationQuestion],
    documents_dir: Path,
) -> None:
    source_files = sorted({question.source_file for question in questions if question.source_file})
    for source_file in source_files:
        source_path = documents_dir / source_file
        if not source_path.exists():
            raise FileNotFoundError(f"Missing evaluation source document: {source_path}")
        pipeline.ingest_document(
            file_path=str(source_path),
            file_name=source_file,
            file_id=source_path.stem,
        )


def _score_local_metrics(
    question: EvaluationQuestion,
    response: dict[str, Any],
    attempts: int,
    latency_seconds: float,
) -> CoreEvalResult:
    answer = response.get("answer", "")
    retrieved_chunks = response.get("retrieved_chunks", [])
    retrieved_contexts = [chunk.get("text", "") for chunk in retrieved_chunks]

    citation_accuracy = None
    refusal_accuracy = None
    if question.is_answerable:
        citation_accuracy = _score_citation_accuracy(question, answer, retrieved_chunks)
    else:
        refusal_accuracy = 1.0 if _looks_like_refusal(answer) else 0.0

    return CoreEvalResult(
        question=question.question,
        difficulty=question.difficulty,
        is_answerable=question.is_answerable,
        expected_answer=question.expected_answer,
        answer=answer,
        status="success",
        error="",
        attempts=attempts,
        latency_seconds=latency_seconds,
        retrieved_contexts=retrieved_contexts,
        retrieved_sources=_format_retrieved_sources(retrieved_chunks),
        sources=response.get("sources", []),
        citation_accuracy=citation_accuracy,
        refusal_accuracy=refusal_accuracy,
        metric_errors={},
    )


def _score_citation_accuracy(
    question: EvaluationQuestion,
    answer: str,
    retrieved_chunks: list[dict[str, Any]],
) -> float:
    cited_ids = _cited_source_ids(answer)
    if not cited_ids or not retrieved_chunks:
        return 0.0

    cited_texts = []
    for source_id in cited_ids:
        chunk_index = source_id - 1
        if 0 <= chunk_index < len(retrieved_chunks):
            cited_texts.append(retrieved_chunks[chunk_index].get("text", ""))

    cited_context = " ".join(cited_texts)
    if not cited_context.strip():
        return 0.0

    return 1.0 if _keyword_coverage(question.expected_keywords, cited_context) >= 0.5 else 0.0


def _error_result(
    question: EvaluationQuestion,
    exc: Exception,
    attempts: int,
    latency_seconds: float,
) -> CoreEvalResult:
    return CoreEvalResult(
        question=question.question,
        difficulty=question.difficulty,
        is_answerable=question.is_answerable,
        expected_answer=question.expected_answer,
        answer="",
        status="error",
        error=f"{type(exc).__name__}: {exc}",
        attempts=attempts,
        latency_seconds=latency_seconds,
        retrieved_contexts=[],
        retrieved_sources=[],
        sources=[],
        citation_accuracy=0.0 if question.is_answerable else None,
        refusal_accuracy=0.0 if not question.is_answerable else None,
        metric_errors={},
    )


def _should_run_judge(
    question: EvaluationQuestion,
    result: CoreEvalResult,
    include_unanswerable: bool,
) -> bool:
    if result.status != "success":
        return False
    if not result.retrieved_contexts:
        return False
    return question.is_answerable or include_unanswerable


def _with_judge_metrics(
    result: CoreEvalResult,
    judge: Callable[[CoreEvalResult], tuple[dict[str, float | None], dict[str, str]]],
) -> CoreEvalResult:
    scores, errors = judge(result)
    return CoreEvalResult(
        **{
            **result.__dict__,
            "context_precision": scores.get("context_precision"),
            "context_recall": scores.get("context_recall"),
            "answer_relevancy": scores.get("answer_relevancy"),
            "faithfulness": scores.get("faithfulness"),
            "metric_errors": errors,
        }
    )


def _build_ragas_judge(
    model_name: str,
    provider: str,
) -> Callable[[CoreEvalResult], tuple[dict[str, float | None], dict[str, str]]]:
    _patch_ragas_vertexai_import()
    try:
        from ragas.embeddings.base import embedding_factory
        from ragas.llms import llm_factory
        from ragas.metrics.collections import (
            AnswerRelevancy,
            ContextPrecision,
            ContextRecall,
            Faithfulness,
        )
    except ImportError as exc:
        raise RuntimeError(_missing_framework_message("RAGAS", exc)) from exc

    provider = _normalize_ragas_provider(provider)
    _prepare_google_api_key(provider)
    llm_client = _ragas_llm_client(provider=provider, model_name=model_name)
    llm = llm_factory(
        model_name,
        provider=_ragas_llm_factory_provider(provider),
        client=llm_client,
        adapter="instructor" if provider == "ollama" else "auto",
        max_tokens=_ragas_judge_max_tokens(provider),
    )
    embeddings = _ragas_embedding_factory(embedding_factory, provider=provider)
    scorers = {
        "context_precision": ContextPrecision(llm=llm),
        "context_recall": ContextRecall(llm=llm),
        "answer_relevancy": AnswerRelevancy(llm=llm, embeddings=embeddings),
        "faithfulness": Faithfulness(llm=llm),
    }

    def judge(result: CoreEvalResult) -> tuple[dict[str, float | None], dict[str, str]]:
        scores: dict[str, float | None] = {}
        errors: dict[str, str] = {}
        for metric_name, scorer in scorers.items():
            try:
                scores[metric_name] = _as_optional_float(
                    _score_ragas_metric(metric_name, scorer, result)
                )
            except Exception as exc:
                scores[metric_name] = None
                errors[metric_name] = f"{type(exc).__name__}: {exc}"
        return scores, errors

    return judge


def _normalize_ragas_provider(provider: str) -> str:
    provider = provider.lower()
    if provider in {"google", "gemini"}:
        return "google"
    if provider in {"openai", "ollama"}:
        return provider
    raise RuntimeError(
        f"Unsupported RAGAS judge provider: {provider}. "
        "Use `google`, `gemini`, `openai`, or `ollama`."
    )


def _ragas_llm_factory_provider(provider: str) -> str:
    """Return the provider name RAGAS/instructor should use internally."""

    provider = _normalize_ragas_provider(provider)
    if provider == "ollama":
        return "openai"
    return provider


def _ragas_judge_max_tokens(provider: str) -> int:
    default = "4096" if _normalize_ragas_provider(provider) == "ollama" else "2048"
    value = os.getenv("RAGAS_JUDGE_MAX_TOKENS", default)
    try:
        return max(1024, int(value))
    except ValueError:
        return int(default)


def _score_ragas_metric(metric_name: str, scorer: Any, result: CoreEvalResult) -> Any:
    if metric_name == "context_precision":
        score = scorer.score(
            user_input=result.question,
            reference=result.expected_answer,
            retrieved_contexts=result.retrieved_contexts,
        )
    elif metric_name == "context_recall":
        score = scorer.score(
            user_input=result.question,
            reference=result.expected_answer,
            retrieved_contexts=result.retrieved_contexts,
        )
    elif metric_name == "answer_relevancy":
        score = scorer.score(user_input=result.question, response=result.answer)
    else:
        score = scorer.score(
            user_input=result.question,
            response=result.answer,
            retrieved_contexts=result.retrieved_contexts,
        )
    return getattr(score, "value", score)


def _patch_ragas_vertexai_import() -> None:
    """Provide a compatibility shim for RAGAS 0.4.x and LangChain 1.x."""

    module_name = "langchain_community.chat_models.vertexai"
    if module_name in sys.modules:
        return

    try:
        from langchain_community.llms import VertexAI
    except ImportError:
        return

    import types

    vertexai_module = types.ModuleType(module_name)
    vertexai_module.ChatVertexAI = VertexAI
    sys.modules[module_name] = vertexai_module


def _missing_framework_message(framework_name: str, exc: ImportError) -> str:
    missing_name = getattr(exc, "name", "") or str(exc)
    if "langchain_community.chat_models.vertexai" in missing_name:
        return (
            "RAGAS is installed, but a LangChain VertexAI compatibility dependency is missing. "
            "Run `python -m pip install -r requirements.txt --upgrade`, then retry. "
            "If the issue persists, pin a RAGAS/LangChain-compatible set."
        )

    return (
        f"{framework_name} or one of its dependencies is not installed. "
        "Run `python -m pip install -r requirements.txt --upgrade`, then retry."
    )


def _ragas_llm_client(provider: str, model_name: str) -> Any:
    provider = _normalize_ragas_provider(provider)
    if provider in {"google", "gemini"}:
        try:
            from google import genai
        except ImportError as exc:
            raise RuntimeError(
                "RAGAS with Gemini requires google-genai. "
                "Run `python -m pip install -r requirements.txt --upgrade`, then retry."
            ) from exc

        api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY or GOOGLE_API_KEY is required for RAGAS Gemini evaluation.")
        return genai.Client(api_key=api_key).aio

    if provider == "openai":
        from openai import AsyncOpenAI

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is required for RAGAS OpenAI evaluation.")
        return AsyncOpenAI(api_key=api_key)

    if provider == "ollama":
        from openai import AsyncOpenAI

        base_url = os.getenv("OLLAMA_OPENAI_BASE_URL", "http://localhost:11434/v1")
        api_key = os.getenv("OLLAMA_API_KEY", "ollama")
        return AsyncOpenAI(base_url=base_url, api_key=api_key)

    raise RuntimeError(
        f"Unsupported RAGAS judge provider: {provider}. "
        "Use `google`, `gemini`, `openai`, or `ollama`."
    )


def _ragas_embedding_factory(embedding_factory: Callable[..., Any], provider: str) -> Any:
    embedding_provider = os.getenv(
        "RAGAS_EMBEDDING_PROVIDER",
        _default_ragas_embedding_provider(provider),
    ).lower()

    if embedding_provider == "google":
        from google import genai

        api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY or GOOGLE_API_KEY is required for RAGAS Google embeddings.")
        model = os.getenv("RAGAS_EMBEDDING_MODEL", "gemini-embedding-001")
        return embedding_factory("google", model=model, client=genai.Client(api_key=api_key).aio)

    if embedding_provider == "openai":
        from openai import AsyncOpenAI

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is required for RAGAS OpenAI embeddings.")
        model = os.getenv("RAGAS_EMBEDDING_MODEL", "text-embedding-3-small")
        return embedding_factory("openai", model=model, client=AsyncOpenAI(api_key=api_key))

    if embedding_provider == "huggingface":
        model = os.getenv("RAGAS_EMBEDDING_MODEL") or _default_huggingface_embedding_model()
        return embedding_factory("huggingface", model=model)

    model = os.getenv("RAGAS_EMBEDDING_MODEL")
    return embedding_factory(embedding_provider, model=model)


def _default_ragas_embedding_provider(provider: str) -> str:
    provider = _normalize_ragas_provider(provider)
    if provider == "google":
        return "google"
    if provider == "ollama":
        return "huggingface"
    return "openai"


def _default_huggingface_embedding_model() -> str:
    configured_model = os.getenv("EMBEDDING_MODEL", "")
    if configured_model and configured_model.strip().lower() not in {
        "local-hash",
        "local_hash",
        "hash",
        "hashing",
    }:
        return configured_model
    return "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def _prepare_google_api_key(provider: str) -> None:
    provider = _normalize_ragas_provider(provider)
    if provider != "google" or os.getenv("GOOGLE_API_KEY"):
        return
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    if gemini_api_key:
        os.environ["GOOGLE_API_KEY"] = gemini_api_key


def _cited_source_ids(answer: str) -> tuple[int, ...]:
    source_ids = []
    for match in CITATION_GROUP_PATTERN.finditer(answer):
        for part in match.group(1).split(","):
            part = part.strip()
            if part.isdigit():
                source_ids.append(int(part))
    return tuple(dict.fromkeys(source_ids))


def _keyword_coverage(expected_keywords: tuple[str, ...], answer: str) -> float:
    if not expected_keywords or not answer.strip():
        return 0.0

    matches = 0
    for keyword in expected_keywords:
        if _keyword_matches(keyword, answer):
            matches += 1
    return matches / len(expected_keywords)


def _keyword_matches(keyword: str, text: str) -> bool:
    keyword = keyword.strip()
    if not keyword:
        return False
    if keyword.lower() in text.lower():
        return True
    return lexical_similarity(keyword, text) >= 0.6


def _looks_like_refusal(answer: str) -> bool:
    normalized = " ".join(answer.lower().split())
    refusal_markers = [
        REFUSAL_EN.lower(),
        REFUSAL_VI.lower(),
        "could not find",
        "cannot find",
        "not enough information",
        "insufficient information",
        "khong tim thay",
        "khong co thong tin",
        "khong du thong tin",
    ]
    return any(marker in normalized for marker in refusal_markers)


def _is_retryable_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return any(marker in message for marker in ("429", "quota", "rate limit", "retry"))


def _retry_sleep_seconds(exc: Exception, default_sleep_seconds: float) -> float:
    message = str(exc)
    match = re.search(r"retry_delay\s*\{\s*seconds:\s*(\d+)", message)
    if match:
        return max(float(match.group(1)) + 1.0, default_sleep_seconds)

    match = re.search(r"retry in\s+([0-9.]+)s", message, flags=re.IGNORECASE)
    if match:
        return max(float(match.group(1)) + 1.0, default_sleep_seconds)

    return default_sleep_seconds


def _format_retrieved_sources(retrieved_chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    sources = []
    for chunk in retrieved_chunks:
        metadata = chunk.get("metadata", {})
        sources.append(
            {
                "file_name": metadata.get("file_name", "unknown"),
                "page": metadata.get("page"),
                "chunk_id": chunk.get("chunk_id", "unknown"),
                "score": round(float(chunk.get("score", 0.0)), 4),
            }
        )
    return sources


def _mean_optional(values: Any) -> float | None:
    scores = [float(value) for value in values if value is not None]
    return sum(scores) / len(scores) if scores else None


def _as_optional_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _optional_score(value: float | None) -> str:
    return "" if value is None else f"{value:.4f}"


def _format_percent(value: float | None) -> str:
    return "" if value is None else f"{value:.1%}"


def _shorten(value: str, max_length: int) -> str:
    value = " ".join(value.split())
    if len(value) <= max_length:
        return value
    return value[: max_length - 3].rstrip() + "..."


def _escape_markdown_table(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")


def _parse_bool(value: str) -> bool:
    return value.strip().lower() in {"true", "1", "yes", "y"}


def _parse_pages(row: dict[str, str]) -> tuple[int, ...]:
    value = row.get("source_pages") or row.get("source_page", "")
    pages = []
    for part in str(value).replace(",", ";").split(";"):
        page = _parse_optional_int(part)
        if page is not None:
            pages.append(page)
    return tuple(sorted(set(pages)))


def _parse_keywords(value: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in str(value).split(";") if part.strip())


def _parse_optional_int(value: str) -> int | None:
    value = str(value).strip()
    if not value:
        return None
    return int(float(value))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the RAG app with core metrics only.")
    parser.add_argument("--questions", type=Path, default=Path("evaluation/questions.csv"))
    parser.add_argument("--documents-dir", type=Path, default=Path("data/samples"))
    parser.add_argument("--output", type=Path, default=Path("evaluation/results.csv"))
    parser.add_argument("--report-output", type=Path, default=Path("evaluation/results.md"))
    parser.add_argument("--judge-model", default=os.getenv("EVAL_JUDGE_MODEL", "gemini-2.5-flash"))
    parser.add_argument("--judge-provider", default=os.getenv("EVAL_JUDGE_PROVIDER", "google"))
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--similarity-threshold", type=float, default=0.3)
    parser.add_argument("--collection-name", default="document_qa_core_evaluation")
    parser.add_argument("--persist-directory", type=Path, default=Path("vector_db/core_evaluation"))
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--sleep-seconds", type=float, default=12.0)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--include-unanswerable-in-judge", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    questions = load_questions(args.questions)
    results = run_core_evaluation(
        questions=questions,
        documents_dir=args.documents_dir,
        top_k=args.top_k,
        similarity_threshold=args.similarity_threshold,
        collection_name=args.collection_name,
        persist_directory=args.persist_directory,
        judge_model=args.judge_model,
        judge_provider=args.judge_provider,
        sleep_seconds=args.sleep_seconds,
        max_retries=args.max_retries,
        limit=args.limit,
        include_unanswerable_in_judge=args.include_unanswerable_in_judge,
        verbose=not args.quiet,
    )
    write_core_results(results, args.output)
    write_core_report(
        results,
        args.report_output,
        judge_model=args.judge_model,
    )
    print(f"Wrote core evaluation rows to {args.output}")
    print(f"Wrote core evaluation report to {args.report_output}")


if __name__ == "__main__":
    main()
