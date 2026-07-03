from pathlib import Path

import pytest

from evaluation.debug_utils import (
    DebugThresholds,
    PredictionCache,
    answer_keyword_pass,
    citation_debug,
    citation_scores,
    classify_retrieval_status,
    expected_behavior_accuracy,
    evaluation_context_support,
    generate_with_context,
    keyword_hits,
    page_hit,
    prediction_cache_key,
    reciprocal_rank,
    retrieval_debug_row,
    retrieve_for_question,
    select_questions,
    unsupported_claim_accuracy,
)
from app.context_support import NO_SUPPORT, STRONG_SUPPORT
from app.prompt_template import REFUSAL_EN
from evaluation.evaluate import EvaluationQuestion


def _question(is_answerable: bool = True) -> EvaluationQuestion:
    return EvaluationQuestion(
        question="Who are Luu Bei's sworn brothers?",
        difficulty="easy",
        expected_answer="Guan Yu and Zhang Fei.",
        source_file="tam-quoc.pdf",
        source_pages=(3,),
        expected_keywords=("Guan Yu", "Zhang Fei", "sworn brothers"),
        is_answerable=is_answerable,
    )


def test_keyword_hits_are_case_insensitive_and_unicode_stable() -> None:
    hits, total, rate = keyword_hits(
        ("Quan Vũ", "Trương Phi"),
        "quan vũ kết nghĩa với TRƯƠNG PHI.",
    )

    assert (hits, total, rate) == (2, 2, 1.0)


def test_page_hit_reads_chunk_metadata_pages() -> None:
    chunks = [{"metadata": {"page": "3"}}]

    assert page_hit((3, 4), chunks) is True
    assert page_hit((5,), chunks) is False


def test_retrieval_status_classifies_missing_answerable_context() -> None:
    status, reason = classify_retrieval_status(
        question=_question(),
        keyword_hit_count=0,
        keyword_hit_rate=0.0,
        page_hit_at_5=False,
        top_score=0.8,
        thresholds=DebugThresholds(),
    )

    assert (status, reason) == ("FAIL", "RETRIEVAL_MISS")


def test_retrieval_status_warns_when_expected_page_is_found_but_keywords_are_low() -> None:
    status, reason = classify_retrieval_status(
        question=_question(),
        keyword_hit_count=0,
        keyword_hit_rate=0.0,
        page_hit_at_5=True,
        top_score=0.8,
        thresholds=DebugThresholds(),
    )

    assert (status, reason) == ("WARNING", "PAGE_HIT_KEYWORD_LOW")


def test_retrieval_debug_row_marks_page_mismatch() -> None:
    row = retrieval_debug_row(
        _question(),
        [
            {
                "text": "Guan Yu and Zhang Fei are mentioned.",
                "score": 0.9,
                "metadata": {"page": 9},
            }
        ],
        DebugThresholds(),
    )

    assert row["status"] == "FAIL"
    assert row["failure_reason"] == "PAGE_MISMATCH"


def test_reciprocal_rank_uses_expected_page_when_available() -> None:
    chunks = [
        {"text": "Guan Yu and Zhang Fei are mentioned.", "metadata": {"page": 9}},
        {"text": "Other context.", "metadata": {"page": 3}},
    ]

    assert reciprocal_rank(_question(), chunks) == 0.5


def test_prediction_cache_round_trips_rows(tmp_path: Path) -> None:
    cache_path = tmp_path / "predictions.csv"
    cache = PredictionCache(cache_path, namespace="ns1")
    contexts = ["context one", "context two"]

    cache.set("Question?", contexts, "Answer [1].", 1.25, mode="test")
    loaded = PredictionCache(cache_path, namespace="ns1")

    assert loaded.get("Question?", contexts)["answer"] == "Answer [1]."
    assert loaded.get("Question?", contexts)["namespace"] == "ns1"
    assert PredictionCache(cache_path, namespace="ns2").get("Question?", contexts) is None
    assert prediction_cache_key("Question?", contexts, namespace="ns1") != prediction_cache_key(
        "Question?",
        contexts,
        namespace="ns2",
    )


def test_select_questions_can_return_stratified_sample(tmp_path: Path) -> None:
    csv_path = tmp_path / "questions.csv"
    csv_path.write_text(
        "\n".join(
            [
                "question,difficulty,expected_answer,source_file,source_pages,expected_keywords,is_answerable",
                "Easy answerable,easy,A,paper.pdf,1,a,true",
                "Hard answerable,hard,A,paper.pdf,2,b,true",
                "Medium answerable,medium,A,paper.pdf,3,c,true",
                "OOS easy,easy,No,paper.pdf,,x,false",
                "OOS hard,hard,No,paper.pdf,,y,false",
            ]
        ),
        encoding="utf-8",
    )

    questions = select_questions(
        csv_path,
        limit=4,
        sample_strategy="stratified",
        hard_count=1,
        unanswerable_count=1,
    )

    assert len(questions) == 4
    assert any(question.difficulty == "hard" and question.is_answerable for question in questions)
    assert any(not question.is_answerable for question in questions)


def test_citation_debug_maps_citation_ids_to_pages_and_chunks() -> None:
    chunks = [
        {
            "chunk_id": "c1",
            "text": "Guan Yu and Zhang Fei are sworn brothers.",
            "metadata": {"page": 3},
        },
        {
            "chunk_id": "c2",
            "text": "Unrelated text.",
            "metadata": {"page": 9},
        },
    ]

    debug = citation_debug(_question(), "They are Guan Yu and Zhang Fei [1].", chunks)

    assert debug["cited_source_ids"] == "1"
    assert debug["cited_pages"] == "3"
    assert debug["cited_chunk_ids"] == "c1"
    assert debug["citation_expected_page_hit"] is True
    assert debug["citation_keyword_hit_rate"] == "1.0000"


def test_citation_scores_separate_strict_page_keyword_and_weighted_scores() -> None:
    chunks = [
        {
            "chunk_id": "c1",
            "text": "Guan Yu and Zhang Fei are mentioned but not the relationship.",
            "metadata": {"page": 3},
        }
    ]

    scores = citation_scores(_question(), "They are Guan Yu and Zhang Fei [1].", chunks)

    assert scores["citation_page_accuracy"] == 1.0
    assert scores["citation_keyword_support"] == pytest.approx(2 / 3, rel=1e-3)
    assert scores["citation_weighted_score"] > 0.8
    assert scores["citation_relaxed_accuracy"] == 1.0


def test_unsupported_claim_accuracy_accepts_negative_claim_answer() -> None:
    question = EvaluationQuestion(
        question="Does the paper claim Luu Bei abolished the oath?",
        difficulty="medium",
        expected_answer="The document does not support this claim.",
        source_file="tam-quoc.pdf",
        source_pages=(),
        expected_keywords=(),
        is_answerable=False,
        question_type="unsupported_claim",
        expected_behavior="state_not_supported",
    )

    answer = "No, the paper does not claim that Luu Bei abolished the oath. [1]"

    assert unsupported_claim_accuracy(question, answer) == 1.0
    assert expected_behavior_accuracy(question, answer) == 1.0


def test_answer_keyword_pass_treats_unanswerable_refusal_as_pass() -> None:
    question = _question(is_answerable=False)

    assert answer_keyword_pass(
        question,
        "I could not find this information in the provided document.",
        DebugThresholds(),
    ) is True


def test_retrieve_for_question_handles_default_qa_plan_without_top_k_override() -> None:
    class FakePreprocessor:
        @staticmethod
        def clean_query(text):
            return text

    class FakeRetriever:
        top_k = 5
        similarity_threshold = 0.3

        def __init__(self):
            self.seen_top_k = None

        def retrieve(self, _query, top_k, **_kwargs):
            self.seen_top_k = top_k
            return []

    class FakePipeline:
        preprocessor = FakePreprocessor()

        def __init__(self):
            self.retriever = FakeRetriever()

    pipeline = FakePipeline()

    chunks, _latency = retrieve_for_question(pipeline, _question(), top_k=5)

    assert chunks == []
    assert pipeline.retriever.seen_top_k == 5


def test_evaluation_context_support_uses_gold_page_hit_for_answerable_questions() -> None:
    question = _question()
    chunks = [
        {
            "text": "Unrelated wording that still comes from the expected page.",
            "metadata": {"page": 3},
            "score": 0.1,
        }
    ]

    assert evaluation_context_support(question, chunks) == STRONG_SUPPORT


def test_evaluation_context_support_treats_refuse_behavior_as_no_support() -> None:
    question = EvaluationQuestion(
        question="Does the paper evaluate GPT?",
        difficulty="easy",
        expected_answer="Not found.",
        source_file="paper.pdf",
        source_pages=(),
        expected_keywords=(),
        is_answerable=False,
        question_type="true_out_of_scope",
        expected_behavior="refuse",
    )

    assert evaluation_context_support(question, [{"text": "GPT", "metadata": {}, "score": 0.9}]) == NO_SUPPORT


def test_generate_with_context_skips_llm_for_refuse_behavior_without_support() -> None:
    class FakeLLM:
        def __init__(self):
            self.calls = 0

        def generate(self, _prompt):
            self.calls += 1
            return "This should not be called."

    class FakePipeline:
        def __init__(self):
            self.llm = FakeLLM()

    question = EvaluationQuestion(
        question="Does the paper evaluate GPT?",
        difficulty="easy",
        expected_answer="Not found.",
        source_file="paper.pdf",
        source_pages=(),
        expected_keywords=(),
        is_answerable=False,
        question_type="true_out_of_scope",
        expected_behavior="refuse",
    )
    pipeline = FakePipeline()

    answer, _latency, cache_hit = generate_with_context(
        pipeline,
        question,
        chunks=[{"text": "Labor market informality.", "metadata": {}, "score": 0.1}],
        cache=None,
        mode="test",
    )

    assert answer == REFUSAL_EN
    assert cache_hit is False
    assert pipeline.llm.calls == 0
