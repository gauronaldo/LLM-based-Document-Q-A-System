from app.query_intent import (
    INTENT_COMPARISON,
    INTENT_EXPLANATION,
    INTENT_EXTRACTION,
    INTENT_QA,
    INTENT_SUMMARY,
    detect_query_intent,
    is_vietnamese_query,
    retrieval_plan_for_intent,
)


def test_detect_query_intent() -> None:
    assert detect_query_intent("Summarize this document") == INTENT_SUMMARY
    assert detect_query_intent("Compare these two requirements") == INTENT_COMPARISON
    assert detect_query_intent("Explain this rule") == INTENT_EXPLANATION
    assert detect_query_intent("List the requirements") == INTENT_EXTRACTION
    assert detect_query_intent("What is the deadline?") == INTENT_QA


def test_is_vietnamese_query() -> None:
    assert is_vietnamese_query("Sinh vien can gi?")
    assert is_vietnamese_query("Sinh vi\u00ean c\u1ea7n g\u00ec?")
    assert not is_vietnamese_query("What is the deadline?")


def test_retrieval_plan_for_intent() -> None:
    summary_plan = retrieval_plan_for_intent(INTENT_SUMMARY, 5, 0.3)
    assert summary_plan.top_k == 8
    assert summary_plan.similarity_threshold == 0.0

    explain_plan = retrieval_plan_for_intent(INTENT_EXPLANATION, 5, 0.3)
    assert explain_plan.top_k == 7
    assert explain_plan.similarity_threshold == 0.2

    qa_plan = retrieval_plan_for_intent(INTENT_QA, 5, 0.3)
    assert qa_plan.top_k is None
    assert qa_plan.similarity_threshold is None
