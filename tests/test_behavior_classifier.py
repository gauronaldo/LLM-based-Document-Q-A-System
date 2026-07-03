from app.behavior_classifier import classify_expected_behavior


def test_classifies_yes_no_questions_as_claim_verification() -> None:
    assert classify_expected_behavior("Does the document mention API limits?") == "claim_verification"


def test_classifies_regular_questions_as_answer() -> None:
    assert classify_expected_behavior("What are the API limits?") == "answer"
