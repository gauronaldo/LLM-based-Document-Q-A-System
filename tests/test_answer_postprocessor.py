from app.answer_postprocessor import (
    contains_answer_content,
    looks_like_refusal,
    normalize_unsupported_claim_answer,
    postprocess_answer_behavior,
)
from app.context_support import NO_SUPPORT, PARTIAL_SUPPORT, STRONG_SUPPORT
from app.prompt_template import REFUSAL_EN


def test_postprocess_refusal_with_no_support_returns_canonical_refusal() -> None:
    answer = "I couldn't find this information in the provided document. [3]"

    assert postprocess_answer_behavior(answer, "What is missing?", NO_SUPPORT) == REFUSAL_EN


def test_postprocess_mixed_answer_removes_refusal_when_context_has_support() -> None:
    answer = (
        "The model uses survey data to study informal employment. [1] "
        "However, I could not find this information in the provided document."
    )

    processed = postprocess_answer_behavior(answer, "What data does the model use?", STRONG_SUPPORT)

    assert processed == "The model uses survey data to study informal employment. [1]"


def test_unsupported_claim_answer_starts_with_clear_statement() -> None:
    answer = "No. Informal firms also face regulatory costs. [1]"

    processed = normalize_unsupported_claim_answer(
        answer,
        "Does the paper claim informal firms never face regulatory costs?",
    )

    assert processed.startswith("The document does not support this claim.")
    assert "[1]" not in processed


def test_refusal_detector_covers_common_variants() -> None:
    assert looks_like_refusal("The provided document does not mention Bitcoin prices. [2]")
    assert looks_like_refusal("I was unable to find the answer in the context.")


def test_contains_answer_content_ignores_pure_refusal() -> None:
    assert contains_answer_content("I could not find this information in the provided document.") is False
    assert contains_answer_content("The sample covers employed workers in Brazil. [1]") is True


def test_postprocess_leaves_partial_supported_non_refusal_answer() -> None:
    answer = "The context partially supports this: workers are split by education. [1]"

    assert postprocess_answer_behavior(answer, "Why split workers?", PARTIAL_SUPPORT) == answer
