from app.context_support import (
    NO_SUPPORT,
    PARTIAL_SUPPORT,
    STRONG_SUPPORT,
    estimate_context_support,
)


def test_estimate_context_support_detects_strong_term_overlap() -> None:
    chunks = [
        {
            "text": "The policy simulations abolish informality and report worker welfare changes.",
            "metadata": {},
            "score": 0.4,
        }
    ]

    assert (
        estimate_context_support("What policy simulations are considered?", chunks)
        == STRONG_SUPPORT
    )


def test_estimate_context_support_uses_score_with_some_overlap_for_partial_support() -> None:
    chunks = [
        {
            "text": "The descriptive section discusses gender differences among workers.",
            "metadata": {},
            "score": 0.6,
        }
    ]

    assert (
        estimate_context_support("How does informality differ by gender?", chunks)
        == PARTIAL_SUPPORT
    )


def test_estimate_context_support_rejects_unrelated_high_level_context() -> None:
    chunks = [
        {
            "text": "The paper studies labor markets and informal employment.",
            "metadata": {},
            "score": 0.5,
        }
    ]

    assert (
        estimate_context_support("Does the paper evaluate GPT or Gemini models?", chunks)
        == NO_SUPPORT
    )
